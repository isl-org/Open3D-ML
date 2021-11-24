import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from sklearn.neighbors import KDTree
from open3d.ml.tf.ops import knn_search

from .base_model import BaseModel
from ...utils import MODEL
from ...datasets.augment import SemsegAugmentation
from ...datasets.utils import DataProcessing
from ..utils.pointnet.pointnet2_utils import furthest_point_sample_v2

tf.no_gradient("Open3DKnnSearch")


class PointTransformer(BaseModel):
    """Semantic Segmentation model. Based on PointTransformer architecture
    https://arxiv.org/pdf/2012.09164.pdf

    Uses Encoder-Decoder architecture with Transformer layers.

    Attributes:
        name: Name of model.
          Default to "PointTransformer".
        blocks: Number of Bottleneck layers.
        in_channels: Number of features(default 6).
        num_classes: Number of classes.
        voxel_size: Voxel length for subsampling.
        max_voxels: Maximum number of voxels.
        augment: dictionary for augmentation.
    """

    def __init__(self,
                 name="PointTransformer",
                 blocks=[2, 2, 2, 2, 2],
                 in_channels=6,
                 num_classes=13,
                 voxel_size=0.04,
                 max_voxels=80000,
                 augment=None,
                 **kwargs):
        super(PointTransformer, self).__init__(name=name,
                                               blocks=blocks,
                                               in_channels=in_channels,
                                               num_classes=num_classes,
                                               voxel_size=voxel_size,
                                               max_voxels=max_voxels,
                                               augment=augment,
                                               **kwargs)
        cfg = self.cfg
        self.in_channels = in_channels
        self.augmenter = SemsegAugmentation(cfg.augment)
        self.in_planes, planes = in_channels, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        block = Bottleneck

        self.encoders = []
        for i in range(5):
            self.encoders.append(
                self._make_enc(
                    block,
                    planes[i],
                    blocks[i],
                    share_planes,
                    stride=stride[i],
                    nsample=nsample[i]))  # N/1, N/4, N/16, N/64, N/256

        self.decoders = []
        for i in range(4, -1, -1):
            self.decoders.append(
                self._make_dec(block,
                               planes[i],
                               2,
                               share_planes,
                               nsample=nsample[i],
                               is_head=True if i == 4 else False))

        self.cls = tf.keras.models.Sequential(
            (layers.InputLayer(input_shape=(planes[0],)),
             layers.Dense(planes[0]),
             layers.BatchNormalization(momentum=0.9,
                                       epsilon=1e-5), layers.ReLU(),
             layers.Dense(num_classes)))

    def _make_enc(self,
                  block,
                  planes,
                  blocks,
                  share_planes=8,
                  stride=1,
                  nsample=16):
        """Private method to create encoder.

        Args:
            block: Bottleneck block consisting transformer layers.
            planes: list of feature dimension.
            blocks: Number of `block` layers.
            share_planes: Number of common planes for transformer.
            stride: stride for pooling.
            nsample: number of neighbour to sample.

        Returns:
            Returns encoder object.
        """
        encoder_inputs = [
            layers.Input(shape=(3)),
            layers.Input(shape=(self.in_planes)),
            layers.Input(shape=(0,), dtype=tf.int64)
        ]
        x = TransitionDown(self.in_planes, planes * block.expansion, stride,
                           nsample)(encoder_inputs)
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            x = block(self.in_planes,
                      self.in_planes,
                      share_planes,
                      nsample=nsample)(x)

        encoder = tf.keras.Model(inputs=encoder_inputs,
                                 outputs=x,
                                 name="encoder")

        return encoder

    def _make_dec(self,
                  block,
                  planes,
                  blocks,
                  share_planes=8,
                  nsample=16,
                  is_head=False):
        """Private method to create decoder.

        Args:
            block: Bottleneck block consisting transformer layers.
            planes: list of feature dimension.
            blocks: Number of `block` layers.
            share_planes: Number of common planes for transformer.
            nsample: number of neighbour to sample.
            is_head: bool type for head layer.

        Returns:
            Returns decoder object.
        """
        decoder = []
        decoder.append(
            TransitionUp(self.in_planes,
                         None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion

        if is_head:
            decoder_inputs = [
                layers.Input(shape=(3)),
                layers.Input(shape=(self.in_planes)),
                layers.Input(shape=(0,), dtype=tf.int64)
            ]
        else:
            decoder_inputs = [
                layers.Input(shape=(3)),
                layers.Input(shape=(planes * block.expansion)),
                layers.Input(shape=(0,), dtype=tf.int64)
            ]
        for i in range(1, blocks):
            if i == 1:
                x = block(self.in_planes,
                          self.in_planes,
                          share_planes,
                          nsample=nsample)(decoder_inputs)
            else:
                x = block(self.in_planes,
                          self.in_planes,
                          share_planes,
                          nsample=nsample)(x)

        decoder.append(
            tf.keras.Model(inputs=decoder_inputs, outputs=x, name="decoder"))

        return decoder

    def call(self, inputs, training=False):
        """Forward pass for the model.

        Args:
            inputs: A dict object for inputs with following keys
                point (tf.float32): Input pointcloud (N,3)
                feat (tf.float32): Input features (N, 3)
                row_splits (tf.int64): row splits for batches (b+1,)
            training: training mode of model.

        Returns:
            Returns the probability distribution.

        """
        points = [inputs['point']]  # (n, 3)
        feats = [inputs['feat']]  # (n, c)
        row_splits = [inputs['row_splits']]  # (b)

        feats[0] = points[0] if self.in_channels == 3 else tf.concat(
            (points[0], feats[0]), 1)

        for i in range(5):
            p, f, r = self.encoders[i]([points[i], feats[i], row_splits[i]],
                                       training=training)
            points.append(p)
            feats.append(f)
            row_splits.append(r)

        for i in range(4, -1, -1):
            if i == 4:
                feats[i + 1] = self.decoders[4 - i][1]([
                    points[i + 1], self.decoders[4 - i][0](
                        [points[i + 1], feats[i + 1], row_splits[i + 1]],
                        training=training), row_splits[i + 1]
                ],
                                                       training=training)[1]
            else:
                feats[i + 1] = self.decoders[4 - i][1]([
                    points[i + 1], self.decoders[4 - i][0](
                        [points[i + 1], feats[i + 1], row_splits[i + 1]],
                        [points[i + 2], feats[i + 2], row_splits[i + 2]],
                        training=training), row_splits[i + 1]
                ],
                                                       training=training)[1]

        feat = self.cls(feats[1], training=training)

        return feat

    def preprocess(self, data, attr):
        """Data preprocessing function.

        This function is called before training to preprocess the data from a
        dataset. It consists of subsampling pointcloud with voxelization,
        augmentation and normalizing the features.

        Args:
            data: A sample from the dataset.
            attr: The corresponding attributes.

        Returns:
            Returns the preprocessed data

        """
        cfg = self.cfg
        points = np.array(data['point'], dtype=np.float32)

        if data.get('label') is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))

        if data.get('feat') is None:
            feat = None
        else:
            feat = np.array(data['feat'], dtype=np.float32)

        if attr['split'] in ['training', 'train']:
            points, feat, labels = self.augmenter.augment(
                points, feat, labels, self.cfg.get('augment', None))

        data = dict()

        if (cfg.voxel_size):
            points_min = np.min(points, 0)
            points -= points_min

            if (feat is None):
                sub_points, sub_labels = DataProcessing.grid_subsampling(
                    points, labels=labels, grid_size=cfg.voxel_size)
                sub_feat = None
            else:
                sub_points, sub_feat, sub_labels = DataProcessing.grid_subsampling(
                    points,
                    features=feat,
                    labels=labels,
                    grid_size=cfg.voxel_size)
        else:
            sub_points, sub_feat, sub_labels = points, feat, labels

        if attr['split'] not in ['test', 'testing']:
            if cfg.max_voxels and sub_labels.shape[0] > cfg.max_voxels:
                init_idx = np.random.randint(
                    sub_labels.shape[0]
                ) if 'train' in attr['split'] else sub_labels.shape[0] // 2
                crop_idx = np.argsort(
                    np.sum(np.square(sub_points - sub_points[init_idx]),
                           1))[:cfg.max_voxels]
                if sub_feat is not None:
                    sub_points, sub_feat, sub_labels = sub_points[
                        crop_idx], sub_feat[crop_idx], sub_labels[crop_idx]
                else:
                    sub_points, sub_labels = sub_points[crop_idx], sub_labels[
                        crop_idx]

        search_tree = KDTree(sub_points)

        points_min, points_max = np.min(sub_points, 0), np.max(sub_points, 0)
        sub_points -= (points_min + points_max) / 2.0
        sub_feat /= 255.0

        data['point'] = sub_points.astype(np.float32)
        data['feat'] = sub_feat.astype(np.float32)
        data['label'] = sub_labels.astype(np.int32)
        data['search_tree'] = search_tree

        if attr['split'] in ["test", "testing"]:
            proj_inds = np.squeeze(
                search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            data['proj_inds'] = proj_inds

        return data

    def transform(self, point, feat, label, splits):
        """Transform function for the point cloud and features.

        This function is called after preprocess method by dataset generator.
        It consists of mapping data to dict.

        Args:
            point: Input pointcloud.
            feat: Input features.
            label: Input labels.
            splits: row_splits defining batches.

        Returns:
            Returns dictionary data with keys
            (point, feat, label, row_splits).

        """
        return {
            'point': point,
            'feat': feat,
            'label': label,
            'row_splits': splits
        }

    def get_batch_gen(self, dataset, steps_per_epoch=None, batch_size=1):

        def concat_batch_gen():
            iters = dataset.num_pc // batch_size
            if dataset.num_pc % batch_size:
                iters += 1

            for batch_id in range(iters):
                pc = []
                feat = []
                label = []
                splits = [0]
                start_id = batch_id * batch_size
                end_id = min(start_id + batch_size, dataset.num_pc)

                for cloud_id in range(start_id, end_id):
                    data, attr = dataset.read_data(cloud_id)
                    pc.append(data['point'])
                    feat.append(data['feat'])
                    label.append(data['label'])
                    splits.append(splits[-1] + data['point'].shape[0])

                pc = np.concatenate(pc, 0)
                feat = np.concatenate(feat, 0)
                label = np.concatenate(label, 0)
                splits = np.array(splits, dtype=np.int64)

                yield pc, feat, label, splits

        gen_func = concat_batch_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int64)
        gen_shapes = ([None, 3], [None, 3], [None], [None])

        return gen_func, gen_types, gen_shapes

    def inference_begin(self, data):
        data = self.preprocess(data, {'split': 'test'})

        self.inference_input = self.transform(
            tf.constant(data['point']), tf.constant(data['feat']),
            tf.constant(data['label']), tf.constant([0,
                                                     data['point'].shape[0]]))
        self.inference_input['proj_inds'] = data['proj_inds']

    def inference_preprocess(self):
        return self.inference_input

    def inference_end(self, results):
        results = tf.reshape(results, [-1, self.cfg.num_classes])

        m_softmax = tf.keras.layers.Softmax(axis=-1)
        results = m_softmax(results)
        results = results.numpy()

        probs = np.reshape(results, [-1, self.cfg.num_classes])
        reproj_inds = self.inference_input['proj_inds']
        probs = probs[reproj_inds]

        pred_l = np.argmax(probs, 1)

        self.inference_result = {
            'predict_labels': pred_l,
            'predict_scores': probs
        }

        return True

    def get_loss(self, sem_seg_loss, results, inputs):
        """Calculate the loss on output of the model.

        Args:
            sem_seg_loss: Object of type `SemSegLoss`.
            results: Output of the model.
            inputs: Input of the model.
            device: device(cpu or cuda).

        Returns:
            Returns loss, labels and scores.
        """
        labels = inputs['label']
        scores, labels = sem_seg_loss.filter_valid_label(results, labels)
        loss = sem_seg_loss.weighted_CrossEntropyLoss(scores, labels)

        return loss, labels, scores

    def get_optimizer(self, cfg_pipeline):
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg_pipeline.optimizer.learning_rate)

        return optimizer


MODEL._register_module(PointTransformer, 'tf')


class Transformer(layers.Layer):
    """Transformer layer of the model, uses self attention."""

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        """Constructor for Transformer Layer.

        Args:
            in_planes (int): Number of input planes.
            out_planes (int): Number of output planes.
            share_planes (int): Number of shared planes.
            nsample (int): Number of neighbours.
        """
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample

        self.linear_q = layers.Dense(mid_planes)
        self.linear_k = layers.Dense(mid_planes)
        self.linear_v = layers.Dense(out_planes)

        self.linear_p = tf.keras.models.Sequential(
            (layers.InputLayer(input_shape=(self.nsample, 3)), layers.Dense(3),
             layers.BatchNormalization(momentum=0.9,
                                       epsilon=1e-5), layers.ReLU(),
             layers.Dense(out_planes)))

        self.linear_w = tf.keras.models.Sequential(
            (layers.InputLayer(input_shape=(self.nsample, mid_planes)),
             layers.BatchNormalization(momentum=0.9,
                                       epsilon=1e-5), layers.ReLU(),
             layers.Dense(mid_planes // share_planes),
             layers.BatchNormalization(momentum=0.9,
                                       epsilon=1e-5), layers.ReLU(),
             layers.Dense(out_planes // share_planes)))

        self.softmax = layers.Softmax(axis=1)

    def call(self, pxo, training):
        """Forward call for Transformer.

        Args:
            pxo: [point, feat, row_splits] with shapes
                (n, 3), (n, c) and (b+1,)
            training: training mode of model.

        Returns:
            Transformer features.
        """
        point, feat, row_splits = pxo  # (n, 3), (n, c), (b+1)
        feat_q, feat_k, feat_v = self.linear_q(feat), self.linear_k(
            feat), self.linear_v(feat)  # (n, c)
        feat_k = queryandgroup(self.nsample,
                               point,
                               point,
                               feat_k,
                               None,
                               row_splits,
                               row_splits,
                               use_xyz=True)  # (n, nsample, 3+c)
        feat_v = queryandgroup(self.nsample,
                               point,
                               point,
                               feat_v,
                               None,
                               row_splits,
                               row_splits,
                               use_xyz=False)  # (n, nsample, c)
        point_r, feat_k = feat_k[:, :, 0:3], feat_k[:, :, 3:]

        point_r = self.linear_p(point_r, training=training)

        w = feat_k - tf.expand_dims(feat_q, 1) + tf.reduce_sum(
            tf.reshape(point_r,
                       (-1, self.nsample, self.out_planes // self.mid_planes,
                        self.mid_planes)), 2)

        w = self.linear_w(w, training=training)

        w = self.softmax(w)  # (n, nsample, c)

        n, nsample, c = feat_v.shape
        s = self.share_planes

        feat = tf.reshape(
            tf.reduce_sum(
                tf.reshape(feat_v + point_r,
                           (-1, nsample, s, c // s)) * tf.expand_dims(w, 2), 1),
            (-1, c))

        return feat


class TransitionDown(layers.Layer):
    """TransitionDown layer for PointTransformer.

    Subsamples points and increase receptive field.
    """

    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        """Constructor for TransitionDown Layer.

        Args:
            in_planes (int): Number of input planes.
            out_planes (int): Number of output planes.
            stride (int): subsampling factor.
            nsample (int): Number of neighbours.

        """
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = layers.Dense(out_planes, use_bias=False)
            self.pool = layers.MaxPool1D(nsample)
        else:
            self.linear = layers.Dense(out_planes, use_bias=False)
        self.bn = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu = layers.ReLU()

    @staticmethod
    def compute_new_row_splits(row_splits, stride):
        new_row_splits = [0]
        count = 0
        for i in range(1, row_splits.shape[0]):
            count += (row_splits[i].item() - row_splits[i - 1].item()) // stride
            new_row_splits.append(count)

        return np.array(new_row_splits, dtype=np.int64)

    def call(self, pxo, training):
        """Forward call for TransitionDown

        Args:
            pxo: [point, feat, row_splits] with shapes
                (n, 3), (n, c) and (b+1,)
            training: training mode of model.

        Returns:
            List of point, feat, row_splits.

        """
        point, feat, row_splits = pxo  # (n, 3), (n, c), (b+1)
        if self.stride != 1:
            new_row_splits = tf.numpy_function(self.compute_new_row_splits,
                                               [row_splits, self.stride],
                                               tf.int64)

            idx = tf.numpy_function(furthest_point_sample_v2,
                                    [point, row_splits, new_row_splits],
                                    tf.int64)
            new_point = tf.gather(point, tf.reshape(idx, (-1,)))
            feat = queryandgroup(self.nsample,
                                 point,
                                 new_point,
                                 feat,
                                 None,
                                 row_splits,
                                 new_row_splits,
                                 use_xyz=True)  # (m, nsample, 3+c)
            feat = self.relu(self.bn(self.linear(feat),
                                     training=training))  # (m, c, nsample)
            feat = tf.squeeze(self.pool(feat), 1)  # (m, c)

            point, row_splits = new_point, new_row_splits
        else:
            feat = self.relu(self.bn(self.linear(feat),
                                     training=training))  # (n, c)
        return [point, feat, row_splits]


class TransitionUp(layers.Layer):
    """Decoder layer for PointTransformer.

    Interpolate points based on corresponding encoder layer.
    """

    def __init__(self, in_planes, out_planes=None):
        """Constructor for TransitionUp Layer.

        Args:
            in_planes (int): Number of input planes.
            out_planes (int): Number of output planes.

        """
        super().__init__()
        if out_planes is None:
            self.linear1 = tf.keras.models.Sequential(
                (layers.InputLayer(input_shape=(2 * in_planes,)),
                 layers.Dense(in_planes),
                 layers.BatchNormalization(momentum=0.9,
                                           epsilon=1e-5), layers.ReLU()))
            self.linear2 = tf.keras.models.Sequential(
                (layers.InputLayer(input_shape=(in_planes,)),
                 layers.Dense(in_planes), layers.ReLU()))
        else:
            self.linear1 = tf.keras.models.Sequential(
                (layers.InputLayer(input_shape=(out_planes,)),
                 layers.Dense(out_planes),
                 layers.BatchNormalization(momentum=0.9,
                                           epsilon=1e-5), layers.ReLU()))
            self.linear2 = tf.keras.models.Sequential(
                (layers.InputLayer(input_shape=(in_planes,)),
                 layers.Dense(out_planes),
                 layers.BatchNormalization(momentum=0.9,
                                           epsilon=1e-5), layers.ReLU()))

    def call(self, pxo1, pxo2=None, training=False):
        """Forward call for TransitionUp

        Args:
            pxo1: [point, feat, row_splits] with shapes
                (n, 3), (n, c) and (b+1,)
            pxo2: [point, feat, row_splits] with shapes
                (n, 3), (n, c) and (b+1,)
            training: training mode of model.

        Returns:
            Interpolated features.

        """
        if pxo2 is None:
            _, feat, row_splits = pxo1  # (n, 3), (n, c), (b)
            feat_tmp = []
            for i in range(0, tf.shape(row_splits)[0] - 1):
                start_i, end_i, count = row_splits[i], row_splits[i + 1], (
                    row_splits[i + 1] - row_splits[i]).numpy()
                feat_b = feat[start_i:end_i, :]

                tmp = self.linear2(tf.reduce_sum(feat_b, 0, keepdims=True) /
                                   count,
                                   training=training)
                tmp = tf.reshape(
                    tf.repeat(tf.expand_dims(tmp, 0), repeats=count, axis=0),
                    (-1, tmp.shape[1]))
                feat_b = tf.concat((feat_b, tmp), 1)
                feat_tmp.append(feat_b)
            feat = tf.concat(feat_tmp, 0)
            feat = self.linear1(feat, training=training)
        else:
            point_1, feat_1, row_splits_1 = pxo1
            point_2, feat_2, row_splits_2 = pxo2
            feat = self.linear1(feat_1, training=training) + interpolation(
                point_2, point_1, self.linear2(feat_2, training=training),
                row_splits_2, row_splits_1)
        return feat


class Bottleneck(layers.Layer):
    """Bottleneck layer for PointTransformer.

    Block of layers using Transformer layer as building block.
    """
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        """Constructor for Bottleneck Layer.

        Args:
            in_planes (int): Number of input planes.
            planes (int): Number of output planes.
            share_planes (int): Number of shared planes.
            nsample (int): Number of neighbours.

        """
        super(Bottleneck, self).__init__()
        self.linear1 = layers.Dense(planes, use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.transformer2 = Transformer(planes, planes, share_planes, nsample)
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.linear3 = layers.Dense(planes * self.expansion, use_bias=False)
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu = layers.ReLU()

    def call(self, pxo, training):
        """Forward call for Bottleneck

        Args:
            pxo: [point, feat, row_splits] with shapes
                (n, 3), (n, c) and (b+1,)
            training: training mode of model.

        Returns:
            List of point, feat, row_splits.

        """
        point, feat, row_splits = pxo  # (n, 3), (n, c), (b)
        identity = feat
        feat = self.relu(self.bn1(self.linear1(feat), training=training))
        feat = self.relu(
            self.bn2(self.transformer2([point, feat, row_splits],
                                       training=training),
                     training=training))
        feat = self.bn3(self.linear3(feat), training=training)
        feat += identity
        feat = self.relu(feat)
        return [point, feat, row_splits]


def queryandgroup(nsample,
                  points,
                  queries,
                  feat,
                  idx,
                  points_row_splits,
                  queries_row_splits,
                  use_xyz=True):
    """Find nearest neighbours and returns grouped features.

    Args:
        nsample: Number of neighbours (k).
        points: Input pointcloud (n, 3).
        queries: Queries for Knn (m, 3).
        feat: features (n, c).
        idx: Optional knn index list.
        points_row_splits: row_splits for batching points.
        queries_row_splits: row_splits for batching queries.
        use_xyz: Whether to return xyz concatenated with features.

    Returns:
        Returns grouped features (m, nsample, c) or (m, nsample, 3+c).

    """
    if queries is None:
        queries = points
    if idx is None:
        points_row_splits = tf.reshape(points_row_splits, (-1,))
        queries_row_splits = tf.reshape(queries_row_splits, (-1,))
        ans = knn_search(points,
                         queries,
                         k=nsample,
                         points_row_splits=points_row_splits,
                         queries_row_splits=queries_row_splits,
                         return_distances=False)  # (n, 3)
        idx = tf.cast(tf.reshape(ans.neighbors_index, (-1, nsample)), tf.int64)

    n, m, c = points.shape[0], queries.shape[0], feat.shape[1]
    grouped_xyz = tf.reshape(tf.gather(points, tf.reshape(idx, (-1,))),
                             (-1, nsample, 3))
    grouped_xyz -= tf.expand_dims(queries, 1)  # (m, nsample, 3)

    grouped_feat = tf.reshape(tf.gather(feat, tf.reshape(idx, (-1,))),
                              (-1, nsample, c))

    if use_xyz:
        return tf.concat((grouped_xyz, grouped_feat), -1)  # (m, nsample, 3+c)
    else:
        return grouped_feat


def interpolation(points,
                  queries,
                  feat,
                  points_row_splits,
                  queries_row_splits,
                  k=3):
    """Interpolation of features with nearest neighbours.

    Args:
        points: Input pointcloud (m, 3).
        queries: Queries for Knn (n, 3).
        feat: features (m, c).
        points_row_splits: row_splits for batching points.
        queries_row_splits: row_splits for batching queries.
        k: Number of neighbours.

    Returns:
        Returns interpolated features (n, c).

    """
    ans = knn_search(points,
                     queries,
                     k=k,
                     points_row_splits=points_row_splits,
                     queries_row_splits=queries_row_splits,
                     return_distances=True)
    idx = tf.cast(tf.reshape(ans.neighbors_index, (-1, k)), tf.int64)
    dist = tf.reshape(ans.neighbors_distance, (-1, k))

    dist_recip = 1.0 / (dist + 1e-8)  # (n, k)
    norm = tf.reduce_sum(dist_recip, 1, keepdims=True)
    weight = dist_recip / norm  # (n, k)

    new_feat = tf.zeros((queries.shape[0], feat.shape[1]))

    for i in range(k):
        new_feat += tf.gather(feat, idx[:, i]) * tf.expand_dims(
            weight[:, i], -1)

    return new_feat
