import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from sklearn.neighbors import KDTree
import open3d.core as o3c

from .base_model import BaseModel
from ...utils import MODEL
from ...datasets.augment import SemsegAugmentation
from ...datasets.utils import DataProcessing
from ..utils.pointnet.pointnet2_utils import furthest_point_sample_v2

# def furthest_point_sample_v2(points, row_splits, new_row_splits):
#     idxs = np.arange(points.shape[0])
#     ret = []
#     for i in range(1, row_splits.shape[0]):
#         count = new_row_splits[i] - new_row_splits[i - 1]
#         ret += list(idxs[row_splits[i - 1]:row_splits[i - 1] + count])

#     return np.array(ret, dtype=np.int64)


class PointTransformer(BaseModel):

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
        self.enc1 = self._make_enc(block,
                                   planes[0],
                                   blocks[0],
                                   share_planes,
                                   stride=stride[0],
                                   nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block,
                                   planes[1],
                                   blocks[1],
                                   share_planes,
                                   stride=stride[1],
                                   nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block,
                                   planes[2],
                                   blocks[2],
                                   share_planes,
                                   stride=stride[2],
                                   nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block,
                                   planes[3],
                                   blocks[3],
                                   share_planes,
                                   stride=stride[3],
                                   nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block,
                                   planes[4],
                                   blocks[4],
                                   share_planes,
                                   stride=stride[4],
                                   nsample=nsample[4])  # N/256
        self.dec5 = self._make_dec(block,
                                   planes[4],
                                   2,
                                   share_planes,
                                   nsample=nsample[4],
                                   is_head=True)  # transform p5
        self.dec4 = self._make_dec(block,
                                   planes[3],
                                   2,
                                   share_planes,
                                   nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block,
                                   planes[2],
                                   2,
                                   share_planes,
                                   nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block,
                                   planes[1],
                                   2,
                                   share_planes,
                                   nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block,
                                   planes[0],
                                   2,
                                   share_planes,
                                   nsample=nsample[0])  # fusion p2 and p1

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
        for _ in range(1, blocks):
            x = block(self.in_planes,
                      self.in_planes,
                      share_planes,
                      nsample=nsample)(decoder_inputs)

        decoder.append(
            tf.keras.Model(inputs=decoder_inputs, outputs=x, name="decoder"))

        return decoder

    def call(self, inputs, training=False):
        point_0, feat_0, row_splits_0 = inputs['point'], inputs['feat'], inputs[
            'row_splits']  # (n, 3), (n, c), (b)

        feat_0 = point_0 if self.in_channels == 3 else tf.concat(
            (point_0, feat_0), 1)  # maybe use feat for in_channels == 3
        point_1, feat_1, row_splits_1 = self.enc1(
            [point_0, feat_0, row_splits_0], training=training)
        point_2, feat_2, row_splits_2 = self.enc2(
            [point_1, feat_1, row_splits_1], training=training)
        point_3, feat_3, row_splits_3 = self.enc3(
            [point_2, feat_2, row_splits_2], training=training)
        point_4, feat_4, row_splits_4 = self.enc4(
            [point_3, feat_3, row_splits_3], training=training)
        point_5, feat_5, row_splits_5 = self.enc5(
            [point_4, feat_4, row_splits_4], training=training)

        feat_5 = self.dec5[1]([
            point_5, self.dec5[0]([point_5, feat_5, row_splits_5],
                                  training=training), row_splits_5
        ],
                              training=training)[1]
        feat_4 = self.dec4[1]([
            point_4, self.dec4[0]([point_4, feat_4, row_splits_4],
                                  [point_5, feat_5, row_splits_5],
                                  training=training), row_splits_4
        ],
                              training=training)[1]
        feat_3 = self.dec3[1]([
            point_3, self.dec3[0]([point_3, feat_3, row_splits_3],
                                  [point_4, feat_4, row_splits_4],
                                  training=training), row_splits_3
        ],
                              training=training)[1]
        feat_2 = self.dec2[1]([
            point_2, self.dec2[0]([point_2, feat_2, row_splits_2],
                                  [point_3, feat_3, row_splits_3],
                                  training=training), row_splits_2
        ],
                              training=training)[1]
        feat_1 = self.dec1[1]([
            point_1, self.dec1[0]([point_1, feat_1, row_splits_1],
                                  [point_2, feat_2, row_splits_2],
                                  training=training), row_splits_1
        ],
                              training=training)[1]
        feat = self.cls(feat_1, training=training)

        return feat

    def preprocess(self, data, attr):
        cfg = self.cfg
        points = np.array(data['point'], dtype=np.float32)

        if 'label' not in data or data['label'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))

        if 'feat' not in data or data['feat'] is None:
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

    def get_loss(self, Loss, results, inputs):
        labels = inputs['label']
        scores, labels = Loss.filter_valid_label(results, labels)
        loss = Loss.weighted_CrossEntropyLoss(scores, labels)

        return loss, labels, scores

    def get_optimizer(self, cfg_pipeline):
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg_pipeline.adam_lr)

        return optimizer


MODEL._register_module(PointTransformer, 'tf')


class Transformer(layers.Layer):

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
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

        self.softmax = layers.Softmax(axis=-1)

    def call(self, pxo, training):
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

    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
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

    def __init__(self, in_planes, out_planes=None):
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
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(Bottleneck, self).__init__()
        self.linear1 = layers.Dense(planes, use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.transformer2 = Transformer(planes, planes, share_planes, nsample)
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.linear3 = layers.Dense(planes * self.expansion, use_bias=False)
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu = layers.ReLU()

    def call(self, pxo, training):
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
    """
    input: xyz: (n, 3), new_xyz: (m, 3), feat: (n, c), idx: (m, nsample), offset: (b), new_offset: (b)
    output: new_feat: (m, c+3, nsample), grouped_idx: (m, nsample)
    """
    # assert points.is_contiguous() and queries.is_contiguous(
    # ) and feat.is_contiguous()
    if queries is None:
        queries = points
    if idx is None:
        idx = tf.py_function(knn_batch,
                             inp=[
                                 points, queries, nsample, points_row_splits,
                                 queries_row_splits, False
                             ],
                             Tout=tf.int64)

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


def knn_batch(points,
              queries,
              k,
              points_row_splits,
              queries_row_splits,
              return_distances=True):
    assert points_row_splits.shape[0] == queries_row_splits.shape[
        0], "KNN(points and queries must have same batch size)"

    # idxs = []
    # dists = []
    # for i in range(0, points_row_splits.shape[0] - 1):
    #     idx = np.random.randint(0,
    #                             points_row_splits[i + 1] - points_row_splits[i],
    #                             size=(queries_row_splits[i + 1] -
    #                                   queries_row_splits[i], k))
    #     dist = np.ones(idx.shape, dtype=np.float32)
    #     idx += points_row_splits[i]
    #     idxs.append(tf.convert_to_tensor(idx))
    #     dists.append(tf.convert_to_tensor(dist))

    # if return_distances:
    #     return tf.concat(idxs, 0), tf.concat(dists, 0)
    # else:
    #     return tf.concat(idxs, 0)

    points = points.cpu()
    queries = queries.cpu()
    points = o3c.Tensor.from_dlpack(tf.experimental.dlpack.to_dlpack(points))
    queries = o3c.Tensor.from_dlpack(tf.experimental.dlpack.to_dlpack(queries))
    idxs = []
    dists = []

    for i in range(0, points_row_splits.shape[0] - 1):
        curr_points = points[points_row_splits[i]:points_row_splits[i + 1]]
        nns = o3c.nns.NearestNeighborSearch(curr_points)
        nns.knn_index()
        idx, dist = nns.knn_search(
            queries[queries_row_splits[i]:queries_row_splits[i + 1]], k)
        if idx.shape[1] < k:
            oversample = np.random.choice(np.arange(idx.shape[1]), k)
            idx = idx[:, oversample]
            dist = dist[:, oversample]
        idx += points_row_splits[i]
        idxs.append(tf.convert_to_tensor(idx.numpy()))
        dists.append(tf.convert_to_tensor(dist.numpy()))

    if return_distances:
        return tf.concat(idxs, 0), tf.concat(dists, 0)
    else:
        return tf.concat(idxs, 0)


def interpolation(points,
                  queries,
                  feat,
                  points_row_splits,
                  queries_row_splits,
                  k=3):
    """
    input: xyz: (m, 3), new_xyz: (n, 3), feat: (m, c), offset: (b), new_offset: (b)
    output: (n, c)
    """
    idx, dist = tf.py_function(
        knn_batch,
        inp=[points, queries, k, points_row_splits, queries_row_splits, True],
        Tout=[tf.int64, tf.float32])
    idx, dist = knn_batch(points,
                          queries,
                          k=k,
                          points_row_splits=points_row_splits,
                          queries_row_splits=queries_row_splits,
                          return_distances=True)  # (n, 3), (n, 3)

    idx, dist = tf.reshape(idx, (-1, 3)), tf.reshape(dist, (-1, 3))

    dist_recip = 1.0 / (dist + 1e-8)  # (n, 3)
    norm = tf.reduce_sum(dist_recip, 1, keepdims=True)
    weight = dist_recip / norm  # (n, 3)

    new_feat = tf.zeros((queries.shape[0], feat.shape[1]))

    for i in range(k):
        new_feat += tf.gather(feat, idx[:, i]) * tf.expand_dims(
            weight[:, i], -1)

    return new_feat
