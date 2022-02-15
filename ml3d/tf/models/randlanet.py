import tensorflow as tf
import numpy as np

from tqdm import tqdm
from sklearn.neighbors import KDTree

from .base_model import BaseModel
from ...datasets.augment import SemsegAugmentation
from ...utils import MODEL
from ...datasets.utils import (DataProcessing, trans_normalize, trans_crop_pc)


class RandLANet(BaseModel):
    """Class defining RandLANet, a Semantic Segmentation model.  Based on the
    architecture from the paper `RandLA-Net: Efficient Semantic Segmentation of
    Large-Scale Point Clouds <https://arxiv.org/abs/1911.11236>`__.

    RandLA-Net is an efficient and lightweight neural architecture which
    directly infer per-point semantics for large-scale point clouds. The key
    approach is to use random point sampling instead of more complex point
    selection approaches.  Although remarkably computation and memory
    efficient, random sampling can discard key features by chance. To overcome
    this, we introduce a novel local feature aggregation module to
    progressively increase the receptive field for each 3D point, thereby
    effectively preserving geometric details.

    **Architecture**

    .. image:: https://user-images.githubusercontent.com/23613902/150006228-34fb9e04-76b6-4022-af08-c308da6dcaae.png
        :width: 100%

    References:
        https://github.com/QingyongHu/RandLA-Net
    """

    def __init__(
            self,
            name='RandLANet',
            num_neighbors=16,
            num_layers=4,
            num_points=4096 * 11,
            num_classes=19,
            ignored_label_inds=[0],
            sub_sampling_ratio=[4, 4, 4, 4],
            in_channels=3,  # 3 + feature_dimension.
            dim_features=8,
            dim_output=[16, 64, 128, 256],
            grid_size=0.06,
            batcher='DefaultBatcher',
            ckpt_path=None,
            augment=None,
            **kwargs):

        super().__init__(name=name,
                         num_neighbors=num_neighbors,
                         num_layers=num_layers,
                         num_points=num_points,
                         num_classes=num_classes,
                         ignored_label_inds=ignored_label_inds,
                         sub_sampling_ratio=sub_sampling_ratio,
                         in_channels=in_channels,
                         dim_features=dim_features,
                         dim_output=dim_output,
                         grid_size=grid_size,
                         batcher=batcher,
                         ckpt_path=ckpt_path,
                         augment=augment,
                         **kwargs)

        cfg = self.cfg
        self.augmenter = SemsegAugmentation(cfg.augment, seed=self.rng)

        self.fc0 = tf.keras.layers.Dense(cfg.dim_features)
        self.bn0 = tf.keras.layers.BatchNormalization(-1,
                                                      momentum=0.99,
                                                      epsilon=1e-6)
        self.lr0 = tf.keras.layers.LeakyReLU(0.2)

        # Encoder
        self.encoder = []
        encoder_dim_list = []
        dim_feature = cfg.dim_features
        for i in range(cfg.num_layers):
            self.encoder.append(
                LocalFeatureAggregation(dim_feature, cfg.dim_output[i],
                                        cfg.num_neighbors))
            dim_feature = 2 * cfg.dim_output[i]
            if i == 0:
                encoder_dim_list.append(dim_feature)
            encoder_dim_list.append(dim_feature)

        self.mlp = SharedMLP(dim_feature,
                             dim_feature,
                             activation_fn=tf.keras.layers.LeakyReLU(0.2))

        # Decoder
        self.decoder = []
        for i in range(cfg.num_layers):
            self.decoder.append(
                SharedMLP(encoder_dim_list[-i - 2] + dim_feature,
                          encoder_dim_list[-i - 2],
                          transpose=True,
                          activation_fn=tf.keras.layers.LeakyReLU(0.2)))
            dim_feature = encoder_dim_list[-i - 2]

        self.fc1 = tf.keras.models.Sequential(
            (SharedMLP(dim_feature,
                       64,
                       activation_fn=tf.keras.layers.LeakyReLU(0.2)),
             SharedMLP(64, 32, activation_fn=tf.keras.layers.LeakyReLU(0.2)),
             tf.keras.layers.Dropout(0.5),
             SharedMLP(32, cfg.num_classes, bn=False)))

    def call(self, inputs, training=True):
        self.training = training
        cfg = self.cfg
        num_layers = cfg.num_layers
        coords_list = inputs[:num_layers]
        neighbor_indices_list = inputs[num_layers:2 * num_layers]
        subsample_indices_list = inputs[2 * num_layers:3 * num_layers]
        interpolation_indices_list = inputs[3 * num_layers:4 * num_layers]
        feat = inputs[4 * num_layers]

        feat = self.fc0(feat)  # (B, N, dim_feature)
        feat = self.bn0(feat, training=training)  # (B, N, dim_feature)
        feat = self.lr0(feat)
        feat = tf.expand_dims(feat, axis=2)  # (B, N, 1, dim_feature)

        # Encoder
        encoder_feat_list = []
        for i in range(cfg.num_layers):
            feat_encoder_i = self.encoder[i](coords_list[i],
                                             feat,
                                             neighbor_indices_list[i],
                                             training=training)
            feat_sampled_i = self.random_sample(feat_encoder_i,
                                                subsample_indices_list[i])
            if i == 0:
                encoder_feat_list.append(tf.identity(feat_encoder_i))
            encoder_feat_list.append(tf.identity(feat_sampled_i))
            feat = feat_sampled_i

        feat = self.mlp(feat, training=training)

        # Decoder
        for i in range(cfg.num_layers):
            feat_interpolation_i = self.nearest_interpolation(
                feat, interpolation_indices_list[-i - 1])
            feat_decoder_i = tf.concat(
                [encoder_feat_list[-i - 2], feat_interpolation_i], axis=-1)
            feat_decoder_i = self.decoder[i](feat_decoder_i, training=training)
            feat = feat_decoder_i

        scores = self.fc1(feat, training=training)

        return tf.squeeze(scores, [2])

    def get_optimizer(self, cfg_pipeline):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            cfg_pipeline.optimizer.learning_rate,
            decay_steps=100000,
            decay_rate=cfg_pipeline.scheduler_gamma)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        return optimizer

    def get_loss(self, Loss, results, inputs):
        """Calculate the loss on output of the model.

        Args:
            Loss: Object of type `SemSegLoss`.
            results: Output of the model (B, N, C).
            inputs: Input of the model.
            device: device(cpu or cuda).

        Returns:
            Returns loss, labels and scores.

        """
        cfg = self.cfg
        labels = inputs[-1]

        scores, labels = Loss.filter_valid_label(results, labels)

        loss = Loss.weighted_CrossEntropyLoss(scores, labels)

        return loss, labels, scores

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        Args:
            feature: [B, d, N, 1] input features matrix
            pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling

        Returns:
             pool_features = [B, N', d] pooled features matrix

        """
        feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])

        pool_features = tf.gather(feature, pool_idx, axis=1, batch_dims=1)

        pool_features = tf.reshape(pool_features,
                                   [batch_size, -1, num_neigh, d])

        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)

        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        Args:
            feature: [B, d, N] input features matrix
            interp_idx: [B, up_num_points, 1] nearest neighbour index

        Returns:
             [B, up_num_points, d] interpolated features matrix

        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])

        interpolatedim_features = tf.gather(feature,
                                            interp_idx,
                                            axis=1,
                                            batch_dims=1)
        interpolatedim_features = tf.expand_dims(interpolatedim_features,
                                                 axis=2)
        return interpolatedim_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(
            features, [batch_size, num_points,
                       tf.shape(neighbor_idx)[-1], d])
        return features

    def get_batch_gen(self, dataset, steps_per_epoch=None, batch_size=1):
        cfg = self.cfg
        rng = self.rng

        def gen():
            n_iters = dataset.num_pc if steps_per_epoch is None else steps_per_epoch * batch_size
            for i in range(n_iters):
                data, attr = dataset.read_data(i % dataset.num_pc)

                pc = data['point'].copy()
                label = data['label'].copy()
                feat = data['feat'].copy() if data['feat'] is not None else None
                tree = data['search_tree']

                pick_idx = np.random.choice(len(pc), 1)
                center_point = pc[pick_idx, :].reshape(1, -1)
                pc, feat, label, _ = trans_crop_pc(pc, feat, label, tree,
                                                   pick_idx,
                                                   self.cfg.num_points)

                augment_cfg = self.cfg.get('augment', {}).copy()
                val_augment_cfg = {}
                if 'recenter' in augment_cfg:
                    val_augment_cfg['recenter'] = augment_cfg['recenter']
                    augment_cfg.pop('recenter')
                if 'normalize' in augment_cfg:
                    val_augment_cfg['normalize'] = augment_cfg['normalize']
                    augment_cfg.pop('normalize')

                self.augmenter.augment(pc,
                                       feat,
                                       label,
                                       val_augment_cfg,
                                       seed=rng)

                if attr['split'] in ['training', 'train']:
                    pc, feat, label = self.augmenter.augment(pc,
                                                             feat,
                                                             label,
                                                             augment_cfg,
                                                             seed=rng)

                if feat is None:
                    feat = pc.copy()
                else:
                    feat = np.concatenate([pc, feat], axis=1)

                if cfg.in_channels != feat.shape[1]:
                    raise RuntimeError(
                        "Wrong feature dimension, please update in_channels(3 + feature_dimension) in config"
                    )

                yield (pc.astype(np.float32), feat.astype(np.float32),
                       label.astype(np.float32))

        gen_func = gen
        gen_types = (tf.float32, tf.float32, tf.int32)
        gen_shapes = ([None, 3], [None, cfg.in_channels], [None])

        return gen_func, gen_types, gen_shapes

    def transform_inference(self, data, min_possibility_idx):
        cfg = self.cfg
        inputs = dict()

        pc = data['point'].copy()
        label = data['label'].copy()
        feat = data['feat'].copy() if data['feat'] is not None else None
        tree = data['search_tree']

        pick_idx = min_possibility_idx
        center_point = pc[pick_idx, :].reshape(1, -1)

        pc, feat, label, selected_idx = trans_crop_pc(pc, feat, label, tree,
                                                      pick_idx,
                                                      self.cfg.num_points)

        dists = np.sum(np.square(pc.astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists))
        self.possibility[selected_idx] += delta
        inputs['point_inds'] = selected_idx

        t_normalize = cfg.get('t_normalize', {})
        pc, feat = trans_normalize(pc, feat, t_normalize)

        if feat is None:
            feat = pc.copy()
        else:
            feat = np.concatenate([pc, feat], axis=1)

        if self.cfg.in_channels != feat.shape[1]:
            raise RuntimeError(
                "Wrong feature dimension, please update in_channels(3 + feature_dimension) in config"
            )

        features = feat
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):
            neighbour_idx = DataProcessing.knn_search(pc, pc, cfg.num_neighbors)

            sub_points = pc[:pc.shape[0] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:pc.shape[0] // cfg.sub_sampling_ratio[i], :]
            up_i = DataProcessing.knn_search(sub_points, pc, 1)
            input_points.append(pc)
            input_neighbors.append(neighbour_idx.astype(np.int64))
            input_pools.append(pool_i.astype(np.int64))
            input_up_samples.append(up_i.astype(np.int64))
            pc = sub_points

        inputs['xyz'] = input_points
        inputs['neigh_idx'] = input_neighbors
        inputs['sub_idx'] = input_pools
        inputs['interp_idx'] = input_up_samples
        inputs['features'] = features

        inputs['labels'] = label.astype(np.int64)

        return inputs

    def transform(self, pc, feat, label):
        cfg = self.cfg

        if feat is pc:
            feat = None

        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):
            neighbour_idx = tf.numpy_function(DataProcessing.knn_search,
                                              [pc, pc, cfg.num_neighbors],
                                              tf.int32)

            sub_points = pc[:tf.shape(pc)[0] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:tf.shape(pc)[0] //
                                   cfg.sub_sampling_ratio[i], :]
            up_i = tf.numpy_function(DataProcessing.knn_search,
                                     [sub_points, pc, 1], tf.int32)
            input_points.append(pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            pc = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [feat, label]

        return input_list

    def inference_begin(self, data):
        self.test_smooth = 0.95
        attr = {'split': 'test'}
        self.inference_data = self.preprocess(data, attr)
        num_points = self.inference_data['search_tree'].data.shape[0]
        self.possibility = np.random.rand(num_points) * 1e-3
        self.test_probs = np.zeros(shape=[num_points, self.cfg.num_classes],
                                   dtype=np.float16)
        self.pbar = tqdm(total=self.possibility.shape[0])
        self.pbar_update = 0

    def inference_preprocess(self):
        min_possibility_idx = np.argmin(self.possibility)
        data = self.transform_inference(self.inference_data,
                                        min_possibility_idx)
        inputs = {'data': data, 'attr': []}
        # inputs = self.batcher.collate_fn([inputs])
        self.inference_input = inputs

        flat_inputs = data['xyz'] + data['neigh_idx'] + data['sub_idx'] + data[
            'interp_idx']
        flat_inputs += [data['features'], data['labels']]

        for i in range(len(flat_inputs)):
            flat_inputs[i] = np.expand_dims(flat_inputs[i], 0)

        return flat_inputs

    def inference_end(self, results):
        inputs = self.inference_input
        results = tf.reshape(results, (-1, self.cfg.num_classes))
        results = tf.nn.softmax(results, axis=-1)
        results = results.cpu().numpy()

        probs = np.reshape(results, [-1, self.cfg.num_classes])
        inds = inputs['data']['point_inds']
        self.test_probs[inds] = self.test_smooth * self.test_probs[inds] + (
            1 - self.test_smooth) * probs

        self.pbar.update(self.possibility[self.possibility > 0.5].shape[0] -
                         self.pbar_update)
        self.pbar_update = self.possibility[self.possibility > 0.5].shape[0]

        if np.min(self.possibility) > 0.5:
            self.pbar.close()
            reproj_inds = self.inference_data['proj_inds']
            self.test_probs = self.test_probs[reproj_inds]
            inference_result = {
                'predict_labels': np.argmax(self.test_probs, 1),
                'predict_scores': self.test_probs
            }
            self.inference_result = inference_result
            return True
        else:
            return False

    def preprocess(self, data, attr):
        cfg = self.cfg

        points = np.array(data['point'][:, 0:3], dtype=np.float32)

        if 'label' not in data or data['label'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))

        if 'feat' not in data or data['feat'] is None:
            feat = None
        else:
            feat = np.array(data['feat'], dtype=np.float32)

        split = attr['split']
        data = dict()

        if feat is None:
            sub_points, sub_labels = DataProcessing.grid_subsampling(
                points, labels=labels, grid_size=cfg.grid_size)
            sub_feat = None

        else:
            sub_points, sub_feat, sub_labels = DataProcessing.grid_subsampling(
                points, features=feat, labels=labels, grid_size=cfg.grid_size)

        search_tree = KDTree(sub_points)

        data['point'] = sub_points
        data['feat'] = sub_feat
        data['label'] = sub_labels
        data['search_tree'] = search_tree

        if split in ["test", "testing"]:
            proj_inds = np.squeeze(
                search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            data['proj_inds'] = proj_inds

        return data


MODEL._register_module(RandLANet, 'tf')


class SharedMLP(tf.keras.layers.Layer):
    """Module consisting of commonly used layers conv, batchnorm
    and any activation function.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 transpose=False,
                 bn=True,
                 activation_fn=None):
        super(SharedMLP, self).__init__()

        if transpose:
            self.conv = tf.keras.layers.Conv2DTranspose(filters=out_channels,
                                                        kernel_size=kernel_size,
                                                        strides=stride,
                                                        padding='valid')
        else:
            self.conv = tf.keras.layers.Conv2D(filters=out_channels,
                                               kernel_size=kernel_size,
                                               strides=stride,
                                               padding='valid')

        self.batch_norm = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=0.99, epsilon=1e-6) if bn else None
        self.activation_fn = activation_fn

    def call(self, input, training):
        """Forward pass of the Module.

        Args:
            input: tf.Tensor of shape (B, dim_in, N, K)

        Returns:
            tf.Tensor, shape (B, dim_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x, training=training)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(tf.keras.layers.Layer):
    """This module computes k neighbour feature encoding for each point.
    Encoding consists of absolute distance, relative distance, positions.
    """

    def __init__(self, dim_in, dim_out, num_neighbors, encode_pos=False):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(dim_in,
                             dim_out,
                             activation_fn=tf.keras.layers.LeakyReLU(0.2))
        self.encode_pos = encode_pos

    def gather_neighbor(self, coords, neighbor_indices):
        """Gather features based on neighbor indices.

        Args:
            coords: tf.Tensor of shape (B, N, d)
            neighbor_indices: tf.Tensor of shape (B, N, K)

        Returns:
            gathered neighbors of shape (B, dim, N, K)

        """
        B, N, K = neighbor_indices.shape
        dim = coords.shape[2]

        extended_indices = tf.reshape(neighbor_indices, shape=(-1, N * K))
        neighbor_coords = tf.gather(coords,
                                    extended_indices,
                                    axis=1,
                                    batch_dims=1)

        return tf.reshape(neighbor_coords, [-1, N, K, dim])

    def call(self,
             coords,
             features,
             neighbor_indices,
             training,
             relative_features=None):
        """Forward pass of the Module.

        Args:
            coords: coordinates of the pointcloud
                tf.Tensor of shape (B, N, 3)
            features: features of the pointcloud.
                tf.Tensor of shape (B, d, N, 1)
            neighbor_indices: indices of k neighbours.
                tf.Tensor of shape (B, N, K)
            training: whether called under training mode.
            relative_features: relative neighbor features calculated
              on first pass. Required for second pass.

        Returns:
            tf.Tensor of shape (B, 2*d, N, K)
        """
        # finding neighboring points
        if self.encode_pos:
            neighbor_coords = self.gather_neighbor(coords, neighbor_indices)

            extended_coords = tf.tile(tf.expand_dims(
                coords, axis=2), [1, 1, tf.shape(neighbor_indices)[-1], 1])
            relative_pos = extended_coords - neighbor_coords
            relative_dist = tf.sqrt(
                tf.reduce_sum(tf.square(relative_pos), axis=-1, keepdims=True))
            relative_features = tf.concat(
                [relative_dist, relative_pos, extended_coords, neighbor_coords],
                axis=-1)
        else:
            if relative_features is None:
                raise ValueError(
                    "LocalSpatialEncoding: Require relative_features for second pass."
                )

        relative_features = self.mlp(relative_features, training=training)

        neighbor_features = self.gather_neighbor(tf.squeeze(features, axis=2),
                                                 neighbor_indices)

        return tf.concat([neighbor_features, relative_features],
                         axis=-1), relative_features


class AttentivePooling(tf.keras.layers.Layer):
    """This module pools down k neighbour features to a single encoding
    using weighted average with attention scores.
    """

    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = tf.keras.models.Sequential(
            ((tf.keras.layers.Dense(in_channels),
              tf.keras.layers.Softmax(axis=-2))))

        self.mlp = SharedMLP(in_channels,
                             out_channels,
                             activation_fn=tf.keras.layers.LeakyReLU(0.2))

    def call(self, x, training):
        """Forward pass of the Module.

        Args:
            x: tf.Tensor of shape (B, N, K, dim_in).

        Returns:
            tf.Tensor of shape (B, d_out, N, 1).
        """
        # computing attention scores
        scores = self.score_fn(x)

        # sum over the neighbors
        features = tf.reduce_sum(scores * x, axis=-2,
                                 keepdims=True)  # shape (B, d_in, N, 1)

        return self.mlp(features, training=training)


class LocalFeatureAggregation(tf.keras.layers.Layer):
    """The neighbour features returned from LocalSpatialEncoding
    and pooled from AttentivePooling are aggregated and processed
    in multiple layers in this module.
    """

    def __init__(self, d_in, d_out, num_neighbors):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in,
                              d_out // 2,
                              activation_fn=tf.keras.layers.LeakyReLU(0.2))
        self.lse1 = LocalSpatialEncoding(10,
                                         d_out // 2,
                                         num_neighbors,
                                         encode_pos=True)
        self.pool1 = AttentivePooling(d_out, d_out // 2)

        self.lse2 = LocalSpatialEncoding(d_out // 2, d_out // 2, num_neighbors)
        self.pool2 = AttentivePooling(d_out, d_out)
        self.mlp2 = SharedMLP(d_out, 2 * d_out)

        self.shortcut = SharedMLP(d_in, 2 * d_out)
        self.lrelu = tf.keras.layers.LeakyReLU(0.2)

    def call(self, coords, feat, neighbor_indices, training):
        """Forward pass of the Module.

        Args:
            coords: coordinates of the pointcloud
                tf.Tensor of shape (B, N, 3).
            feat: features of the pointcloud.
                tf.Tensor of shape (B, d, N, 1)
            neighbor_indices: Indices of neighbors.

        Returns:
            tf.Tensor of shape (B, 2*d_out, N, 1).

        """
        # knn_output = knn(coords.cpu().contiguous(), coords.cpu().contiguous(), self.num_neighbors)

        x = self.mlp1(feat, training=training)

        x, neighbor_features = self.lse1(coords,
                                         x,
                                         neighbor_indices,
                                         training=training)
        x = self.pool1(x, training=training)

        x, _ = self.lse2(coords,
                         x,
                         neighbor_indices,
                         relative_features=neighbor_features,
                         training=training)
        x = self.pool2(x, training=training)

        return self.lrelu(
            self.mlp2(x, training=training) +
            self.shortcut(feat, training=training))
