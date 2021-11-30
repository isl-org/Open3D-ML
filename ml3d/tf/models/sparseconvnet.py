import numpy as np
import tensorflow as tf

from .base_model import BaseModel
from ...utils import MODEL
from ...datasets.augment import SemsegAugmentation

from open3d.ml.tf.layers import SparseConv, SparseConvTranspose
from open3d.ml.tf.ops import voxelize, reduce_subarrays_sum


class SparseConvUnet(BaseModel):
    """Semantic Segmentation model.

    Uses UNet architecture replacing convolutions with Sparse Convolutions.

    Attributes:
        name: Name of model.
          Default to "SparseConvUnet".
        device: Which device to use (cpu or cuda).
        voxel_size: Voxel length for subsampling.
        multiplier: min length of feature length in each layer.
        conv_block_reps: repetition of Unet Blocks.
        residual_blocks: Whether to use Residual Blocks.
        in_channels: Number of features(default 3 for color).
        num_classes: Number of classes.
    """

    def __init__(
            self,
            name="SparseConvUnet",
            device="cuda",
            multiplier=16,
            voxel_size=0.05,
            conv_block_reps=1,  # Conv block repetitions.
            residual_blocks=False,
            in_channels=3,
            num_classes=20,
            grid_size=4096,
            augment=None,
            **kwargs):
        super(SparseConvUnet, self).__init__(name=name,
                                             device=device,
                                             multiplier=multiplier,
                                             voxel_size=voxel_size,
                                             conv_block_reps=conv_block_reps,
                                             residual_blocks=residual_blocks,
                                             in_channels=in_channels,
                                             num_classes=num_classes,
                                             grid_size=grid_size,
                                             augment=augment,
                                             **kwargs)
        cfg = self.cfg
        self.multiplier = cfg.multiplier
        self.augmenter = SemsegAugmentation(cfg.augment)
        self.input_layer = InputLayer()
        self.sub_sparse_conv = SubmanifoldSparseConv(in_channels=in_channels,
                                                     filters=multiplier,
                                                     kernel_size=[3, 3, 3])
        self.unet = UNet(conv_block_reps, [
            multiplier, 2 * multiplier, 3 * multiplier, 4 * multiplier,
            5 * multiplier, 6 * multiplier, 7 * multiplier
        ], residual_blocks)

        self.batch_norm = BatchNormBlock()
        self.relu = ReLUBlock()
        self.linear = LinearBlock(multiplier, num_classes)
        self.output_layer = OutputLayer()

    def call(self, inputs, training=False):
        pos_list = []
        feat_list = []
        index_map_list = []
        start_idx = 0

        for length in inputs['batch_lengths']:
            pos = inputs['point'][start_idx:start_idx + length]
            feat = inputs['feat'][start_idx:start_idx + length]

            feat, pos, index_map = self.input_layer(feat, pos)
            pos_list.append(pos)
            feat_list.append(feat)
            index_map_list.append(index_map)

            start_idx += length

        feat_list = self.sub_sparse_conv(feat_list, pos_list, voxel_size=1.0)
        feat_list = self.unet(pos_list, feat_list, training=training)
        feat_list = self.batch_norm(feat_list, training=training)
        feat_list = self.relu(feat_list)
        feat_list = self.linear(feat_list)
        output = self.output_layer(feat_list, index_map_list)

        return output

    def preprocess(self, data, attr):
        points = np.array(data['point'], dtype=np.float32)

        if 'label' not in data or data['label'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))

        if 'feat' not in data or data['feat'] is None:
            raise Exception(
                "SparseConvnet doesn't work without feature values.")

        feat = np.array(data['feat'], dtype=np.float32)

        # Scale to voxel size.
        points *= 1. / self.cfg.voxel_size  # Scale = 1/voxel_size

        if attr['split'] in ['training', 'train']:
            points, feat, labels = self.augmenter.augment(
                points, feat, labels, self.cfg.get('augment', None))

        m = points.min(0)
        M = points.max(0)

        # Randomly place pointcloud in 4096 size grid.
        grid_size = self.cfg.grid_size
        offset = -m + np.clip(
            grid_size - M + m - 0.001, 0, None) * self.rng.random(3) + np.clip(
                grid_size - M + m + 0.001, None, 0) * self.rng.random(3)

        points += offset
        idxs = (points.min(1) >= 0) * (points.max(1) < 4096)

        points = points[idxs]
        feat = feat[idxs]
        labels = labels[idxs]

        points = (points.astype(np.int32) + 0.5).astype(
            np.float32)  # Move points to voxel center.

        data = {}
        data['point'] = points
        data['feat'] = feat
        data['label'] = labels

        return data

    def transform(self, point, feat, label, lengths):
        return {
            'point': point,
            'feat': feat,
            'label': label,
            'batch_lengths': lengths
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
                lengths = []
                start_id = batch_id * batch_size
                end_id = min(start_id + batch_size, dataset.num_pc)

                for cloud_id in range(start_id, end_id):
                    data, attr = dataset.read_data(cloud_id)
                    pc.append(data['point'])
                    feat.append(data['feat'])
                    label.append(data['label'])
                    lengths.append(data['point'].shape[0])

                pc = np.concatenate(pc, 0)
                feat = np.concatenate(feat, 0)
                label = np.concatenate(label, 0)
                lengths = np.array(lengths, dtype=np.int32)

                yield pc, feat, label, lengths

        gen_func = concat_batch_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None])

        return gen_func, gen_types, gen_shapes

    def inference_begin(self, data):
        data = self.preprocess(data, {'split': 'test'})

        self.inference_input = self.transform(
            tf.constant(data['point']), tf.constant(data['feat']),
            tf.constant(data['label']), tf.constant([data['point'].shape[0]]))

    def inference_preprocess(self):
        return self.inference_input

    def inference_end(self, results):
        results = tf.reshape(results, [-1, self.cfg.num_classes])

        m_softmax = tf.keras.layers.Softmax(axis=-1)
        results = m_softmax(results)
        results = results.numpy()

        probs = np.reshape(results, [-1, self.cfg.num_classes])

        pred_l = np.argmax(probs, 1)

        self.inference_result = {
            'predict_labels': pred_l,
            'predict_scores': probs
        }

        return True

    def get_loss(self, Loss, results, inputs):
        """Calculate the loss on output of the model.

        Attributes:
            Loss: Object of type `SemSegLoss`.
            results: Output of the model.
            inputs: Input of the model.
            device: device(cpu or cuda).
        
        Returns:
            Returns loss, labels and scores.
        """
        labels = inputs['label']
        scores, labels = Loss.filter_valid_label(results, labels)
        loss = Loss.weighted_CrossEntropyLoss(scores, labels)

        return loss, labels, scores

    def get_optimizer(self, cfg_pipeline):
        optimizer = tf.keras.optimizers.Adam(**cfg_pipeline.optimizer)

        return optimizer


MODEL._register_module(SparseConvUnet, 'tf')


class BatchNormBlock(tf.keras.layers.Layer):

    def __init__(self, eps=1e-4, momentum=0.99):
        super(BatchNormBlock, self).__init__()

        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum,
                                                     epsilon=eps)

    def call(self, feat_list, training):
        lengths = [feat.shape[0] for feat in feat_list]
        out = self.bn(tf.concat(feat_list, 0), training=training)
        out_list = []
        start = 0
        for l in lengths:
            out_list.append(out[start:start + l])
            start += l

        return out_list

    def __name__(self):
        return "BatchNormBlock"


class ReLUBlock(tf.keras.layers.Layer):

    def __init__(self):
        super(ReLUBlock, self).__init__()
        self.relu = tf.keras.layers.ReLU()

    def call(self, feat_list):
        lengths = [feat.shape[0] for feat in feat_list]
        out = self.relu(tf.concat(feat_list, 0))
        out_list = []
        start = 0
        for l in lengths:
            out_list.append(out[start:start + l])
            start += l

        return out_list

    def __name__(self):
        return "ReLUBlock"


class LinearBlock(tf.keras.layers.Layer):

    def __init__(self, a, b):
        super(LinearBlock, self).__init__()
        self.linear = tf.keras.layers.Dense(b)

    def call(self, feat_list):
        out_list = []
        for feat in feat_list:
            out_list.append(self.linear(feat))

        return out_list

    def __name__(self):
        return "LinearBlock"


class InputLayer(tf.keras.layers.Layer):

    def __init__(self, voxel_size=1.0):
        super(InputLayer, self).__init__()
        self.voxel_size = tf.constant([voxel_size, voxel_size, voxel_size])

    def get_reverse_map_voxelize(self, voxel_point_indices, length):
        length = length.shape[0]
        rev = np.zeros((length,))
        rev[voxel_point_indices] = np.arange(length)

        return rev.astype(np.int32)

    def get_reverse_map_sort(self, count):
        return np.repeat(np.arange(count.shape[0]), count).astype(np.int32)

    def call(self, features, in_positions):
        v = voxelize(
            in_positions,
            tf.convert_to_tensor([0, in_positions.shape[0]], dtype=tf.int64),
            self.voxel_size, tf.constant([0., 0., 0.]),
            tf.constant([40960., 40960., 40960.]))

        # Contiguous repeating positions.
        in_positions = tf.gather(in_positions, v.voxel_point_indices)
        features = tf.gather(features, v.voxel_point_indices)

        # Find reverse mapping.
        reverse_map_voxelize = tf.numpy_function(
            self.get_reverse_map_voxelize,
            [v.voxel_point_indices, in_positions], tf.int32)

        # Unique positions.
        in_positions = tf.gather(in_positions, v.voxel_point_row_splits[:-1])

        # Mean of features.
        count = v.voxel_point_row_splits[1:] - v.voxel_point_row_splits[:-1]
        reverse_map_sort = tf.numpy_function(self.get_reverse_map_sort, [count],
                                             tf.int32)

        features_avg_0 = tf.expand_dims(
            reduce_subarrays_sum(features[:, 0], v.voxel_point_row_splits), 1)
        features_avg_1 = tf.expand_dims(
            reduce_subarrays_sum(features[:, 1], v.voxel_point_row_splits), 1)
        features_avg_2 = tf.expand_dims(
            reduce_subarrays_sum(features[:, 2], v.voxel_point_row_splits), 1)

        features_avg = tf.concat(
            [features_avg_0, features_avg_1, features_avg_2], 1)

        features_avg = features_avg / tf.expand_dims(tf.cast(count, tf.float32),
                                                     1)

        return features_avg, in_positions, tf.gather(reverse_map_sort,
                                                     reverse_map_voxelize)


class OutputLayer(tf.keras.layers.Layer):

    def __init__(self, voxel_size=1.0):
        super(OutputLayer, self).__init__()

    def call(self, features_list, index_map_list):
        out = []
        for feat, index_map in zip(features_list, index_map_list):
            out.append(tf.gather(feat, index_map))
        return tf.concat(out, 0)


class SubmanifoldSparseConv(tf.keras.layers.Layer):

    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size,
                 use_bias=False,
                 offset=None,
                 normalize=False):
        super(SubmanifoldSparseConv, self).__init__()

        if offset is None:
            if kernel_size[0] % 2:
                offset = 0.
            else:
                offset = 0.5

        offset = tf.fill((3,), offset)
        self.net = SparseConv(in_channels=in_channels,
                              filters=filters,
                              kernel_size=kernel_size,
                              use_bias=use_bias,
                              offset=offset,
                              normalize=normalize)

    def call(self,
             features_list,
             in_positions_list,
             out_positions_list=None,
             voxel_size=1.0):
        if out_positions_list is None:
            out_positions_list = in_positions_list

        out_feat = []
        for feat, in_pos, out_pos in zip(features_list, in_positions_list,
                                         out_positions_list):
            out_feat.append(self.net(feat, in_pos, out_pos, voxel_size))

        return out_feat

    def __name__(self):
        return "SubmanifoldSparseConv"


class Convolution(tf.keras.layers.Layer):

    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size,
                 use_bias=False,
                 offset=None,
                 normalize=False):
        super(Convolution, self).__init__()

        if offset is None:
            if kernel_size[0] % 2:
                offset = 0.
            else:
                offset = -0.5

        offset = tf.fill((3,), offset)
        self.net = SparseConv(in_channels=in_channels,
                              filters=filters,
                              kernel_size=kernel_size,
                              use_bias=use_bias,
                              offset=offset,
                              normalize=normalize)

    @staticmethod
    def tf_unique_2d(x):
        v = voxelize(tf.cast(x, tf.float32), tf.constant([1., 1., 1.]),
                     tf.constant([0., 0., 0.]),
                     tf.constant([40960., 40960., 40960.]))
        idx = tf.gather(v.voxel_point_indices, v.voxel_point_row_splits[:-1])
        out = tf.gather(x, idx)

        out = tf.gather(out, tf.argsort(out[:, 2]))
        out = tf.gather(out, tf.argsort(out[:, 1]))
        out = tf.gather(out, tf.argsort(out[:, 0]))

        return out

    @staticmethod
    def calculate_grid(in_positions):
        filter = np.array([[-1, -1, -1], [-1, -1, 0], [-1, 0, -1], [-1, 0, 0],
                           [0, -1, -1], [0, -1, 0], [0, 0, -1], [0, 0, 0]])

        out_pos = in_positions.astype(np.int32).repeat(filter.shape[0],
                                                       axis=0).reshape(-1, 3)
        filter = np.expand_dims(filter, 0).repeat(in_positions.shape[0],
                                                  axis=0).reshape(-1, 3)

        out_pos = out_pos + filter
        out_pos = out_pos[out_pos.min(1) >= 0]
        out_pos = out_pos[(
            ~((out_pos.astype(np.int32) % 2).astype(np.bool)).any(1))]
        out_pos = np.unique(out_pos, axis=0)

        return (out_pos + 0.5).astype(np.float32)

    def call(self, features_list, in_positions_list, voxel_size=1.0):
        out_positions_list = []
        for in_positions in in_positions_list:
            out_positions_list.append(
                tf.numpy_function(self.calculate_grid, [in_positions],
                                  tf.float32))

        out_feat = []
        for feat, in_pos, out_pos in zip(features_list, in_positions_list,
                                         out_positions_list):
            out_feat.append(self.net(feat, in_pos, out_pos, voxel_size))

        out_positions_list = [out / 2 for out in out_positions_list]

        return out_feat, out_positions_list

    def __name__(self):
        return "Convolution"


class DeConvolution(tf.keras.layers.Layer):

    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size,
                 use_bias=False,
                 offset=None,
                 normalize=False):
        super(DeConvolution, self).__init__()

        if offset is None:
            if kernel_size[0] % 2:
                offset = 0.
            else:
                offset = -0.5

        offset = tf.fill((3,), offset)
        self.net = SparseConvTranspose(in_channels=in_channels,
                                       filters=filters,
                                       kernel_size=kernel_size,
                                       use_bias=use_bias,
                                       offset=offset,
                                       normalize=normalize)

    def call(self,
             features_list,
             in_positions_list,
             out_positions_list,
             voxel_size=1.0):
        out_feat = []
        for feat, in_pos, out_pos in zip(features_list, in_positions_list,
                                         out_positions_list):
            out_feat.append(self.net(feat, in_pos, out_pos, voxel_size))

        return out_feat

    def __name__(self):
        return "DeConvolution"


class ConcatFeat(tf.keras.layers.Layer):

    def __init__(self):
        super(ConcatFeat, self).__init__()

    def __name__(self):
        return "ConcatFeat"

    def call(self, feat):
        return feat


class JoinFeat(tf.keras.layers.Layer):

    def __init__(self):
        super(JoinFeat, self).__init__()

    def __name__(self):
        return "JoinFeat"

    def call(self, feat_cat, feat):
        out = []
        for a, b in zip(feat_cat, feat):
            out.append(tf.concat([a, b], -1))

        return out


class NetworkInNetwork(tf.keras.layers.Layer):

    def __init__(self, nIn, nOut, bias=False):
        super(NetworkInNetwork, self).__init__()
        if nIn == nOut:
            self.linear = tf.keras.layers.Lambda(lambda x: x)
        else:
            self.linear = tf.keras.layers.Dense(nOut, use_bias=bias)

    def call(self, inputs):
        out = []
        for inp in inputs:
            out.append(self.linear(inp))

        return out


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, nIn, nOut):
        super(ResidualBlock, self).__init__()

        self.lin = NetworkInNetwork(nIn, nOut)

        self.batch_norm1 = BatchNormBlock()

        self.relu1 = ReLUBlock()
        self.sub_sparse_conv1 = SubmanifoldSparseConv(in_channels=nIn,
                                                      filters=nOut,
                                                      kernel_size=[3, 3, 3])

        self.batch_norm2 = BatchNormBlock()

        self.relu2 = ReLUBlock()
        self.sub_sparse_conv2 = SubmanifoldSparseConv(in_channels=nOut,
                                                      filters=nOut,
                                                      kernel_size=[3, 3, 3])

    def call(self, feat_list, pos_list, training):
        out1 = self.lin(feat_list)

        feat_list = self.batch_norm1(feat_list, training)

        feat_list = self.relu1(feat_list)

        feat_list = self.sub_sparse_conv1(feat_list, pos_list)

        feat_list = self.batch_norm2(feat_list, training)
        feat_list = self.relu2(feat_list)

        out2 = self.sub_sparse_conv2(feat_list, pos_list)

        return [a + b for a, b in zip(out1, out2)]

    def __name__(self):
        return "ResidualBlock"


class UNet(tf.keras.layers.Layer):

    def __init__(self,
                 conv_block_reps,
                 nPlanes,
                 residual_blocks=False,
                 downsample=[2, 2],
                 leakiness=0):
        super(UNet, self).__init__()
        self.net = self.get_UNet(nPlanes, residual_blocks, conv_block_reps)
        self.residual_blocks = residual_blocks

    @staticmethod
    def block(layers, a, b, residual_blocks):
        if residual_blocks:
            layers.append(ResidualBlock(a, b))
        else:
            layers.append(BatchNormBlock())
            layers.append(ReLUBlock())
            layers.append(
                SubmanifoldSparseConv(in_channels=a,
                                      filters=b,
                                      kernel_size=[3, 3, 3]))

    @staticmethod
    def get_UNet(nPlanes, residual_blocks, conv_block_reps):
        layers = []
        for i in range(conv_block_reps):
            UNet.block(layers, nPlanes[0], nPlanes[0], residual_blocks)

        if len(nPlanes) > 1:
            layers.append(ConcatFeat())
            layers.append(BatchNormBlock())
            layers.append(ReLUBlock())
            layers.append(
                Convolution(in_channels=nPlanes[0],
                            filters=nPlanes[1],
                            kernel_size=[2, 2, 2]))
            layers = layers + UNet.get_UNet(nPlanes[1:], residual_blocks,
                                            conv_block_reps)
            layers.append(BatchNormBlock())
            layers.append(ReLUBlock())
            layers.append(
                DeConvolution(in_channels=nPlanes[1],
                              filters=nPlanes[0],
                              kernel_size=[2, 2, 2]))

            layers.append(JoinFeat())

            for i in range(conv_block_reps):
                UNet.block(layers, nPlanes[0] * (2 if i == 0 else 1),
                           nPlanes[0], residual_blocks)

        return layers

    def call(self, pos_list, feat_list, training):
        conv_pos = []
        concat_feat = []
        for module in self.net:
            if isinstance(module, BatchNormBlock):
                feat_list = module(feat_list, training=training)
            elif isinstance(module, ReLUBlock):
                feat_list = module(feat_list)

            elif isinstance(module, ResidualBlock):
                feat_list = module(feat_list, pos_list, training=training)

            elif isinstance(module, SubmanifoldSparseConv):
                feat_list = module(feat_list, pos_list)

            elif isinstance(module, Convolution):
                conv_pos.append([tf.identity(pos) for pos in pos_list])
                feat_list, pos_list = module(feat_list, pos_list)

            elif isinstance(module, DeConvolution):
                feat_list = module(feat_list, [2 * pos for pos in pos_list],
                                   conv_pos[-1])
                pos_list = conv_pos.pop()

            elif isinstance(module, ConcatFeat):
                concat_feat.append(
                    [tf.identity(feat) for feat in module(feat_list)])

            elif isinstance(module, JoinFeat):
                feat_list = module(concat_feat.pop(), feat_list)

            else:
                raise Exception("Unknown module {}".format(module))

        return feat_list
