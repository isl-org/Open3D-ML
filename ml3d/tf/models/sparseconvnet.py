import numpy as np
import tensorflow as tf

from .base_model import BaseModel
from ...utils import MODEL
from ...datasets.augment import SemsegAugmentation

from open3d.ml.tf.layers import SparseConv, SparseConvTranspose
from open3d.ml.tf.ops import voxelize, reduce_subarrays_sum


class SparseConvUnet(BaseModel):

    def __init__(
            self,
            name="SparseConvUnet",
            device="cuda",
            m=16,
            voxel_size=0.05,
            reps=1,  # Conv block repetitions.
            residual_blocks=False,
            in_channels=3,
            num_classes=20,
            **kwargs):
        super(SparseConvUnet, self).__init__(name=name,
                                             device=device,
                                             m=m,
                                             voxel_size=voxel_size,
                                             reps=reps,
                                             residual_blocks=residual_blocks,
                                             in_channels=in_channels,
                                             num_classes=num_classes,
                                             **kwargs)
        cfg = self.cfg
        self.m = cfg.m
        self.augment = SemsegAugmentation(cfg.augment)
        self.inp = InputLayer()
        self.ssc = SubmanifoldSparseConv(in_channels=in_channels,
                                         filters=m,
                                         kernel_size=[3, 3, 3])
        self.unet = UNet(reps, [m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m],
                         residual_blocks)

        self.bn = tf.keras.layers.BatchNormalization(momentum=0.99,
                                                     epsilon=1e-4)
        self.relu = tf.keras.layers.ReLU()

        self.linear = tf.keras.layers.Dense(num_classes)
        self.out = OutputLayer()

    def call(self, inputs, training=False):
        output = []
        start_idx = 0
        for length in inputs['batch_lengths']:
            pos = inputs['point'][start_idx:start_idx + length]
            feat = inputs['feat'][start_idx:start_idx + length]

            feat, pos, rev = self.inp(feat, pos)
            feat = self.ssc(feat, pos, voxel_size=1.0)
            feat = self.unet(pos, feat)
            feat = self.bn(feat)
            feat = self.relu(feat)
            feat = self.linear(feat)
            feat = self.out(feat, rev)

            output.append(feat)
            start_idx += length

        return tf.concat(output, 0)

    def preprocess(self, data, attr):
        points = np.array(data['point'], dtype=np.float32)

        if 'label' not in data.keys() or data['label'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))

        if 'feat' not in data.keys() or data['feat'] is None:
            raise Exception(
                "SparseConvnet doesn't work without feature values.")

        feat = np.array(data['feat'], dtype=np.float32)

        # Scale to voxel size.
        points *= 1. / self.cfg.voxel_size  # Scale = 1/voxel_size

        if attr['split'] in ['training', 'train']:
            points, feat, labels = self.augment.augment(
                points, feat, labels, self.cfg.get('augment', None))

        m = points.min(0)
        M = points.max(0)
        offset = -m + np.clip(4096 - M + m - 0.001, 0, None) * np.random.rand(
            3) + np.clip(4096 - M + m + 0.001, None, 0) * np.random.rand(3)

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
        cfg = self.cfg

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
        data = self.preprocess(data, {})

        self.inference_input = self.transform(
            tf.constant(data['point']), tf.constant(data['feat']),
            tf.constant(data['label']), tf.constant([data['point'].shape[0]]))

    def inference_preprocess(self):
        return self.inference_input

    def inference_end(self, results):
        results = results
        results = tf.reshape(results, [-1, self.cfg.num_classes])

        m_softmax = tf.keras.layers.Softmax(axis=-1)
        results = m_softmax(results)
        results = results.numpy()

        probs = np.reshape(results, [-1, self.cfg.num_classes])

        pred_l = np.argmax(probs, 1)

        self.inference_result = {
            'inference_labels': pred_l,
            'inference_scores': probs
        }

        return True

    def get_loss(self, Loss, results, inputs):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """
        cfg = self.cfg
        labels = inputs['label']

        scores, labels = Loss.filter_valid_label(results, labels)

        loss = Loss.weighted_CrossEntropyLoss(scores, labels)

        return loss, labels, scores

    def get_optimizer(self, cfg_pipeline):

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg_pipeline.learning_rate)

        return optimizer


MODEL._register_module(SparseConvUnet, 'tf')


class InputLayer(tf.keras.layers.Layer):

    def __init__(self, voxel_size=1.0):
        super(InputLayer, self).__init__()
        self.voxel_size = tf.constant([voxel_size, voxel_size, voxel_size])

    def rev_mapping_1(self, voxel_point_indices, length):
        length = length.shape[0]
        rev = np.zeros((length,))
        rev[voxel_point_indices] = np.arange(length)

        return rev.astype(np.int32)

    def rev_mapping_2(self, count):
        return np.repeat(np.arange(count.shape[0]), count).astype(np.int32)

    def call(self, features, inp_positions):
        v = voxelize(inp_positions, self.voxel_size, tf.constant([0., 0., 0.]),
                     tf.constant([40960., 40960., 40960.]))

        # Contiguous repeating positions.
        inp_positions = tf.gather(inp_positions, v.voxel_point_indices)
        features = tf.gather(features, v.voxel_point_indices)

        # Find reverse mapping.
        rev1 = tf.numpy_function(self.rev_mapping_1,
                                 [v.voxel_point_indices, inp_positions],
                                 tf.int32)

        # Unique positions.
        inp_positions = tf.gather(inp_positions, v.voxel_point_row_splits[:-1])

        # Mean of features.
        count = v.voxel_point_row_splits[1:] - v.voxel_point_row_splits[:-1]
        rev2 = tf.numpy_function(self.rev_mapping_2, [count], tf.int32)

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

        return features_avg, inp_positions, tf.gather(rev2, rev1)


class OutputLayer(tf.keras.layers.Layer):

    def __init__(self, voxel_size=1.0):
        super(OutputLayer, self).__init__()

    def call(self, features, rev):
        return tf.gather(features, rev)


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

    def call(self, features, inp_positions, out_positions=None, voxel_size=1.0):
        if out_positions is None:
            out_positions = inp_positions
        return self.net(features, inp_positions, out_positions, voxel_size)

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
    def tf_unique_2d(x):  # TODO : Compare with np.unique and tf.numpy_function
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
    def calculate_grid(inp_positions):
        filter = np.array([[-1, -1, -1], [-1, -1, 0], [-1, 0, -1], [-1, 0, 0],
                           [0, -1, -1], [0, -1, 0], [0, 0, -1], [0, 0, 0]])

        out_pos = inp_positions.astype(np.int32).repeat(filter.shape[0],
                                                        axis=0).reshape(-1, 3)
        filter = np.expand_dims(filter, 0).repeat(inp_positions.shape[0],
                                                  axis=0).reshape(-1, 3)

        out_pos = out_pos + filter
        out_pos = out_pos[out_pos.min(1) >= 0]
        out_pos = out_pos[(
            ~((out_pos.astype(np.int32) % 2).astype(np.bool)).any(1))]
        out_pos = np.unique(out_pos, axis=0)

        return (out_pos + 0.5).astype(np.float32)

    def call(self, features, inp_positions, voxel_size=1.0):
        out_positions = tf.numpy_function(self.calculate_grid, [inp_positions],
                                          tf.float32)
        out = self.net(features, inp_positions, out_positions, voxel_size)
        return out, out_positions / 2

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

    def call(self, features, inp_positions, out_positions, voxel_size=1.0):
        return self.net(features, inp_positions, out_positions, voxel_size)

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
        return tf.concat([feat_cat, feat], -1)


class NetworkInNetwork(tf.keras.layers.Layer):

    def __init__(self, nIn, nOut, bias=False):
        super(NetworkInNetwork, self).__init__()
        if nIn == nOut:
            self.linear = tf.keras.layers.Lambda(lambda x: x)
        else:
            self.linear = tf.keras.layers.Dense(nOut, use_bias=bias)

    def call(self, inputs):
        return self.linear(inputs)


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, nIn, nOut):
        super(ResidualBlock, self).__init__()

        self.lin = NetworkInNetwork(nIn, nOut)

        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.99,
                                                      epsilon=1e-4)
        self.relu1 = tf.keras.layers.LeakyReLU(0.)
        self.scn1 = SubmanifoldSparseConv(in_channels=nIn,
                                          filters=nOut,
                                          kernel_size=[3, 3, 3])

        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.99,
                                                      epsilon=1e-4)
        self.relu2 = tf.keras.layers.LeakyReLU(0.)
        self.scn2 = SubmanifoldSparseConv(in_channels=nOut,
                                          filters=nOut,
                                          kernel_size=[3, 3, 3])

    def call(self, feat, pos, training):
        out1 = self.lin(feat)

        feat = self.relu1(self.bn1(feat, training))
        feat = self.scn1(feat, pos)

        feat = self.relu2(self.bn2(feat, training))
        out2 = self.scn2(feat, pos)

        return out1 + out2

    def __name__(self):
        return "ResidualBlock"


class UNet(tf.keras.layers.Layer):

    def __init__(self,
                 reps,
                 nPlanes,
                 residual_blocks=False,
                 downsample=[2, 2],
                 leakiness=0):
        super(UNet, self).__init__()
        self.net = self.U(nPlanes, residual_blocks, reps)
        self.residual_blocks = residual_blocks

    @staticmethod
    def block(m, a, b, residual_blocks):
        if residual_blocks:
            m.append(ResidualBlock(a, b))
        else:
            m.append(
                tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-4))
            m.append(tf.keras.layers.LeakyReLU(0.))
            m.append(
                SubmanifoldSparseConv(in_channels=a,
                                      filters=b,
                                      kernel_size=[3, 3, 3]))

    @staticmethod
    def U(nPlanes, residual_blocks, reps):
        m = []
        for i in range(reps):
            UNet.block(m, nPlanes[0], nPlanes[0], residual_blocks)

        if len(nPlanes) > 1:
            m.append(ConcatFeat())
            m.append(
                tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-4))
            m.append(tf.keras.layers.LeakyReLU(0.))
            m.append(
                Convolution(in_channels=nPlanes[0],
                            filters=nPlanes[1],
                            kernel_size=[2, 2, 2]))
            m = m + UNet.U(nPlanes[1:], residual_blocks, reps)
            m.append(
                tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-4))
            m.append(tf.keras.layers.LeakyReLU(0.))
            m.append(
                DeConvolution(in_channels=nPlanes[1],
                              filters=nPlanes[0],
                              kernel_size=[2, 2, 2]))

            m.append(JoinFeat())

            for i in range(reps):
                UNet.block(m, nPlanes[0] * (2 if i == 0 else 1), nPlanes[0],
                           residual_blocks)

        return m

    def call(self, pos, feat, training=False):
        conv_pos = []
        concat_feat = []
        for module in self.net:
            if isinstance(module, tf.keras.layers.BatchNormalization):
                feat = module(feat, training=training)
            elif isinstance(module, tf.keras.layers.LeakyReLU):
                feat = module(feat)

            elif module.__name__() == "ResidualBlock":
                feat = module(feat, pos, training=training)

            elif module.__name__() == "SubmanifoldSparseConv":
                feat = module(feat, pos)

            elif module.__name__() == "Convolution":
                conv_pos.append(tf.identity(pos))
                feat, pos = module(feat, pos)
            elif module.__name__() == "DeConvolution":
                feat = module(feat, 2 * pos, conv_pos[-1])
                pos = conv_pos.pop()

            elif module.__name__() == "ConcatFeat":
                concat_feat.append(tf.identity(module(feat)))
            elif module.__name__() == "JoinFeat":
                feat = module(concat_feat.pop(), feat)

            else:
                raise Exception("Unknown module {}".format(module))

        return feat
