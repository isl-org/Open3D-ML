# Common libs
import time
import numpy as np
import pickle
import torch
import yaml
import math
from os import listdir
from os.path import exists, join, isdir

from ..models.kpconv import batch_grid_subsampling, batch_neighbors

from torch.utils.data import Sampler, get_worker_info


class KPConvBatch:
    """Batched results for KPConv."""

    def __init__(self, batches):
        """Initialize.

        Args:
            batches: A batch of data

        Returns:
            class: The corresponding class.
        """
        self.neighborhood_limits = []
        p_list = []
        f_list = []
        l_list = []
        fi_list = []
        p0_list = []
        s_list = []
        R_list = []
        r_inds_list = []
        r_mask_list = []
        val_labels_list = []
        batch_n = 0

        self.cfg = batches[0]['data']['cfg']
        batch_limit = int(self.cfg.batch_limit)

        for batch in batches:
            # Stack batch
            data = batch['data']

            for p in data['p_list']:
                batch_n += p.shape[0]
            if batch_n > batch_limit:
                break

            p_list += data['p_list']
            f_list += data['f_list']
            l_list += data['l_list']
            p0_list += data['p0_list']
            s_list += data['s_list']
            R_list += data['R_list']
            r_inds_list += data['r_inds_list']
            r_mask_list += data['r_mask_list']
            val_labels_list += data['val_labels_list']

        ###################
        # Concatenate batch
        ###################

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        frame_inds = np.array(fi_list, dtype=np.int32)
        frame_centers = np.stack(p0_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features (Use reflectance, input height or all coordinates)
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.cfg.in_features_dim == 1:
            pass
        elif self.cfg.in_features_dim == 2:
            # Use original height coordinate
            stacked_features = np.hstack((stacked_features, features[:, 2:3]))
        elif self.cfg.in_features_dim == 3:
            # Use height + reflectance
            assert features.shape[1] > 3, "feat from dataset can not be None \
                        or try to set in_features_dim = 1, 2, 4"

            stacked_features = np.hstack((stacked_features, features[:, 2:4]))
        elif self.cfg.in_features_dim == 4:
            # Use all coordinates
            stacked_features = np.hstack((stacked_features, features[:, :3]))
        elif self.cfg.in_features_dim == 5:
            assert features.shape[1] >= 6, "feat from dataset should have \
                    at least 3 dims, or try to set in_features_dim = 1, 2, 4"

            # Use color + height
            stacked_features = np.hstack((stacked_features, features[:, 2:6]))
        elif self.cfg.in_features_dim >= 6:

            assert features.shape[1] > 3, "feat from dataset can not be None \
                        or try to set in_features_dim = 1, 2, 4"

            # Use all coordinates + reflectance
            stacked_features = np.hstack((stacked_features, features))
        else:
            raise ValueError('in_features_dim should be >= 0')

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points, stacked_features,
                                              labels.astype(np.int64),
                                              stack_lengths)

        # Add scale and rotation for testing
        input_list += [
            scales, rots, frame_inds, frame_centers, r_inds_list, r_mask_list,
            val_labels_list
        ]

        input_list = [self.cfg.num_layers] + input_list

        # Number of layers
        L = int(input_list[0])

        # Extract input tensors from the list of numpy array
        ind = 1
        self.points = [
            torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]
        ]
        ind += L
        self.neighbors = [
            torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]
        ]

        ind += L
        self.pools = [
            torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]
        ]
        ind += L
        self.upsamples = [
            torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]
        ]
        ind += L
        self.lengths = [
            torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]
        ]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.frame_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.frame_centers = torch.from_numpy(input_list[ind])
        ind += 1
        self.reproj_inds = input_list[ind]
        ind += 1
        self.reproj_masks = input_list[ind]
        ind += 1
        self.val_labels = input_list[ind]

        return

    def big_neighborhood_filter(self, neighbors, layer):
        """Filter neighborhoods with max number of neighbors.

        Limit is set to keep XX% of the neighborhoods untouched. Limit is
        computed at initialization
        """
        # crop neighbors matrix
        if len(self.neighborhood_limits) > 0:
            return neighbors[:, :self.neighborhood_limits[layer]]
        else:
            return neighbors

    def segmentation_inputs(self, stacked_points, stacked_features, labels,
                            stack_lengths):

        # Starting radius of convolutions
        r_normal = self.cfg.first_subsampling_dl * self.cfg.conv_radius

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        input_upsamples = []
        input_stack_lengths = []
        deform_layers = []

        ######################
        # Loop over the blocks
        ######################

        arch = self.cfg.architecture

        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block or
                    'global' in block or 'upsample' in block):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            # *****************************

            deform_layer = False
            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks]):
                    r = r_normal * self.cfg.deform_radius / self.cfg.conv_radius
                    deform_layer = True
                else:
                    r = r_normal
                conv_i = batch_neighbors(stacked_points, stacked_points,
                                         stack_lengths, stack_lengths, r)

            else:
                # This layer only perform pooling, no neighbors required
                conv_i = np.zeros((0, 1), dtype=np.int32)

            # Pooling neighbors indices
            # *************************

            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / self.cfg.conv_radius

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling(stacked_points,
                                                        stack_lengths,
                                                        sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * self.cfg.deform_radius / self.cfg.conv_radius
                    deform_layer = True
                else:
                    r = r_normal

                # Subsample indices
                pool_i = batch_neighbors(pool_p, stacked_points, pool_b,
                                         stack_lengths, r)

                # Upsample indices (with the radius of the next layer to keep wanted density)
                up_i = batch_neighbors(stacked_points, pool_p, stack_lengths,
                                       pool_b, 2 * r)

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = np.zeros((0, 1), dtype=np.int32)
                pool_p = np.zeros((0, 3), dtype=np.float32)
                pool_b = np.zeros((0,), dtype=np.int32)
                up_i = np.zeros((0, 1), dtype=np.int32)

            # Reduce size of neighbors matrices by eliminating furthest point
            conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
            pool_i = self.big_neighborhood_filter(pool_i, len(input_points))
            if up_i.shape[0] > 0:
                up_i = self.big_neighborhood_filter(up_i, len(input_points) + 1)

            # Updating input lists
            input_points += [stacked_points]
            input_neighbors += [conv_i.astype(np.int64)]
            input_pools += [pool_i.astype(np.int64)]
            input_upsamples += [up_i.astype(np.int64)]
            input_stack_lengths += [stack_lengths]
            deform_layers += [deform_layer]

            # New points for next layer
            stacked_points = pool_p
            stack_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

        ###############
        # Return inputs
        ###############

        # list of network inputs
        li = input_points + input_neighbors + input_pools + input_upsamples + input_stack_lengths
        li += [stacked_features, labels]

        return li

    def pin_memory(self):
        """Manual pinning of the memory."""
        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [
            in_tensor.pin_memory() for in_tensor in self.neighbors
        ]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.upsamples = [
            in_tensor.pin_memory() for in_tensor in self.upsamples
        ]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.frame_inds = self.frame_inds.pin_memory()
        self.frame_centers = self.frame_centers.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.frame_inds = self.frame_inds.to(device)
        self.frame_centers = self.frame_centers.to(device)

        return self

    def unstack_points(self, layer=None):
        """Unstack the points."""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices."""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices."""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """Return a list of the stacked elements in the batch at a certain
        layer.

        If no layer is given, then return all layers
        """
        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i + 1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(
                            self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


class SparseConvUnetBatch:

    def __init__(self, batches):
        pc = []
        feat = []
        label = []
        lengths = []

        for batch in batches:
            data = batch['data']
            pc.append(data['point'])
            feat.append(data['feat'])
            label.append(data['label'])
            lengths.append(data['point'].shape[0])

        self.point = pc
        self.feat = feat
        self.label = label
        self.batch_lengths = lengths

    def pin_memory(self):
        self.point = [pc.pin_memory() for pc in self.point]
        self.feat = [feat.pin_memory() for feat in self.feat]
        self.label = [label.pin_memory() for label in self.label]
        return self

    def to(self, device):
        self.point = [pc.to(device) for pc in self.point]
        self.feat = [feat.to(device) for feat in self.feat]
        self.label = [label.to(device) for label in self.label]

    @staticmethod
    def scatter(batch, num_gpu):
        batch_size = len(batch.batch_lengths)

        new_batch_size = math.ceil(batch_size / num_gpu)
        batches = [SparseConvUnetBatch([]) for _ in range(num_gpu)]
        for i in range(num_gpu):
            start = new_batch_size * i
            end = min(new_batch_size * (i + 1), batch_size)
            batches[i].point = batch.point[start:end]
            batches[i].feat = batch.feat[start:end]
            batches[i].label = batch.label[start:end]
            batches[i].batch_lengths = batch.batch_lengths[start:end]

        return [b for b in batches if len(b.point)]  # filter empty batch


class PointTransformerBatch:

    def __init__(self, batches):
        pc = []
        feat = []
        label = []
        splits = [0]

        for batch in batches:
            data = batch['data']
            pc.append(data['point'])
            feat.append(data['feat'])
            label.append(data['label'])
            splits.append(splits[-1] + data['point'].shape[0])

        self.point = torch.cat(pc, 0)
        self.feat = torch.cat(feat, 0)
        self.label = torch.cat(label, 0)
        self.row_splits = torch.LongTensor(splits)

    def pin_memory(self):
        self.point = self.point.pin_memory()
        self.feat = self.feat.pin_memory()
        self.label = self.label.pin_memory()
        return self

    def to(self, device):
        self.point = self.point.to(device)
        self.feat = self.feat.to(device)
        self.label = self.label.to(device)


class ObjectDetectBatch:

    def __init__(self, batches):
        """Initialize.

        Args:
            batches: A batch of data

        Returns:
            class: The corresponding class.
        """
        self.point = []
        self.labels = []
        self.bboxes = []
        self.bbox_objs = []
        self.calib = []
        self.attr = []

        for batch in batches:
            data = batch['data']
            self.point.append(torch.tensor(data['point'], dtype=torch.float32))
            self.labels.append(
                torch.tensor(data['labels'], dtype=torch.int64) if 'labels' in
                data else None)
            if len(data.get('bboxes', [])) > 0:
                self.bboxes.append(
                    torch.tensor(data['bboxes'], dtype=torch.float32
                                ) if 'bboxes' in data else None)
            else:
                self.bboxes.append(torch.zeros((0, 7)))
            self.bbox_objs.append(data.get('bbox_objs'))
            self.calib.append(data.get('calib'))

    def pin_memory(self):
        for i in range(len(self.point)):
            self.point[i] = self.point[i].pin_memory()
            if self.labels[i] is not None:
                self.labels[i] = self.labels[i].pin_memory()
            if self.bboxes[i] is not None:
                self.bboxes[i] = self.bboxes[i].pin_memory()

        return self

    def to(self, device):
        for i in range(len(self.point)):
            self.point[i] = self.point[i].to(device)
            if self.labels[i] is not None:
                self.labels[i] = self.labels[i].to(device)
            if self.bboxes[i] is not None:
                self.bboxes[i] = self.bboxes[i].to(device)

    @staticmethod
    def scatter(batch, num_gpu):
        batch_size = len(batch.point)

        new_batch_size = math.ceil(batch_size / num_gpu)
        batches = [ObjectDetectBatch([]) for _ in range(num_gpu)]
        for i in range(num_gpu):
            start = new_batch_size * i
            end = min(new_batch_size * (i + 1), batch_size)
            batches[i].point = batch.point[start:end]
            batches[i].labels = batch.labels[start:end]
            batches[i].bboxes = batch.bboxes[start:end]
            batches[i].bbox_objs = batch.bbox_objs[start:end]
            batches[i].attr = batch.attr[start:end]

        return [b for b in batches if len(b.point)]  # filter empty batch


class ConcatBatcher(object):
    """ConcatBatcher for KPConv."""

    def __init__(self, device, model='KPConv'):
        """Initialize.

        Args:
            device: torch device 'gpu' or 'cpu'

        Returns:
            class: The corresponding class.
        """
        super(ConcatBatcher, self).__init__()
        self.device = device
        self.model = model

    def collate_fn(self, batches):
        """Collate function called by original PyTorch dataloader.

        Args:
            batches: a batch of data

        Returns:
            class: the batched result
        """
        if self.model == "KPConv" or self.model == "KPFCNN":
            batching_result = KPConvBatch(batches)
            batching_result.to(self.device)
            return {'data': batching_result, 'attr': []}

        elif self.model == "SparseConvUnet":
            return {'data': SparseConvUnetBatch(batches), 'attr': {}}

        elif self.model == "PointTransformer":
            return {'data': PointTransformerBatch(batches), 'attr': {}}

        elif self.model == "PointPillars" or self.model == "PointRCNN":
            batching_result = ObjectDetectBatch(batches)
            return batching_result

        else:
            raise Exception(
                f"Please define collate_fn for {self.model}, or use Default Batcher"
            )
