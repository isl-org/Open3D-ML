import time
import math
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_
from sklearn.neighbors import KDTree

from open3d.ml.contrib import subsample_batch
from open3d.ml.torch.layers import FixedRadiusSearch
from open3d.ml.torch.ops import ragged_to_dense

# use relative import for being compatible with Open3d main repo
from .base_model import BaseModel
from ..modules.losses import filter_valid_label
from ...utils import MODEL

from ...datasets.utils import (DataProcessing, trans_normalize, trans_augment,
                               trans_crop_pc, create_3D_rotations)


class bcolors:  # See https://stackoverflow.com/questions/287871
    WARNING = '\033[93m'
    ENDC = '\033[0m'


class KPFCNN(BaseModel):
    """Class defining KPFCNN.

    A model for Semantic Segmentation.
    """

    def __init__(
            self,
            name='KPFCNN',
            lbl_values=[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19
            ],
            num_classes=19,  # Number of valid classes
            ignored_label_inds=[0],
            ckpt_path=None,
            batcher='ConcatBatcher',
            architecture=[
                'simple', 'resnetb', 'resnetb_strided', 'resnetb', 'resnetb',
                'resnetb_strided', 'resnetb', 'resnetb', 'resnetb_strided',
                'resnetb', 'resnetb', 'resnetb_strided', 'resnetb',
                'nearest_upsample', 'unary', 'nearest_upsample', 'unary',
                'nearest_upsample', 'unary', 'nearest_upsample', 'unary'
            ],
            in_radius=4.0,
            max_in_points=100000,
            batch_num=8,
            batch_limit=30000,
            val_batch_num=8,
            num_kernel_points=15,
            first_subsampling_dl=0.06,
            conv_radius=2.5,
            deform_radius=6.0,
            KP_extent=1.2,
            KP_influence='linear',
            aggregation_mode='sum',
            first_features_dim=128,
            in_features_dim=2,
            modulated=False,
            use_batch_norm=True,
            batch_norm_momentum=0.02,
            deform_fitting_mode='point2point',
            deform_fitting_power=1.0,
            repulse_extent=1.2,
            augment_scale_anisotropic=True,
            augment_symmetries=[True, False, False],
            augment_rotation='vertical',
            augment_scale_min=0.8,
            augment_scale_max=1.2,
            augment_noise=0.001,
            augment_color=0.8,
            in_points_dim=3,
            fixed_kernel_points='center',
            num_layers=5,
            l_relu=0.1,
            reduce_fc=False,
            **kwargs):

        super().__init__(name=name,
                         lbl_values=lbl_values,
                         num_classes=num_classes,
                         ignored_label_inds=ignored_label_inds,
                         ckpt_path=ckpt_path,
                         batcher=batcher,
                         architecture=architecture,
                         in_radius=in_radius,
                         max_in_points=max_in_points,
                         batch_num=batch_num,
                         batch_limit=batch_limit,
                         val_batch_num=val_batch_num,
                         num_kernel_points=num_kernel_points,
                         first_subsampling_dl=first_subsampling_dl,
                         conv_radius=conv_radius,
                         deform_radius=deform_radius,
                         KP_extent=KP_extent,
                         KP_influence=KP_influence,
                         aggregation_mode=aggregation_mode,
                         first_features_dim=first_features_dim,
                         in_features_dim=in_features_dim,
                         modulated=modulated,
                         use_batch_norm=use_batch_norm,
                         batch_norm_momentum=batch_norm_momentum,
                         deform_fitting_mode=deform_fitting_mode,
                         deform_fitting_power=deform_fitting_power,
                         repulse_extent=repulse_extent,
                         augment_scale_anisotropic=augment_scale_anisotropic,
                         augment_symmetries=augment_symmetries,
                         augment_rotation=augment_rotation,
                         augment_scale_min=augment_scale_min,
                         augment_scale_max=augment_scale_max,
                         augment_noise=augment_noise,
                         augment_color=augment_color,
                         in_points_dim=in_points_dim,
                         fixed_kernel_points=fixed_kernel_points,
                         num_layers=num_layers,
                         l_relu=l_relu,
                         reduce_fc=reduce_fc,
                         **kwargs)

        cfg = self.cfg

        # Current radius of convolution and feature dimension
        layer = 0
        r = cfg.first_subsampling_dl * cfg.conv_radius
        in_dim = cfg.in_features_dim
        out_dim = cfg.first_features_dim
        lbl_values = cfg.lbl_values
        ign_lbls = cfg.ignored_label_inds
        self.K = cfg.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)

        # self.preprocess = None
        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        self.neighborhood_limits = []
        # Loop over consecutive blocks
        for block_i, block in enumerate(cfg.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError(
                    'Equivariant block but features dimension is not a factor of 3'
                )

            # Detect change to next layer for skip connection
            if np.any([
                    tmp in block
                    for tmp in ['pool', 'strided', 'upsample', 'global']
            ]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(
                block_decider(block, r, in_dim, out_dim, layer, cfg))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(cfg.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(cfg.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in cfg.architecture[start_i +
                                                              block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(
                block_decider(block, r, in_dim, out_dim, layer, cfg))

            # Update dimension of input from output
            in_dim = out_dim
            if block_i == 0 and cfg.reduce_fc:
                out_dim = out_dim // 2

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        if reduce_fc:
            self.head_mlp = UnaryBlock(out_dim,
                                       cfg.first_features_dim // 2,
                                       True,
                                       cfg.batch_norm_momentum,
                                       l_relu=cfg.get('l_relu', 0.1))
            self.head_softmax = UnaryBlock(cfg.first_features_dim // 2,
                                           self.C,
                                           False,
                                           1,
                                           no_relu=True,
                                           l_relu=cfg.get('l_relu', 0.1))
        else:
            self.head_mlp = UnaryBlock(out_dim,
                                       cfg.first_features_dim,
                                       False,
                                       0,
                                       l_relu=cfg.get('l_relu', 0.1))
            self.head_softmax = UnaryBlock(cfg.first_features_dim,
                                           self.C,
                                           False,
                                           0,
                                           l_relu=cfg.get('l_relu', 0.1))

        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort(
            [c for c in lbl_values if c not in ign_lbls])

        self.deform_fitting_mode = cfg.deform_fitting_mode
        self.deform_fitting_power = cfg.deform_fitting_power
        self.repulse_extent = cfg.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch):

        # Get input features
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x

    def get_optimizer(self, cfg_pipeline):
        # Optimizer with specific learning rate for deformable KPConv
        deform_params = [v for k, v in self.named_parameters() if 'offset' in k]
        other_params = [
            v for k, v in self.named_parameters() if 'offset' not in k
        ]
        deform_lr = cfg_pipeline.learning_rate * cfg_pipeline.deform_lr_factor
        optimizer = torch.optim.SGD([{
            'params': other_params
        }, {
            'params': deform_params,
            'lr': deform_lr
        }],
                                    lr=cfg_pipeline.learning_rate,
                                    momentum=cfg_pipeline.momentum,
                                    weight_decay=cfg_pipeline.weight_decay)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, cfg_pipeline.scheduler_gamma)

        return optimizer, scheduler

    def get_loss(self, Loss, results, inputs, device):
        """Runs the loss on outputs of the model.

        Args:
            outputs: logits
            labels: labels

        Returns:
            loss
        """
        cfg = self.cfg
        labels = inputs['data'].labels
        outputs = results

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(results, 0, 1).unsqueeze(0)
        labels = labels.unsqueeze(0)

        scores, labels = filter_valid_label(results, labels, cfg.num_classes,
                                            cfg.ignored_label_inds, device)

        # Cross entropy loss
        self.output_loss = Loss.weighted_CrossEntropyLoss(scores, labels)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' +
                             self.deform_fitting_mode)

        # Combined loss
        loss = self.output_loss + self.reg_loss

        return loss, labels, scores

    def preprocess(self, data, attr):
        cfg = self.cfg

        points = np.array(data['point'][:, 0:3], dtype=np.float32)

        if 'label' not in data.keys() or data['label'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))

        if 'feat' not in data.keys() or data['feat'] is None:
            feat = None
        else:
            feat = np.array(data['feat'], dtype=np.float32)

        split = attr['split']

        data = dict()

        if (feat is None):
            sub_points, sub_labels = DataProcessing.grid_subsampling(
                points, labels=labels, grid_size=cfg.first_subsampling_dl)
            sub_feat = None
        else:
            sub_points, sub_feat, sub_labels = DataProcessing.grid_subsampling(
                points,
                features=feat,
                labels=labels,
                grid_size=cfg.first_subsampling_dl)

        search_tree = KDTree(sub_points)

        data['point'] = sub_points
        data['feat'] = sub_feat
        data['label'] = sub_labels
        data['search_tree'] = search_tree

        if split in ["test", "testing", "validation", "valid"]:
            proj_inds = np.squeeze(
                search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            data['proj_inds'] = proj_inds

        return data

    def transform(self, data, attr, is_test=False):
        # Read points
        points = data['point']
        sem_labels = data['label']
        feat = data['feat']
        search_tree = data['search_tree']

        dim_points = points.shape[1]
        if feat is None:
            dim_features = dim_points
        else:
            dim_features = feat.shape[1] + dim_points

        # Initiate merged points
        merged_points = np.zeros((0, dim_points), dtype=np.float32)
        merged_labels = np.zeros((0,), dtype=np.int32)
        merged_coords = np.zeros((0, dim_features), dtype=np.float32)

        # Get center of the first frame in world coordinates
        p_origin = np.zeros((1, 4))
        p_origin[0, 3] = 1
        p0 = p_origin[:, :3]
        p0 = np.squeeze(p0)
        o_pts = None
        o_labels = None

        num_merged = 0

        result_data = {
            'p_list': [],
            'f_list': [],
            'l_list': [],
            'p0_list': [],
            's_list': [],
            'R_list': [],
            'r_inds_list': [],
            'r_mask_list': [],
            'val_labels_list': [],
            'cfg': self.cfg
        }

        curr_num_points = 0
        max_num_points = min(self.cfg.batch_limit, self.cfg.max_in_points)
        min_in_points = self.cfg.get('min_in_points', 3)
        min_in_points = min(min_in_points, self.cfg.max_in_points)

        while curr_num_points < min_in_points:

            new_points = points.copy()
            curr_new_points, mask_inds, p0 = self.trans_point_sampler(
                pc=new_points,
                feat=feat,
                label=sem_labels,
                search_tree=search_tree,
                num_points=min_in_points,
                radius=self.cfg.in_radius)

            curr_sem_labels = sem_labels[mask_inds]

            o_labels = sem_labels.astype(np.int32)
            # In case of validation, keep the original points in memory
            # if attr['split'] in ['test']:
            #     selected_points = curr_new_points.copy()
            #     o_pts = new_points
            #     o_labels = sem_labels.astype(np.int32)

            curr_new_points = curr_new_points - p0
            t_normalize = self.cfg.get('t_normalize', {})
            curr_new_points, curr_feat = trans_normalize(
                curr_new_points, feat, t_normalize)

            if curr_feat is None:
                curr_new_coords = curr_new_points.copy()
            else:
                curr_new_coords = np.hstack(
                    (curr_new_points, curr_feat[mask_inds, :]))

            in_pts = curr_new_points
            in_fts = curr_new_coords
            in_lbls = curr_sem_labels

            n = in_pts.shape[0]

            # Randomly drop some points (augmentation process and safety for GPU memory consumption)
            residual_num_points = max_num_points - curr_num_points
            if n > residual_num_points:
                input_inds = np.random.choice(n,
                                              size=residual_num_points,
                                              replace=False)
                in_pts = in_pts[input_inds, :]
                in_fts = in_fts[input_inds, :]
                in_lbls = in_lbls[input_inds]
                mask_inds = mask_inds[input_inds]
                n = input_inds.shape[0]

            curr_num_points += n

            reproj_mask = mask_inds
            if attr['split'] in ['test']:
                proj_inds = data['proj_inds']
            else:
                proj_inds = np.zeros((0,))
            # Before augmenting, compute reprojection inds (only for validation and test)
            # if attr['split'] in ['test', 'validation']:
            #     proj_inds = np.zeros((0,))
            #     reproj_mask = rand_order
            #     dists = np.sum(np.square(
            #         (o_pts[reproj_mask] - p0).astype(np.float32)),
            #                    axis=1)
            #     delta = np.square(1 - dists / (np.max(dists) + 0.001))

            #     self.possibility[reproj_mask] += delta

            # else:
            #     proj_inds = np.zeros((0,))
            #     reproj_mask = np.zeros((0,))

            # Data augmentation
            in_pts, scale, R = self.augmentation_transform(in_pts,
                                                           is_test=is_test)

            # Color augmentation
            if np.random.rand() > self.cfg.augment_color:
                in_fts[:, 3:] *= 0

            result_data['p_list'] += [in_pts]
            result_data['f_list'] += [in_fts]
            result_data['l_list'] += [np.squeeze(in_lbls)]
            result_data['p0_list'] += [p0]
            result_data['s_list'] += [scale]
            result_data['R_list'] += [R]
            result_data['r_inds_list'] += [proj_inds]
            result_data['r_mask_list'] += [reproj_mask]
            result_data['val_labels_list'] += [o_labels]

        return result_data

    def inference_begin(self, data):
        self.test_smooth = 0.98
        attr = {'split': 'test'}
        self.inference_ori_data = data
        self.inference_data = self.preprocess(data, attr)
        self.inference_proj_inds = self.inference_data['proj_inds']
        num_points = self.inference_data['search_tree'].data.shape[0]

        self.possibility = np.random.rand(num_points) * 1e-3
        self.test_probs = np.zeros(shape=[num_points, self.cfg.num_classes],
                                   dtype=np.float16)
        self.pbar = tqdm(total=self.possibility.shape[0])
        self.pbar_update = 0
        from ..dataloaders import ConcatBatcher
        self.batcher = ConcatBatcher(self.device)

    def inference_preprocess(self):
        attr = {'split': 'test'}
        data = self.transform(self.inference_data, attr, is_test=True)
        inputs = {'data': data, 'attr': attr}
        inputs = self.batcher.collate_fn([inputs])
        self.inference_input = inputs

        return inputs

    def update_probs(self, inputs, results, test_probs):
        self.test_smooth = 0.95
        stk_probs = torch.nn.functional.softmax(results, dim=-1)
        stk_probs = stk_probs.cpu().data.numpy()

        batch = inputs['data']
        stk_labels = batch.labels.cpu().data.numpy()

        # Get probs and labels
        lengths = batch.lengths[0].cpu().numpy()

        f_inds = batch.frame_inds.cpu().numpy()
        r_inds_list = batch.reproj_inds
        r_mask_list = batch.reproj_masks
        labels_list = batch.val_labels

        i0 = 0
        for b_i, length in enumerate(lengths):
            # Get prediction
            probs = stk_probs[i0:i0 + length]

            proj_inds = r_inds_list[b_i]
            proj_mask = r_mask_list[b_i]
            test_probs[proj_mask] = self.test_smooth * test_probs[proj_mask] + (
                1 - self.test_smooth) * probs
            i0 += length

        return test_probs

    def inference_end(self, inputs, results):
        m_softmax = torch.nn.Softmax(dim=-1)
        stk_probs = m_softmax(results)
        stk_probs = results.cpu().data.numpy()

        batch = inputs['data']

        # Get probs and labels
        lengths = batch.lengths[0].cpu().numpy()

        f_inds = batch.frame_inds.cpu().numpy()
        r_inds_list = batch.reproj_inds
        r_mask_list = batch.reproj_masks
        labels_list = batch.val_labels

        i0 = 0
        for b_i, length in enumerate(lengths):
            # Get prediction
            probs = stk_probs[i0:i0 + length]
            proj_inds = r_inds_list[b_i]
            proj_mask = r_mask_list[b_i]
            self.test_probs[proj_mask] = self.test_smooth * self.test_probs[
                proj_mask] + (1 - self.test_smooth) * probs
            i0 += length

        self.pbar.update(self.possibility[self.possibility > 0.5].shape[0] -
                         self.pbar_update)
        self.pbar_update = self.possibility[self.possibility > 0.5].shape[0]
        if np.min(self.possibility) > 0.5:
            self.pbar.close()
            pred_labels = np.argmax(self.test_probs, 1)

            pred_labels = pred_labels[self.inference_proj_inds]
            test_probs = self.test_probs[self.inference_proj_inds]
            inference_result = {
                'predict_labels': pred_labels,
                'predict_scores': test_probs
            }
            data = self.inference_ori_data
            acc = (pred_labels == data['label'] - 1).mean()

            self.inference_result = inference_result
            return True
        else:
            return False

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

    def augmentation_transform(self,
                               points,
                               normals=None,
                               verbose=False,
                               is_test=False):
        """Implementation of an augmentation transform for point clouds."""
        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if self.cfg.augment_rotation == 'vertical':

                # Create random rotations
                theta = np.random.rand() * 2 * np.pi
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]],
                             dtype=np.float32)

            elif self.cfg.augment_rotation == 'all':

                # Choose two random angles for the first vector in polar coordinates
                theta = np.random.rand() * 2 * np.pi
                phi = (np.random.rand() - 0.5) * np.pi

                # Create the first vector in carthesian coordinates
                u = np.array([
                    np.cos(theta) * np.cos(phi),
                    np.sin(theta) * np.cos(phi),
                    np.sin(phi)
                ])

                # Choose a random rotation angle
                alpha = np.random.rand() * 2 * np.pi

                # Create the rotation matrix with this vector and angle
                R = create_3D_rotations(np.reshape(u, (1, -1)),
                                        np.reshape(alpha, (1, -1)))[0]

        R = R.astype(np.float32)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = self.cfg.augment_scale_min
        max_s = self.cfg.augment_scale_max
        if self.cfg.augment_scale_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) - min_s

        # Add random symmetries to the scale factor
        symmetries = np.array(self.cfg.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        #######
        # Noise
        #######

        noise = (np.random.randn(points.shape[0], points.shape[1]) *
                 self.cfg.augment_noise).astype(np.float32)

        ##################
        # Apply transforms
        ##################

        # Do not use np.dot because it is multi-threaded
        #augmented_points = np.dot(points, R) * scale + noise
        augmented_points = np.sum(np.expand_dims(points, 2) * R,
                                  axis=1) * scale + noise

        if is_test:
            return points, scale, R

        if normals is None:
            return augmented_points, scale, R
        else:
            # Anisotropic scale of the normals thanks to cross product formula
            normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            augmented_normals = np.dot(normals, R) * normal_scale
            # Renormalise
            augmented_normals *= 1 / (
                np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

            if verbose:
                test_p = [np.vstack([points, augmented_points])]
                test_n = [np.vstack([normals, augmented_normals])]
                test_l = [
                    np.hstack(
                        [points[:, 2] * 0, augmented_points[:, 2] * 0 + 1])
                ]
                show_ModelNet_examples(test_p, test_n, test_l)

            return augmented_points, augmented_normals, scale, R


#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network blocks
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#

# ----------------------------------------------------------------------------------------------------------------------
#
#           Simple functions
#       \**********************/
#


def gather(x, idx, method=2):
    """Implementation of a custom gather operation for faster backwards.

    Args:
        x: Input with shape [N, D_1, ... D_d]
        idx: Indexing with shape [n_1, ..., n_m]
        method: Choice of the method

    Returns:
        x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """
    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i + 1)
            new_s = list(x.size())
            new_s[i + 1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i + n)
            new_s = list(idx.size())
            new_s[i + n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')


def radius_gaussian(sq_r, sig, eps=1e-9):
    """Compute a radius gaussian (gaussian of distance)

    Args:
        sq_r: input radiuses [dn, ..., d1, d0]
        sig: extents of gaussians [d1, d0] or [d0] or float

    Returns:
        gaussian of sq_r [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * sig**2 + eps))


def closest_pool(x, inds):
    """Pools features from the closest neighbors.

    WARNING: this function assumes the neighbors are ordered.

    Args:
        x: [n1, d] features matrix
        inds: [n2, max_num] Only the first column is used for pooling

    Returns:
        [n2, d] pooled features matrix
    """
    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get features for each pooling location [n2, d]
    return gather(x, inds[:, 0])


def max_pool(x, inds):
    """Pools features with the maximum values.

    Args:
        x: [n1, d] features matrix
        inds: [n2, max_num] pooling indices

    Returns:
        [n2, d] pooled features matrix
    """
    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get all features for each pooling location [n2, max_num, d]
    pool_features = gather(x, inds)

    # Pool the maximum [n2, d]
    max_features, _ = torch.max(pool_features, 1)
    return max_features


def global_average(x, batch_lengths):
    """Block performing a global average over batch pooling.

    Args:
        x: [N, D] input features
        batch_lengths: [B] list of batch lengths

    Returns:
        [B, D] averaged features
    """
    # Loop over the clouds of the batch
    averaged_features = []
    i0 = 0
    for b_i, length in enumerate(batch_lengths):

        # Average features for each batch cloud
        averaged_features.append(torch.mean(x[i0:i0 + length], dim=0))

        # Increment for next cloud
        i0 += length

    # Average features in each batch
    return torch.stack(averaged_features)


# ----------------------------------------------------------------------------------------------------------------------
#
#           KPConv class
#       \******************/
#


class KPConv(nn.Module):

    def __init__(self,
                 kernel_size,
                 p_dim,
                 in_channels,
                 out_channels,
                 KP_extent,
                 radius,
                 fixed_kernel_points='center',
                 KP_influence='linear',
                 aggregation_mode='sum',
                 deformable=False,
                 modulated=False):
        """Initialize parameters for KPConvDeformable.

        Args:
            kernel_size: Number of kernel points.
            p_dim: dimension of the point space.
            in_channels: dimension of input features.
            out_channels: dimension of output features.
            KP_extent: influence radius of each kernel point.
            radius: radius used for kernel point init. Even for deformable, use the config.conv_radius
            fixed_kernel_points: fix position of certain kernel points ('none', 'center' or 'verticals').
            KP_influence: influence function of the kernel points ('constant', 'linear', 'gaussian').
            aggregation_mode: choose to sum influences, or only keep the closest ('closest', 'sum').
            deformable: choose deformable or not
            modulated: choose if kernel weights are modulated in addition to deformed
        """
        super(KPConv, self).__init__()

        # Save parameters
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated

        # Running variable containing deformed KP distance to input points. (used in regularization loss)
        self.min_d2 = None
        self.deformed_KP = None
        self.offset_features = None

        # Initialize weights
        self.weights = Parameter(torch.zeros(
            (self.K, in_channels, out_channels), dtype=torch.float32),
                                 requires_grad=True)

        # Initiate weights for offsets
        if deformable:
            if modulated:
                self.offset_dim = (self.p_dim + 1) * self.K
            else:
                self.offset_dim = self.p_dim * self.K
            self.offset_conv = KPConv(self.K,
                                      self.p_dim,
                                      self.in_channels,
                                      self.offset_dim,
                                      KP_extent,
                                      radius,
                                      fixed_kernel_points=fixed_kernel_points,
                                      KP_influence=KP_influence,
                                      aggregation_mode=aggregation_mode)
            self.offset_bias = Parameter(torch.zeros(self.offset_dim,
                                                     dtype=torch.float32),
                                         requires_grad=True)

        else:
            self.offset_dim = None
            self.offset_conv = None
            self.offset_bias = None

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        # self.kernel_points = self.init_KP()

        if deformable:
            self.kernel_points = self.offset_conv.kernel_points
        else:
            self.kernel_points = self.init_KP()

        return

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.deformable:
            nn.init.zeros_(self.offset_bias)
        return

    def init_KP(self):
        """Initialize the kernel point positions in a sphere

        Returns:
            the tensor of kernel points
        """
        # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
        K_points_numpy = load_kernels(self.radius,
                                      self.K,
                                      dimension=self.p_dim,
                                      fixed=self.fixed_kernel_points)

        return Parameter(torch.tensor(K_points_numpy, dtype=torch.float32),
                         requires_grad=False)

    def forward(self, q_pts, s_pts, neighb_inds, x):

        ###################
        # Offset generation
        ###################

        if self.deformable:

            # Get offsets with a KPConv that only takes part of the features
            self.offset_features = self.offset_conv(q_pts, s_pts, neighb_inds,
                                                    x) + self.offset_bias

            if self.modulated:

                # Get offset (in normalized scale) from features
                unscaled_offsets = self.offset_features[:, :self.p_dim * self.K]
                unscaled_offsets = unscaled_offsets.view(-1, self.K, self.p_dim)

                # Get modulations
                modulations = 2 * torch.sigmoid(
                    self.offset_features[:, self.p_dim * self.K:])

            else:

                # Get offset (in normalized scale) from features
                unscaled_offsets = self.offset_features.view(
                    -1, self.K, self.p_dim)

                # No modulations
                modulations = None

            # Rescale offset for this layer
            offsets = unscaled_offsets * self.KP_extent

        else:
            offsets = None
            modulations = None

        ######################
        # Deformed convolution
        ######################

        # Add a fake point in the last row for shadow neighbors
        s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + 1e6), 0)

        # Get neighbor points [n_points, n_neighbors, dim]
        neighbors = s_pts[neighb_inds, :]

        # Center every neighborhood
        neighbors = neighbors - q_pts.unsqueeze(1)

        # Apply offsets to kernel points [n_points, n_kpoints, dim]
        if self.deformable:
            self.deformed_KP = offsets + self.kernel_points
            deformed_K_points = self.deformed_KP.unsqueeze(1)
        else:
            deformed_K_points = self.kernel_points

        # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
        neighbors.unsqueeze_(2)
        differences = neighbors - deformed_K_points

        # Get the square distances [n_points, n_neighbors, n_kpoints]
        sq_distances = torch.sum(differences**2, dim=3)

        # Optimization by ignoring points outside a deformed KP range
        if self.deformable:

            # Save distances for loss
            self.min_d2, _ = torch.min(sq_distances, dim=1)

            # Boolean of the neighbors in range of a kernel point [n_points, n_neighbors]
            in_range = torch.any(sq_distances < self.KP_extent**2,
                                 dim=2).type(torch.int32)

            # New value of max neighbors
            new_max_neighb = torch.max(torch.sum(in_range, dim=1))

            # For each row of neighbors, indices of the ones that are in range [n_points, new_max_neighb]
            neighb_row_bool, neighb_row_inds = torch.topk(in_range,
                                                          new_max_neighb.item(),
                                                          dim=1)

            # Gather new neighbor indices [n_points, new_max_neighb]
            new_neighb_inds = neighb_inds.gather(1,
                                                 neighb_row_inds,
                                                 sparse_grad=False)

            # Gather new distances to KP [n_points, new_max_neighb, n_kpoints]
            neighb_row_inds.unsqueeze_(2)
            neighb_row_inds = neighb_row_inds.expand(-1, -1, self.K)
            sq_distances = sq_distances.gather(1,
                                               neighb_row_inds,
                                               sparse_grad=False)

            # New shadow neighbors have to point to the last shadow point
            new_neighb_inds *= neighb_row_bool
            new_neighb_inds -= (neighb_row_bool.type(torch.int64) -
                                1) * int(s_pts.shape[0] - 1)
        else:
            new_neighb_inds = neighb_inds

        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)
            all_weights = torch.transpose(all_weights, 1, 2)

        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = torch.clamp(1 -
                                      torch.sqrt(sq_distances) / self.KP_extent,
                                      min=0.0)
            all_weights = torch.transpose(all_weights, 1, 2)

        elif self.KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            sigma = self.KP_extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = torch.transpose(all_weights, 1, 2)
        else:
            raise ValueError(
                'Unknown influence function type (config.KP_influence)')

        # In case of closest mode, only the closest KP can influence each point
        if self.aggregation_mode == 'closest':
            neighbors_1nn = torch.argmin(sq_distances, dim=2)
            all_weights *= torch.transpose(
                nn.functional.one_hot(neighbors_1nn, self.K), 1, 2)

        elif self.aggregation_mode != 'sum':
            raise ValueError(
                "Unknown convolution mode. Should be 'closest' or 'sum'")

        # Add a zero feature for shadow neighbors
        x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        neighb_x = gather(x, new_neighb_inds)

        # Apply distance weights [n_points, n_kpoints, in_fdim]

        weighted_features = torch.matmul(all_weights, neighb_x)

        # Apply modulations
        if self.deformable and self.modulated:
            weighted_features *= modulations.unsqueeze(2)

        # Apply network weights [n_kpoints, n_points, out_fdim]
        weighted_features = weighted_features.permute((1, 0, 2))

        kernel_outputs = torch.matmul(weighted_features, self.weights)

        # Convolution sum [n_points, out_fdim]
        return torch.sum(kernel_outputs, dim=0)

    def __repr__(self):
        return 'KPConv(radius: {:.2f}, in_feat: {:d}, out_feat: {:d})'.format(
            self.radius, self.in_channels, self.out_channels)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Complex blocks
#       \********************/
#


def block_decider(block_name, radius, in_dim, out_dim, layer_ind, config):

    if block_name == 'unary':
        return UnaryBlock(in_dim,
                          out_dim,
                          config.use_batch_norm,
                          config.batch_norm_momentum,
                          l_relu=config.get('l_relu', 0.1))

    elif block_name in [
            'simple', 'simple_deformable', 'simple_invariant',
            'simple_equivariant', 'simple_strided', 'simple_deformable_strided',
            'simple_invariant_strided', 'simple_equivariant_strided'
    ]:
        return SimpleBlock(block_name, in_dim, out_dim, radius, layer_ind,
                           config)

    elif block_name in [
            'resnetb', 'resnetb_invariant', 'resnetb_equivariant',
            'resnetb_deformable', 'resnetb_strided',
            'resnetb_deformable_strided', 'resnetb_equivariant_strided',
            'resnetb_invariant_strided'
    ]:
        return ResnetBottleneckBlock(block_name, in_dim, out_dim, radius,
                                     layer_ind, config)

    elif block_name == 'max_pool' or block_name == 'max_pool_wide':
        return MaxPoolBlock(layer_ind)

    elif block_name == 'global_average':
        return GlobalAverageBlock()

    elif block_name == 'nearest_upsample':
        return NearestUpsampleBlock(layer_ind)

    else:
        raise ValueError(
            'Unknown block name in the architecture definition : ' + block_name)


class BatchNormBlock(nn.Module):

    def __init__(self, in_dim, use_bn, bn_momentum):
        """Initialize a batch normalization block.

        If network does not use batch normalization, replace with biases.

        Args:
            in_dim: dimension input features
            use_bn: boolean indicating if we use Batch Norm
            bn_momentum: Batch norm momentum
        """
        super(BatchNormBlock, self).__init__()
        self.bn_momentum = 1 - bn_momentum
        self.use_bn = use_bn
        self.in_dim = in_dim
        if self.use_bn:
            self.batch_norm = nn.BatchNorm1d(in_dim, momentum=1 - bn_momentum)
        else:
            self.bias = Parameter(torch.zeros(in_dim, dtype=torch.float32),
                                  requires_grad=True)
        return

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.use_bn:

            x = x.unsqueeze(2)
            x = x.transpose(0, 2)
            x = self.batch_norm(x)
            x = x.transpose(0, 2)
            return x.squeeze(2)
        else:
            return x + self.bias

    def __repr__(self):
        return 'BatchNormBlock(in_feat: {:d}, momentum: {:.3f}, only_bias: {:s})'.format(
            self.in_dim, self.bn_momentum, str(not self.use_bn))


class UnaryBlock(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 use_bn,
                 bn_momentum,
                 no_relu=False,
                 l_relu=0.1):
        """Initialize a standard unary block with its ReLU and BatchNorm.

        Args:
            in_dim: dimension input features
            out_dim: dimension input features
            use_bn: boolean indicating if we use Batch Norm
            bn_momentum: Batch norm momentum
            no_relu: Do not use leaky ReLU
            l_relu: Leaky ReLU factor
        """
        super(UnaryBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.no_relu = no_relu
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Linear(in_dim, out_dim, bias=False)
        self.batch_norm = BatchNormBlock(out_dim, self.use_bn, self.bn_momentum)
        if not no_relu:
            self.leaky_relu = nn.LeakyReLU(l_relu)
        return

    def forward(self, x, batch=None):
        x = self.mlp(x)
        x = self.batch_norm(x)
        if not self.no_relu:
            x = self.leaky_relu(x)
        return x

    def __repr__(self):
        return 'UnaryBlock(in_feat: {:d}, out_feat: {:d}, BN: {:s}, ReLU: {:s})'.format(
            self.in_dim, self.out_dim, str(self.use_bn), str(not self.no_relu))


class SimpleBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        """Initialize a simple convolution block with its ReLU and BatchNorm.

        Args:
            block_name: Block name
            in_dim: dimension input features
            out_dim: dimension input features
            radius: current radius of convolution
            layer_ind: Index for layer
            config: parameters
        """
        super(SimpleBlock, self).__init__()

        # get KP_extent from current radius
        current_extent = radius * config.KP_extent / config.conv_radius

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.layer_ind = layer_ind
        self.block_name = block_name
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Define the KPConv class
        self.KPConv = KPConv(config.num_kernel_points,
                             config.in_points_dim,
                             in_dim,
                             out_dim // 2,
                             current_extent,
                             radius,
                             fixed_kernel_points=config.fixed_kernel_points,
                             KP_influence=config.KP_influence,
                             aggregation_mode=config.aggregation_mode,
                             deformable='deform' in block_name,
                             modulated=config.modulated)

        # Other operations
        self.batch_norm = BatchNormBlock(out_dim // 2, self.use_bn,
                                         self.bn_momentum)
        self.leaky_relu = nn.LeakyReLU(config.get('l_relu', 0.1))

        return

    def forward(self, x, batch):

        if 'strided' in self.block_name:
            q_pts = batch.points[self.layer_ind + 1]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.pools[self.layer_ind]

        else:
            q_pts = batch.points[self.layer_ind]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.neighbors[self.layer_ind]

        x = self.KPConv(q_pts, s_pts, neighb_inds, x)
        return self.leaky_relu(self.batch_norm(x))


class ResnetBottleneckBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        """Initialize a resnet bottleneck block.

        Args:
            block_name: Block name
            in_dim: dimension input features
            out_dim: dimension input features
            radius: current radius of convolution
            layer_ind: Layer ind
            config: parameters
        """
        super(ResnetBottleneckBlock, self).__init__()

        # get KP_extent from current radius
        current_extent = radius * config.KP_extent / config.conv_radius

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.block_name = block_name
        self.layer_ind = layer_ind
        self.in_dim = in_dim
        self.out_dim = out_dim
        l_relu = config.get('l_relu', 0.1)

        # First downscaling mlp
        if in_dim != out_dim // 4:
            self.unary1 = UnaryBlock(in_dim,
                                     out_dim // 4,
                                     self.use_bn,
                                     self.bn_momentum,
                                     l_relu=l_relu)
        else:
            self.unary1 = nn.Identity()

        # KPConv block
        self.KPConv = KPConv(config.num_kernel_points,
                             config.in_points_dim,
                             out_dim // 4,
                             out_dim // 4,
                             current_extent,
                             radius,
                             fixed_kernel_points=config.fixed_kernel_points,
                             KP_influence=config.KP_influence,
                             aggregation_mode=config.aggregation_mode,
                             deformable='deform' in block_name,
                             modulated=config.modulated)
        self.batch_norm_conv = BatchNormBlock(out_dim // 4, self.use_bn,
                                              self.bn_momentum)

        # Second upscaling mlp
        self.unary2 = UnaryBlock(out_dim // 4,
                                 out_dim,
                                 self.use_bn,
                                 self.bn_momentum,
                                 no_relu=True,
                                 l_relu=l_relu)

        # Shortcut optional mpl
        if in_dim != out_dim:
            self.unary_shortcut = UnaryBlock(in_dim,
                                             out_dim,
                                             self.use_bn,
                                             self.bn_momentum,
                                             no_relu=True,
                                             l_relu=l_relu)
        else:
            self.unary_shortcut = nn.Identity()

        # Other operations
        self.leaky_relu = nn.LeakyReLU(l_relu)

        return

    def forward(self, features, batch):

        if 'strided' in self.block_name:
            q_pts = batch.points[self.layer_ind + 1]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.pools[self.layer_ind]
        else:
            q_pts = batch.points[self.layer_ind]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.neighbors[self.layer_ind]

        # First downscaling mlp
        x = self.unary1(features)

        # Convolution
        x = self.KPConv(q_pts, s_pts, neighb_inds, x)
        x = self.leaky_relu(self.batch_norm_conv(x))

        # Second upscaling mlp
        x = self.unary2(x)

        # Shortcut
        if 'strided' in self.block_name:
            shortcut = max_pool(features, neighb_inds)
        else:
            shortcut = features
        shortcut = self.unary_shortcut(shortcut)

        return self.leaky_relu(x + shortcut)


class GlobalAverageBlock(nn.Module):

    def __init__(self):
        """Initialize a global average block with its ReLU and BatchNorm."""
        super(GlobalAverageBlock, self).__init__()
        return

    def forward(self, x, batch):
        return global_average(x, batch.lengths[-1])


class NearestUpsampleBlock(nn.Module):

    def __init__(self, layer_ind):
        """Initialize a nearest upsampling block with its ReLU and BatchNorm."""
        super(NearestUpsampleBlock, self).__init__()
        self.layer_ind = layer_ind
        return

    def forward(self, x, batch):
        return closest_pool(x, batch.upsamples[self.layer_ind - 1])

    def __repr__(self):
        return 'NearestUpsampleBlock(layer: {:d} -> {:d})'.format(
            self.layer_ind, self.layer_ind - 1)


class MaxPoolBlock(nn.Module):

    def __init__(self, layer_ind):
        """Initialize a max pooling block with its ReLU and BatchNorm."""
        super(MaxPoolBlock, self).__init__()
        self.layer_ind = layer_ind
        return

    def forward(self, x, batch):
        return max_pool(x, batch.pools[self.layer_ind + 1])


#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Functions handling the disposition of kernel points.
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#

# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#

# Import numpy package and name it "np"
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from os import makedirs
from os.path import join, exists

# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#


def spherical_Lloyd(radius,
                    num_cells,
                    dimension=3,
                    fixed='center',
                    approximation='monte-carlo',
                    approx_n=5000,
                    max_iter=500,
                    momentum=0.9,
                    verbose=0):
    """Creation of kernel point via Lloyd algorithm. We use an approximation of
    the algorithm, and compute the Voronoi cell centers with discretization  of
    space. The exact formula is not trivial with part of the sphere as sides.

    Args:
        radius: Radius of the kernels
        num_cells: Number of cell (kernel points) in the Voronoi diagram.
        dimension: dimension of the space
        fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
        approximation: Approximation method for Lloyd's algorithm ('discretization', 'monte-carlo')
        approx_n: Number of point used for approximation.
        max_iter: Maximum nu;ber of iteration for the algorithm.
        momentum: Momentum of the low pass filter smoothing kernel point positions
        verbose: display option

    Returns:
        points [num_kernels, num_points, dimension]
    """
    #######################
    # Parameters definition
    #######################

    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1.0

    #######################
    # Kernel initialization
    #######################

    # Random kernel points (Uniform distribution in a sphere)
    kernel_points = np.zeros((0, dimension))
    while kernel_points.shape[0] < num_cells:
        new_points = np.random.rand(num_cells,
                                    dimension) * 2 * radius0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[np.logical_and(d2 < radius0**2,
                                                     (0.9 *
                                                      radius0)**2 < d2), :]
    kernel_points = kernel_points[:num_cells, :].reshape((num_cells, -1))

    # Optional fixing
    if fixed == 'center':
        kernel_points[0, :] *= 0
    if fixed == 'verticals':
        kernel_points[:3, :] *= 0
        kernel_points[1, -1] += 2 * radius0 / 3
        kernel_points[2, -1] -= 2 * radius0 / 3

    ##############################
    # Approximation initialization
    ##############################

    # Initialize figure
    if verbose > 1:
        fig = plt.figure()

    # Initialize discretization in this method is chosen
    if approximation == 'discretization':
        side_n = int(np.floor(approx_n**(1. / dimension)))
        dl = 2 * radius0 / side_n
        coords = np.arange(-radius0 + dl / 2, radius0, dl)
        if dimension == 2:
            x, y = np.meshgrid(coords, coords)
            X = np.vstack((np.ravel(x), np.ravel(y))).T
        elif dimension == 3:
            x, y, z = np.meshgrid(coords, coords, coords)
            X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z))).T
        elif dimension == 4:
            x, y, z, t = np.meshgrid(coords, coords, coords, coords)
            X = np.vstack(
                (np.ravel(x), np.ravel(y), np.ravel(z), np.ravel(t))).T
        else:
            raise ValueError('Unsupported dimension (max is 4)')
    elif approximation == 'monte-carlo':
        X = np.zeros((0, dimension))
    else:
        raise ValueError(
            'Wrong approximation method chosen: "{:s}"'.format(approximation))

    # Only points inside the sphere are used
    d2 = np.sum(np.power(X, 2), axis=1)
    X = X[d2 < radius0 * radius0, :]

    #####################
    # Kernel optimization
    #####################

    # Warning if at least one kernel point has no cell
    warning = False

    # moving vectors of kernel points saved to detect convergence
    max_moves = np.zeros((0,))

    for iter in range(max_iter):

        # In the case of monte-carlo, renew the sampled points
        if approximation == 'monte-carlo':
            X = np.random.rand(approx_n, dimension) * 2 * radius0 - radius0
            d2 = np.sum(np.power(X, 2), axis=1)
            X = X[d2 < radius0 * radius0, :]

        # Get the distances matrix [n_approx, K, dim]
        differences = np.expand_dims(X, 1) - kernel_points
        sq_distances = np.sum(np.square(differences), axis=2)

        # Compute cell centers
        cell_inds = np.argmin(sq_distances, axis=1)
        centers = []
        for c in range(num_cells):
            bool_c = (cell_inds == c)
            num_c = np.sum(bool_c.astype(np.int32))
            if num_c > 0:
                centers.append(np.sum(X[bool_c, :], axis=0) / num_c)
            else:
                warning = True
                centers.append(kernel_points[c])

        # Update kernel points with low pass filter to smooth mote carlo
        centers = np.vstack(centers)
        moves = (1 - momentum) * (centers - kernel_points)
        kernel_points += moves

        # Check moves for convergence
        max_moves = np.append(max_moves, np.max(np.linalg.norm(moves, axis=1)))

        # Optional fixing
        if fixed == 'center':
            kernel_points[0, :] *= 0
        if fixed == 'verticals':
            kernel_points[0, :] *= 0
            kernel_points[:3, :-1] *= 0

        if verbose:
            print('iter {:5d} / max move = {:f}'.format(
                iter, np.max(np.linalg.norm(moves, axis=1))))
            if warning:
                print('{:}WARNING: at least one point has no cell{:}'.format(
                    bcolors.WARNING, bcolors.ENDC))
        if verbose > 1:
            plt.clf()
            plt.scatter(X[:, 0],
                        X[:, 1],
                        c=cell_inds,
                        s=20.0,
                        marker='.',
                        cmap=plt.get_cmap('tab20'))
            #plt.scatter(kernel_points[:, 0], kernel_points[:, 1], c=np.arange(num_cells), s=100.0,
            #            marker='+', cmap=plt.get_cmap('tab20'))
            plt.plot(kernel_points[:, 0], kernel_points[:, 1], 'k+')
            circle = plt.Circle((0, 0), radius0, color='r', fill=False)
            fig.axes[0].add_artist(circle)
            fig.axes[0].set_xlim((-radius0 * 1.1, radius0 * 1.1))
            fig.axes[0].set_ylim((-radius0 * 1.1, radius0 * 1.1))
            fig.axes[0].set_aspect('equal')
            plt.draw()
            plt.pause(0.001)
            plt.show(block=False)

    ###################
    # User verification
    ###################

    # Show the convergence to ask user if this kernel is correct
    if verbose:
        if dimension == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[10.4, 4.8])
            ax1.plot(max_moves)
            ax2.scatter(X[:, 0],
                        X[:, 1],
                        c=cell_inds,
                        s=20.0,
                        marker='.',
                        cmap=plt.get_cmap('tab20'))
            # plt.scatter(kernel_points[:, 0], kernel_points[:, 1], c=np.arange(num_cells), s=100.0,
            #            marker='+', cmap=plt.get_cmap('tab20'))
            ax2.plot(kernel_points[:, 0], kernel_points[:, 1], 'k+')
            circle = plt.Circle((0, 0), radius0, color='r', fill=False)
            ax2.add_artist(circle)
            ax2.set_xlim((-radius0 * 1.1, radius0 * 1.1))
            ax2.set_ylim((-radius0 * 1.1, radius0 * 1.1))
            ax2.set_aspect('equal')
            plt.title('Check if kernel is correct.')
            plt.draw()
            plt.show()

        if dimension > 2:
            plt.figure()
            plt.plot(max_moves)
            plt.title('Check if kernel is correct.')
            plt.show()

    # Rescale kernels with real radius
    return kernel_points * radius


def kernel_point_optimization_debug(radius,
                                    num_points,
                                    num_kernels=1,
                                    dimension=3,
                                    fixed='center',
                                    ratio=0.66,
                                    verbose=0):
    """Creation of kernel point via optimization of potentials.

    Args:
        radius: Radius of the kernels
        num_points: points composing kernels
        num_kernels: number of wanted kernels
        dimension: dimension of the space
        fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
        ratio: ratio of the radius where you want the kernels points to be placed
        verbose: display option

    Returns:
        points [num_kernels, num_points, dimension]
    """
    #######################
    # Parameters definition
    #######################

    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1
    diameter0 = 2

    # Factor multiplicating gradients for moving points (~learning rate)
    moving_factor = 1e-2
    continuous_moving_decay = 0.9995

    # Gradient threshold to stop optimization
    thresh = 1e-5

    # Gradient clipping value
    clip = 0.05 * radius0

    #######################
    # Kernel initialization
    #######################

    # Random kernel points
    kernel_points = np.random.rand(num_kernels * num_points - 1,
                                   dimension) * diameter0 - radius0
    while (kernel_points.shape[0] < num_kernels * num_points):
        new_points = np.random.rand(num_kernels * num_points - 1,
                                    dimension) * diameter0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[d2 < 0.5 * radius0 * radius0, :]
    kernel_points = kernel_points[:num_kernels * num_points, :].reshape(
        (num_kernels, num_points, -1))

    # Optional fixing
    if fixed == 'center':
        kernel_points[:, 0, :] *= 0
    if fixed == 'verticals':
        kernel_points[:, :3, :] *= 0
        kernel_points[:, 1, -1] += 2 * radius0 / 3
        kernel_points[:, 2, -1] -= 2 * radius0 / 3

    #####################
    # Kernel optimization
    #####################

    # Initialize figure
    if verbose > 1:
        fig = plt.figure()

    saved_gradient_norms = np.zeros((10000, num_kernels))
    old_gradient_norms = np.zeros((num_kernels, num_points))
    for iter in range(10000):

        # Compute gradients
        # *****************

        # Derivative of the sum of potentials of all points
        A = np.expand_dims(kernel_points, axis=2)
        B = np.expand_dims(kernel_points, axis=1)
        interd2 = np.sum(np.power(A - B, 2), axis=-1)
        inter_grads = (A - B) / (np.power(np.expand_dims(interd2, -1), 3 / 2) +
                                 1e-6)
        inter_grads = np.sum(inter_grads, axis=1)

        # Derivative of the radius potential
        circle_grads = 10 * kernel_points

        # All gradients
        gradients = inter_grads + circle_grads

        if fixed == 'verticals':
            gradients[:, 1:3, :-1] = 0

        # Stop condition
        # **************

        # Compute norm of gradients
        gradients_norms = np.sqrt(np.sum(np.power(gradients, 2), axis=-1))
        saved_gradient_norms[iter, :] = np.max(gradients_norms, axis=1)

        # Stop if all moving points are gradients fixed (low gradients diff)

        if fixed == 'center' and np.max(
                np.abs(old_gradient_norms[:, 1:] -
                       gradients_norms[:, 1:])) < thresh:
            break
        elif fixed == 'verticals' and np.max(
                np.abs(old_gradient_norms[:, 3:] -
                       gradients_norms[:, 3:])) < thresh:
            break
        elif np.max(np.abs(old_gradient_norms - gradients_norms)) < thresh:
            break
        old_gradient_norms = gradients_norms

        # Move points
        # ***********

        # Clip gradient to get moving dists
        moving_dists = np.minimum(moving_factor * gradients_norms, clip)

        # Fix central point
        if fixed == 'center':
            moving_dists[:, 0] = 0
        if fixed == 'verticals':
            moving_dists[:, 0] = 0

        # Move points
        kernel_points -= np.expand_dims(moving_dists,
                                        -1) * gradients / np.expand_dims(
                                            gradients_norms + 1e-6, -1)

        if verbose:
            print('iter {:5d} / max grad = {:f}'.format(
                iter, np.max(gradients_norms[:, 3:])))
        if verbose > 1:
            plt.clf()
            plt.plot(kernel_points[0, :, 0], kernel_points[0, :, 1], '.')
            circle = plt.Circle((0, 0), radius, color='r', fill=False)
            fig.axes[0].add_artist(circle)
            fig.axes[0].set_xlim((-radius * 1.1, radius * 1.1))
            fig.axes[0].set_ylim((-radius * 1.1, radius * 1.1))
            fig.axes[0].set_aspect('equal')
            plt.draw()
            plt.pause(0.001)
            plt.show(block=False)
            print(moving_factor)

        # moving factor decay
        moving_factor *= continuous_moving_decay

    # Rescale radius to fit the wanted ratio of radius
    r = np.sqrt(np.sum(np.power(kernel_points, 2), axis=-1))
    kernel_points *= ratio / np.mean(r[:, 1:])

    # Rescale kernels with real radius
    return kernel_points * radius, saved_gradient_norms


def load_kernels(radius, num_kpoints, dimension, fixed, lloyd=False):

    # Kernel directory
    kernel_dir = 'kernels/dispositions'
    if not exists(kernel_dir):
        makedirs(kernel_dir)

    # To many points switch to Lloyds
    if num_kpoints > 30:
        lloyd = True

    # Kernel_file
    kernel_file = join(
        kernel_dir, 'k_{:03d}_{:s}_{:d}D.npy'.format(num_kpoints, fixed,
                                                     dimension))

    # Check if already done
    if not exists(kernel_file):
        if lloyd:
            # Create kernels
            kernel_points = spherical_Lloyd(1.0,
                                            num_kpoints,
                                            dimension=dimension,
                                            fixed=fixed,
                                            verbose=0)

        else:
            # Create kernels
            kernel_points, grad_norms = kernel_point_optimization_debug(
                1.0,
                num_kpoints,
                num_kernels=100,
                dimension=dimension,
                fixed=fixed,
                verbose=0)

            # Find best candidate
            best_k = np.argmin(grad_norms[-1, :])

            # Save points
            kernel_points = kernel_points[best_k, :, :]

        np.save(kernel_file, kernel_points)

    else:
        kernel_points = np.load(kernel_file)

    # Random roations for the kernel
    # N.B. 4D random rotations not supported yet
    R = np.eye(dimension)
    theta = np.random.rand() * 2 * np.pi
    if dimension == 2:
        if fixed != 'vertical':
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)

    elif dimension == 3:
        if fixed != 'vertical':
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

        else:
            phi = (np.random.rand() - 0.5) * np.pi

            # Create the first vector in carthesian coordinates
            u = np.array([
                np.cos(theta) * np.cos(phi),
                np.sin(theta) * np.cos(phi),
                np.sin(phi)
            ])

            # Choose a random rotation angle
            alpha = np.random.rand() * 2 * np.pi

            # Create the rotation matrix with this vector and angle
            R = create_3D_rotations(np.reshape(u, (1, -1)),
                                    np.reshape(alpha, (1, -1)))[0]

            R = R.astype(np.float32)

    # Add a small noise
    kernel_points = kernel_points + np.random.normal(scale=0.01,
                                                     size=kernel_points.shape)

    # Scale kernels
    kernel_points = radius * kernel_points

    # Rotate kernels
    kernel_points = np.matmul(kernel_points, R)

    return kernel_points.astype(np.float32)


def batch_neighbors(queries, supports, q_batches, s_batches, radius):
    """Computes neighbors for a batch of queries and supports.

    Args:
        queries: (N1, 3) the query points
        supports: (N2, 3) the support points
        q_batches: (B) the list of lengths of batch elements in queries
        s_batches: (B)the list of lengths of batch elements in supports
        radius: float32

    Returns:
        neighbors indices

    """
    q_splits = torch.zeros((len(q_batches) + 1,), dtype=torch.int64)
    s_splits = torch.zeros((len(s_batches) + 1,), dtype=torch.int64)
    q_splits[1:] = torch.cumsum(torch.LongTensor(q_batches), dim=0)
    s_splits[1:] = torch.cumsum(torch.LongTensor(s_batches), dim=0)

    nns = FixedRadiusSearch()
    result = nns(torch.from_numpy(supports), torch.from_numpy(queries), radius,
                 s_splits, q_splits)

    idx = result.neighbors_index.reshape(-1, 1)
    splits = result.neighbors_row_splits

    max_nbrs = torch.max(splits[1:] - splits[:-1]).item()

    dense_idx = ragged_to_dense(
        idx, splits, max_nbrs,
        torch.Tensor([supports.shape[0]]).to(torch.int32)).squeeze(2)

    return dense_idx.numpy()


def batch_grid_subsampling(points,
                           batches_len,
                           features=None,
                           labels=None,
                           sampleDl=0.1,
                           max_p=0,
                           verbose=0,
                           random_grid_orient=True):
    """CPP wrapper for a grid subsampling (method = barycenter for points and features)

    Args:
        points: (N, 3) matrix of input points
        features: optional (N, d) matrix of features (floating number)
        labels: optional (N,) matrix of integer labels
        sampleDl: parameter defining the size of grid voxels
        verbose: 1 to display

    Returns:
        subsampled points, with features and/or labels depending of the input
    """
    R = None
    B = len(batches_len)
    if random_grid_orient:

        ########################################################
        # Create a random rotation matrix for each batch element
        ########################################################

        # Choose two random angles for the first vector in polar coordinates
        theta = np.random.rand(B) * 2 * np.pi
        phi = (np.random.rand(B) - 0.5) * np.pi

        # Create the first vector in carthesian coordinates
        u = np.vstack([
            np.cos(theta) * np.cos(phi),
            np.sin(theta) * np.cos(phi),
            np.sin(phi)
        ])

        # Choose a random rotation angle
        alpha = np.random.rand(B) * 2 * np.pi

        # Create the rotation matrix with this vector and angle
        R = create_3D_rotations(u.T, alpha).astype(np.float32)

        #################
        # Apply rotations
        #################

        i0 = 0
        points = points.copy()
        for bi, length in enumerate(batches_len):
            # Apply the rotation
            points[i0:i0 + length, :] = np.sum(
                np.expand_dims(points[i0:i0 + length, :], 2) * R[bi], axis=1)
            i0 += length

    #######################
    # Sunsample and realign
    #######################

    if (features is None) and (labels is None):
        s_points, s_len = subsample_batch(points,
                                          batches_len,
                                          sampleDl=sampleDl,
                                          max_p=max_p,
                                          verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                s_points[i0:i0 + length, :] = np.sum(
                    np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T,
                    axis=1)
                i0 += length
        return s_points, s_len

    elif (labels is None):
        s_points, s_len, s_features = subsample_batch(points,
                                                      batches_len,
                                                      features=features,
                                                      sampleDl=sampleDl,
                                                      max_p=max_p,
                                                      verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(
                    np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T,
                    axis=1)
                i0 += length
        return s_points, s_len, s_features

    elif (features is None):
        s_points, s_len, s_labels = subsample_batch(points,
                                                    batches_len,
                                                    classes=labels,
                                                    sampleDl=sampleDl,
                                                    max_p=max_p,
                                                    verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(
                    np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T,
                    axis=1)
                i0 += length
        return s_points, s_len, s_labels

    else:
        s_points, s_len, s_features, s_labels = subsample_batch(
            points,
            batches_len,
            features=features,
            classes=labels,
            sampleDl=sampleDl,
            max_p=max_p,
            verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(
                    np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T,
                    axis=1)
                i0 += length
        return s_points, s_len, s_features, s_labels


def p2p_fitting_regularizer(net):

    fitting_loss = 0
    repulsive_loss = 0

    for m in net.modules():

        if isinstance(m, KPConv) and m.deformable:

            ##############
            # Fitting loss
            ##############

            # Get the distance to closest input point and normalize to be independent from layers
            KP_min_d2 = m.min_d2 / (m.KP_extent**2)

            # Loss will be the square distance to closest input point. We use L1 because dist is already squared
            fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))

            ################
            # Repulsive loss
            ################

            # Normalized KP locations
            KP_locs = m.deformed_KP / m.KP_extent

            # Point should not be close to each other
            for i in range(net.K):
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]],
                                     dim=1).detach()
                distances = torch.sqrt(
                    torch.sum((other_KP - KP_locs[:, i:i + 1, :])**2, dim=2))
                rep_loss = torch.sum(torch.clamp_max(distances -
                                                     net.repulse_extent,
                                                     max=0.0)**2,
                                     dim=1)
                repulsive_loss += net.l1(rep_loss,
                                         torch.zeros_like(rep_loss)) / net.K

    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)


MODEL._register_module(KPFCNN, 'torch')
