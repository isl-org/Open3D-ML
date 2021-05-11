import numpy as np
import random
from ..utils.operations import create_3D_rotations


class SemsegAugmentation():
    """Class consisting different augmentation methods for Semantic Segmentation.

    Args:
        cfg: Config for augmentation.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def normalize(pc, feat, cfg):
        if 'points' in cfg:
            cfg_p = cfg['points']
            if cfg_p.get('recentering', False):
                pc -= pc.mean(0)
            if cfg_p.get('method', 'linear') == 'linear':
                pc -= pc.mean(0)
                pc /= (pc.max(0) - pc.min(0)).max()

        if 'feat' in cfg and feat is not None:
            cfg_f = cfg['feat']
            if cfg_f.get('recentering', False):
                feat -= feat.mean(0)
            if cfg_f.get('method', 'linear') == 'linear':
                bias = cfg_f.get('bias', 0)
                scale = cfg_f.get('scale', 1)
                feat -= bias
                feat /= scale

        return pc, feat

    @staticmethod
    def rotate(pc, cfg):
        # Initialize rotation matrix
        R = np.eye(pc.shape[1])

        method = cfg.get('method', 'vertical')

        if method == 'vertical':
            # Create random rotations
            theta = np.random.rand() * 2 * np.pi
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

        elif method == 'all':

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

        return np.matmul(pc, R)

    @staticmethod
    def scale(pc, cfg):

        # Choose random scales for each example
        scale_anisotropic = cfg.get('scale_anisotropic', False)
        min_s = cfg.get('min_s', 1.)
        max_s = cfg.get('max_s', 1.)

        if scale_anisotropic:
            scale = np.random.rand(pc.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) + min_s

        return pc * scale

    @staticmethod
    def noise(pc, cfg):
        noise_std = cfg.get('noise_std', 0.001)
        noise = (np.random.randn(pc.shape[0], pc.shape[1]) * noise_std).astype(
            np.float32)

        return pc + noise

    @staticmethod
    def RandomDropout(pc, feats, labels, cfg):
        dropout_ratio = cfg.get('dropout_ratio', 0.2)
        if random.random() < dropout_ratio:
            N = len(pc)
            inds = np.random.choice(N,
                                    int(N * (1 - dropout_ratio)),
                                    replace=False)
            return pc[inds], feats[inds], labels[inds]
        return pc, feats, labels

    @staticmethod
    def RandomHorizontalFlip(pc, cfg):
        axes = cfg.get('axes', [0, 1])
        if random.random() < 0.95:
            for curr_ax in axes:
                if random.random() < 0.5:
                    pc_max = np.max(pc[:, curr_ax])
                    pc[:, curr_ax] = pc_max - pc[:, curr_ax]

        return pc

    @staticmethod
    def ChromaticAutoContrast(feats, cfg):
        randomize_blend_factor = cfg.get('randomize_blend_factor', True)
        blend_factor = cfg.get('blend_factor', 0.5)
        if random.random() < 0.2:
            lo = feats[:, :3].min(0, keepdims=True)
            hi = feats[:, :3].max(0, keepdims=True)

            assert hi.max(
            ) > 1, "Invalid color value. Color is supposed to be in [0-255] for ChromaticAutoContrast augmentation"

            scale = 255 / (hi - lo)

            contrast_feats = (feats[:, :3] - lo) * scale

            blend_factor = random.random(
            ) if randomize_blend_factor else blend_factor
            feats[:, :3] = (
                1 - blend_factor) * feats[:, :3] + blend_factor * contrast_feats

        return feats

    @staticmethod
    def ChromaticTranslation(feats, cfg):
        trans_range_ratio = cfg.get('trans_range_ratio', 0.1)
        if random.random() < 0.95:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * trans_range_ratio
            feats[:, :3] = np.clip(tr + feats[:, :3], 0, 255)
        return feats

    @staticmethod
    def ChromaticJitter(feats, cfg):
        std = cfg.get('std', 0.01)
        if random.random() < 0.95:
            noise = np.random.randn(feats.shape[0], 3)
            noise *= std * 255
            feats[:, :3] = np.clip(noise + feats[:, :3], 0, 255)
        return feats

    def augment(self, point, feat, labels, cfg):
        if cfg is None:
            return point, feat, labels

        if 'normalize' in cfg:
            point, feat = self.normalize(point, feat, cfg['normalize'])

        if 'rotate' in cfg:
            point = self.rotate(point, cfg['rotate'])

        if 'scale' in cfg:
            point = self.scale(point, cfg['scale'])

        if 'noise' in cfg:
            point = self.noise(point, cfg['noise'])

        if 'RandomDropout' in cfg:
            point, feat, labels = self.RandomDropout(point, feat, labels,
                                                     cfg['RandomDropout'])

        if 'RandomHorizontalFlip' in cfg:
            point = self.RandomHorizontalFlip(point,
                                              cfg['RandomHorizontalFlip'])

        if 'ChromaticAutoContrast' in cfg:
            feat = self.ChromaticAutoContrast(feat,
                                              cfg['ChromaticAutoContrast'])

        if 'ChromaticTranslation' in cfg:
            feat = self.ChromaticTranslation(feat, cfg['ChromaticTranslation'])

        if 'ChromaticJitter' in cfg:
            feat = self.ChromaticJitter(feat, cfg['ChromaticJitter'])

        return point, feat, labels


class ObjdetAugmentation():
    """Class consisting different augmentation for Object Detection"""

    @staticmethod
    def PointShuffle(data):
        np.random.shuffle(data['point'])

        return data

    @staticmethod
    def ObjectRangeFilter(data, pcd_range):
        pcd_range = np.array(pcd_range)
        bev_range = pcd_range[[0, 1, 3, 4]]

        filtered_boxes = []
        for box in data['bbox_objs']:
            if in_range_bev(bev_range, box.to_xyzwhlr()):
                filtered_boxes.append(box)

        return {
            'point': data['point'],
            'bbox_objs': filtered_boxes,
            'calib': data['calib']
        }

    @staticmethod
    def ObjectSample(data, db_boxes_dict, sample_dict):
        rate = 1.0
        points = data['point']
        bboxes = data['bbox_objs']

        gt_labels_3d = [box.label_class for box in data['bbox_objs']]

        sampled_num_dict = {}

        for class_name in sample_dict.keys():
            max_sample_num = sample_dict[class_name]

            existing = np.sum([n == class_name for n in gt_labels_3d])
            sampled_num = int(max_sample_num - existing)
            sampled_num = np.round(rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num

        sampled = []
        for class_name in sampled_num_dict.keys():
            sampled_num = sampled_num_dict[class_name]
            if sampled_num < 0:
                continue

            sampled_cls = sample_class(class_name, sampled_num, bboxes,
                                       db_boxes_dict[class_name])
            sampled += sampled_cls
            bboxes = bboxes + sampled_cls

        if len(sampled) != 0:
            sampled_points = np.concatenate(
                [box.points_inside_box for box in sampled], axis=0)
            points = remove_points_in_boxes(points, sampled)
            points = np.concatenate([sampled_points, points], axis=0)

        return {'point': points, 'bbox_objs': bboxes, 'calib': data['calib']}

    @staticmethod
    def ObjectNoise(input,
                    trans_std=[0.25, 0.25, 0.25],
                    rot_range=[-0.15707963267, 0.15707963267],
                    num_try=100):
        raise NotImplementedError
