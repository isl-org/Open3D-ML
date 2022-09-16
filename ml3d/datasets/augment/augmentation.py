import numpy as np
import os
import pickle
import warnings
from ..utils.operations import *
from ..utils.transforms import in_range_bev


class Augmentation():
    """Class consisting common augmentation methods for different pipelines."""

    def __init__(self, cfg, seed=None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def recenter(self, data, cfg):
        """Recenter pointcloud/features to origin.

        Typically used before rotating the pointcloud.

        Args:
            data: Pointcloud or features.
            cfg: config dict where
                Key 'dim' specifies dimension to be recentered.

        """
        if not cfg:
            return data
        dim = cfg.get('dim', [0, 1, 2])
        data[:, dim] = data[:, dim] - data.mean(0)[dim]
        return data

    def normalize(self, pc, feat, cfg):
        """Normalize pointcloud and/or features.

        Points are normalized in [0, 1] and features can take custom
        scale and bias.

        Args:
            pc: Pointcloud.
            feat: features.
            cfg: configuration dictionary.

        """
        if 'points' in cfg:
            cfg_p = cfg['points']
            if cfg_p.get('method', 'linear') == 'linear':
                pc -= pc.mean(0)
                pc /= (pc.max(0) - pc.min(0)).max()
            else:
                raise ValueError(f"Unsupported method : {cfg_p.get('method')}")

        if 'feat' in cfg and feat is not None:
            cfg_f = cfg['feat']
            if cfg_f.get('method', 'linear') == 'linear':
                bias = cfg_f.get('bias', 0)
                scale = cfg_f.get('scale', 1)
                feat -= bias
                feat /= scale
            else:
                raise ValueError(f"Unsupported method : {cfg_f.get('method')}")

        return pc, feat

    def rotate(self, pc, cfg):
        """Rotate the pointcloud.

        Two methods are supported. `vertical` rotates the pointcloud
        along yaw. `all` randomly rotates the pointcloud in all directions.

        Args:
            pc: Pointcloud to augment.
            cfg: configuration dictionary.

        """
        # Not checking for height dimension as preserving absolute height dimension improves some models.
        if np.abs(pc[:, :2].mean()) > 1e-2:
            warnings.warn(
                f"It is recommended to recenter the pointcloud before calling rotate."
            )

        method = cfg.get('method', 'vertical')

        if method == 'vertical':
            # Create random rotations
            theta = self.rng.random() * 2 * np.pi
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

        elif method == 'all':

            # Choose two random angles for the first vector in polar coordinates
            theta = self.rng.random() * 2 * np.pi
            phi = (self.rng.random() - 0.5) * np.pi

            # Create the first vector in cartesian coordinates
            u = np.array([
                np.cos(theta) * np.cos(phi),
                np.sin(theta) * np.cos(phi),
                np.sin(phi)
            ])

            # Choose a random rotation angle
            alpha = self.rng.random() * 2 * np.pi

            # Create the rotation matrix with this vector and angle
            R = create_3D_rotations(np.reshape(u, (1, -1)),
                                    np.reshape(alpha, (1, -1)))[0]
        else:
            raise ValueError(f"Unsupported method : {method}")

        R = R.astype(np.float32)

        return np.matmul(pc, R)

    def scale(self, pc, cfg):
        """Scale augmentation for pointcloud.

        If `scale_anisotropic` is True, each point is scaled differently.
        else, same scale from range ['min_s', 'max_s') is applied to each point.

        Args:
            pc: Pointcloud to scale.
            cfg: configuration dict.

        """
        # Choose random scales for each example
        scale_anisotropic = cfg.get('scale_anisotropic', False)
        min_s = cfg.get('min_s', 1.)
        max_s = cfg.get('max_s', 1.)

        if scale_anisotropic:
            scale = self.rng.random(pc.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = self.rng.random() * (max_s - min_s) + min_s

        return pc * scale

    def noise(self, pc, cfg):
        noise_std = cfg.get('noise_std', 0.001)
        noise = (self.rng.standard_normal(
            (pc.shape[0], pc.shape[1])) * noise_std).astype(np.float32)

        return pc + noise

    def augment(self, data):
        raise NotImplementedError(
            "Please use one of SemsegAugmentation or ObjdetAugmentation.")


class SemsegAugmentation(Augmentation):
    """Class consisting of different augmentation methods for Semantic Segmentation.

    Args:
        cfg: Config for augmentation.
    """

    def __init__(self, cfg, seed=None):
        super(SemsegAugmentation, self).__init__(cfg, seed=seed)

        # Raise warnings for misspelled/unimplemented methods.
        all_methods = [
            'recenter', 'normalize', 'rotate', 'scale', 'noise',
            'RandomDropout', 'RandomHorizontalFlip', 'ChromaticAutoContrast',
            'ChromaticTranslation', 'ChromaticJitter',
            'HueSaturationTranslation'
        ]
        if cfg is None:
            return

        for method in cfg:
            if method not in all_methods:
                warnings.warn(
                    f"Augmentation method : {method} does not exist. Please verify!"
                )

    def RandomDropout(self, pc, feats, labels, cfg):
        """Randomly drops some points.

        Args:
            pc: Pointcloud.
            feats: Features.
            labels: Labels.
            cfg: configuration dict.
        """
        dropout_ratio = cfg.get('dropout_ratio', 0.2)
        if self.rng.random() < dropout_ratio:
            N = len(pc)
            inds = self.rng.choice(N,
                                   int(N * (1 - dropout_ratio)),
                                   replace=False)
            return pc[inds], feats[inds], labels[inds]
        return pc, feats, labels

    def RandomHorizontalFlip(self, pc, cfg):
        """Randomly flips the given axes.

        Args:
            pc: Pointcloud.
            cfg: configuraiton dict.

        """
        axes = cfg.get('axes', [0, 1])
        if self.rng.random() < 0.95:
            for curr_ax in axes:
                if self.rng.random() < 0.5:
                    pc_max = np.max(pc[:, curr_ax])
                    pc[:, curr_ax] = pc_max - pc[:, curr_ax]

        return pc

    def ChromaticAutoContrast(self, feats, cfg):
        """Improve contrast for RGB features.

        Args:
            feats: RGB features, should be in range [0-255].
            cfg: configuration dict.

        """
        randomize_blend_factor = cfg.get('randomize_blend_factor', True)
        blend_factor = cfg.get('blend_factor', 0.5)
        if self.rng.random() < 0.2:
            lo = feats[:, :3].min(0, keepdims=True)
            hi = feats[:, :3].max(0, keepdims=True)

            assert hi.max(
            ) > 1, "Invalid color value. Color is supposed to be in [0-255] for ChromaticAutoContrast augmentation"

            scale = 255 / (hi - lo)

            contrast_feats = (feats[:, :3] - lo) * scale

            blend_factor = self.rng.random(
            ) if randomize_blend_factor else blend_factor
            feats[:, :3] = (
                1 - blend_factor) * feats[:, :3] + blend_factor * contrast_feats

        return feats

    def ChromaticTranslation(self, feats, cfg):
        """Adds a small translation vector to features.

        Args:
            feats: Features.
            cfg: configuration dict.

        """
        trans_range_ratio = cfg.get('trans_range_ratio', 0.1)
        if self.rng.random() < 0.95:
            tr = (self.rng.random((1, 3)) - 0.5) * 255 * 2 * trans_range_ratio
            feats[:, :3] = np.clip(tr + feats[:, :3], 0, 255)
        return feats

    def ChromaticJitter(self, feats, cfg):
        """Adds a small noise jitter to features.

        Args:
            feats: Features.
            cfg: configuration dict.

        """
        std = cfg.get('std', 0.01)
        if self.rng.random() < 0.95:
            noise = self.rng.standard_normal((feats.shape[0], 3))
            noise *= std * 255
            feats[:, :3] = np.clip(noise + feats[:, :3], 0, 255)
        return feats

    @staticmethod
    def _rgb_to_hsv(rgb):
        """Converts RGB to HSV.

        Translated from source of colorsys.rgb_to_hsv
        r,g,b should be a numpy arrays with values between 0 and 255
        rgb_to_hsv returns an array of floats between 0.0 and 1.0.

        Args:
            rgb: RGB image

        Returns:
            HSV image

        """
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select([r == maxc, g == maxc],
                                [bc - gc, 2.0 + rc - bc],
                                default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0

        return hsv

    @staticmethod
    def _hsv_to_rgb(hsv):
        """Converts HSV to RGB.

        Translated from source of colorsys.hsv_to_rgb
        h,s should be a numpy arrays with values between 0.0 and 1.0
        v should be a numpy array with values between 0.0 and 255.0
        hsv_to_rgb returns an array of uints between 0 and 255.

        Args:
            hsv: HSV image

        Returns:
            RGB image

        """
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)

        return rgb.astype('uint8')

    @staticmethod
    def HueSaturationTranslation(feat, cfg):
        """Adds small noise to hue and saturation.

        Args:
            feat: Features.
            cfg: config dict with keys('hue_max', and 'saturation_max').

        """
        hue_max = cfg.get('hue_max', 0.5)
        saturation_max = cfg.get('saturation_max', 0.2)

        # Assume feat[:, :3] is rgb
        hsv = SemsegAugmentation._rgb_to_hsv(feat[:, :3])
        hue_val = (np.random.rand() - 0.5) * 2 * hue_max
        sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        feat[:, :3] = np.clip(SemsegAugmentation._hsv_to_rgb(hsv), 0, 255)

        return feat

    def augment(self, point, feat, labels, cfg, seed=None):
        if cfg is None:
            return point, feat, labels

        # Override RNG for reproducibility with parallel dataloader.
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if 'recenter' in cfg:
            point = self.recenter(point, cfg['recenter'])

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

        if 'HueSaturationTranslation' in cfg:
            feat = self.HueSaturationTranslation(
                feat, cfg['HueSaturationTranslation'])

        return point, feat, labels


class ObjdetAugmentation(Augmentation):
    """Class consisting different augmentation for Object Detection"""

    def __init__(self, cfg, seed=None):
        super(ObjdetAugmentation, self).__init__(cfg, seed=seed)

        # Raise warnings for misspelled/unimplemented methods.
        all_methods = [
            'recenter', 'normalize', 'rotate', 'scale', 'noise', 'PointShuffle',
            'ObjectRangeFilter', 'ObjectSample'
        ]
        for method in cfg:
            if method not in all_methods:
                warnings.warn(
                    f"Augmentation method : {method} does not exist. Please verify!"
                )

    def PointShuffle(self, data):
        """Shuffle Pointcloud."""
        self.rng.shuffle(data['point'])

        return data

    @staticmethod
    def in_range_bev(box_range, box):
        return (box[0] > box_range[0]) & (box[1] > box_range[1]) & (
            box[0] < box_range[2]) & (box[1] < box_range[3])

    def ObjectRangeFilter(self, data, pcd_range):
        """Filter Objects in the given range."""
        pcd_range = np.array(pcd_range)
        bev_range = pcd_range[[0, 1, 3, 4]]

        filtered_boxes = []
        for box in data['bounding_boxes']:
            if self.in_range_bev(bev_range, box.to_xyzwhlr()):
                filtered_boxes.append(box)

        return {
            'point': data['point'],
            'bounding_boxes': filtered_boxes,
            'calib': data['calib']
        }

    def ObjectSample(self, data, db_boxes_dict, sample_dict):
        """Increase frequency of objects in a pointcloud.

        Randomly place objects in a pointcloud from a database of
        all objects within the dataset. Checks collision with existing objects.

        Args:
            data: Input data dict with keys ('point', 'bounding_boxes', 'calib').
            db_boxes_dict: dict for different objects.
            sample_dict: dict for number of objects to sample.

        """
        rate = 1.0
        points = data['point']
        bboxes = data['bounding_boxes']

        gt_labels_3d = [box.label_class for box in data['bounding_boxes']]

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
            points = np.concatenate([sampled_points[:, :4], points], axis=0)

        return {
            'point': points,
            'bounding_boxes': bboxes,
            'calib': data['calib']
        }

    def load_gt_database(self, pickle_path, min_points_dict, sample_dict):
        """Load ground truth object database.

        Args:
            pickle_path: Path of pickle file generated using `scripts/collect_bbox.py`.
            min_points_dict: A dictionary to filter objects based on number of points inside.
                Format of dict {'class_name': num_points}.
            sample_dict: A dictionary to decide number of objects to sample.
                Format of dict {'class_name': num_instance}

        """
        db_boxes = pickle.load(open(pickle_path, 'rb'))

        if min_points_dict is not None:
            db_boxes = filter_by_min_points(db_boxes, min_points_dict)

        db_boxes_dict = {}
        for key in sample_dict.keys():
            db_boxes_dict[key] = []

        for db_box in db_boxes:
            if db_box.label_class in sample_dict.keys():
                db_boxes_dict[db_box.label_class].append(db_box)

        self.db_boxes_dict = db_boxes_dict

    def augment(self, data, attr, seed=None):
        """Augment object detection data.

        Available augmentations are:
            `ObjectSample`: Insert objects from ground truth database.
            `ObjectRangeFilter`: Filter pointcloud from given bounds.
            `PointShuffle`: Shuffle the pointcloud.

        Args:
            data: A dictionary object returned from the dataset class.
            attr: Attributes for current pointcloud.

        Returns:
            Augmented `data` dictionary.

        """
        cfg = self.cfg

        if cfg is None:
            return data

        # Override RNG for reproducibility with parallel dataloader.
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if 'recenter' in cfg:
            if cfg['recenter']:
                data['point'] = self.recenter(data['point'], cfg['recenter'])

        if 'normalize' in cfg:
            data['point'], _ = self.normalize(data['point'], None,
                                              cfg['normalize'])

        if 'rotate' in cfg:
            data['point'] = self.rotate(data['point'], cfg['rotate'])

        if 'scale' in cfg:
            data['point'] = self.scale(data['point'], cfg['scale'])

        if 'noise' in cfg:
            data['point'] = self.noise(data['point'], cfg['noise'])

        if 'ObjectSample' in cfg:
            if not hasattr(self, 'db_boxes_dict'):
                data_path = attr['path']
                # remove tail of path to get root data path
                for _ in range(3):
                    data_path = os.path.split(data_path)[0]
                pickle_path = os.path.join(data_path, 'bboxes.pkl')
                if 'pickle_path' not in cfg['ObjectSample']:
                    cfg['ObjectSample']['pickle_path'] = pickle_path
                self.load_gt_database(**cfg['ObjectSample'])

            data = self.ObjectSample(
                data,
                db_boxes_dict=self.db_boxes_dict,
                sample_dict=cfg['ObjectSample']['sample_dict'])

        if cfg.get('ObjectRangeFilter', False):
            data = self.ObjectRangeFilter(
                data, cfg['ObjectRangeFilter']['point_cloud_range'])

        if cfg.get('PointShuffle', False):
            data = self.PointShuffle(data)

        return data
