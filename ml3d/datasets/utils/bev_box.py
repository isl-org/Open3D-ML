from ...vis import BoundingBox3D
import numpy as np


class BEVBox3D(BoundingBox3D):
    """Class that defines a special bounding box for object detection, with only
    one rotation axis (yaw).

                            up z    x front (yaw=0.5*pi)
                                ^   ^
                                |  /
                                | /
        (yaw=pi) left y <------ 0

    The relative coordinate of bottom center in a BEV box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    The yaw is 0 at the negative direction of y axis, and increases from
    the negative direction of y to the positive direction of x.
    """

    def __init__(self,
                 center,
                 size,
                 yaw,
                 label_class,
                 confidence,
                 world_cam=None,
                 cam_img=None,
                 **kwargs):
        """Creates a bounding box.

        Args:
            center: (x, y, z) that defines the center of the box
            size: (width, height, depth) that defines the size of the box, as
                measured from edge to edge
            yaw: yaw angle of box
            label_class: integer specifying the classification label. If an LUT is
                specified in create_lines() this will be used to determine the color
                of the box.
            confidence: confidence level of the box
            world_cam: world to camera transformation
            cam_img: camera to image transformation
        """
        self.yaw = yaw
        self.world_cam = world_cam
        self.cam_img = cam_img

        # x-axis
        left = [np.cos(self.yaw), -np.sin(self.yaw), 0]
        # y-axis
        front = [np.sin(self.yaw), np.cos(self.yaw), 0]
        # z-axis
        up = [0, 0, 1]

        super().__init__(center, front, up, left, size, label_class, confidence,
                         **kwargs)

        self.points_inside_box = np.array([])
        self.level = self.get_difficulty()
        self.dis_to_cam = np.linalg.norm(self.to_camera()[:3])

    def to_kitti_format(self, score=1.0):
        """This method transforms the class to KITTI format."""
        box2d = self.to_img()
        box2d[2:] += box2d[:2]  # Add w, h.
        truncation = -1
        occlusion = -1
        box = self.to_camera()
        center = box[:3]
        size = box[3:6]
        ry = box[6]

        x, z = center[0], center[2]
        beta = np.arctan2(z, x)
        alpha = -np.sign(beta) * np.pi / 2 + beta + ry

        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.label_class, truncation, occlusion, alpha, box2d[0], box2d[1],
                       box2d[2], box2d[3], size[0], size[1], size[2], center[0], center[1], center[2],
                       ry, score)
        return kitti_str

    def generate_corners3d(self):
        """Generate corners3d representation for this object.

        Returns:
            corners_3d: (8, 3) corners of box3d in camera coordinates.
        """
        w, h, l = self.size
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.yaw), 0,
                       np.sin(self.yaw)], [0, 1, 0],
                      [-np.sin(self.yaw), 0,
                       np.cos(self.yaw)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.to_camera()[:3]
        return corners3d

    def to_xyzwhlr(self):
        """Returns box in the common 7-sized vector representation: (x, y, z, w,
        l, h, a), where (x, y, z) is the bottom center of the box, (w, l, h) is
        the width, length and height of the box a is the yaw angle.

        Returns:
            box(7,)

        """
        bbox = np.zeros((7,))
        bbox[0:3] = self.center - [0, 0, self.size[1] / 2]
        bbox[3:6] = np.array(self.size)[[0, 2, 1]]
        bbox[6] = self.yaw
        return bbox

    def to_camera(self):
        """Transforms box into camera space.

                     up x    y front
                        ^   ^
                        |  /
                        | /
         left z <------ 0

        Returns box in the common 7-sized vector representation.
        (x, y, z, l, h, w, a), where
        (x, y, z) is the bottom center of the box,
        (l, h, w) is the length, height, width of the box
        a is the yaw angle

        Returns:
            transformed box: (7,)
        """
        if self.world_cam is None:
            return self.to_xyzwhlr()[[1, 2, 0, 4, 5, 3, 6]]

        bbox = np.zeros((7,))
        # In camera space, we define center as center of the bottom face of bounding box.
        bbox[0:3] = self.center - [0, 0, self.size[1] / 2]
        # Transform center to camera frame of reference.
        bbox[0:3] = (np.array([*bbox[0:3], 1.0]) @ self.world_cam)[:3]
        bbox[3:6] = [self.size[1], self.size[0], self.size[2]]  # h, w, l
        bbox[6] = self.yaw
        return bbox

    def to_img(self):
        """Transforms box into 2d box.

        Returns:
            transformed box: (4,)
        """
        if self.cam_img is None:
            return None

        corners = self.generate_corners3d()
        corners = np.concatenate(
            [corners, np.ones((corners.shape[0], 1))], axis=-1)

        bbox_img = np.matmul(corners, self.cam_img)
        bbox_img = bbox_img[:, :2] / bbox_img[:, 3:]

        minxy = np.min(bbox_img, axis=0)
        maxxy = np.max(bbox_img, axis=0)

        size = maxxy - minxy
        center = minxy + size / 2

        return np.concatenate([center, size])

    def get_difficulty(self):
        """General method to compute difficulty, can be overloaded.

        Returns:
            Difficulty depending on projected height of box.
        """
        if self.cam_img is None:
            return 0

        heights = [40, 25]
        height = self.to_img()[3] + 1
        diff = -1
        for j in range(len(heights)):
            if height >= heights[j]:
                diff = j
                break
        return diff

    def to_dict(self):
        """Convert data for evaluation:"""
        return {
            'bbox': self.to_camera(),
            'label': self.label_class,
            'score': self.confidence,
            'difficulty': self.level
        }

    @staticmethod
    def to_dicts(bboxes):
        """Convert data for evaluation:

        Args:
            bboxes: List of BEVBox3D bboxes.
        """
        box_dicts = {
            'bbox': np.empty((len(bboxes), 7)),
            'label': np.empty((len(bboxes),), dtype='<U20'),
            'score': np.empty((len(bboxes),)),
            'difficulty': np.empty((len(bboxes),))
        }

        for i in range(len(bboxes)):
            box_dict = bboxes[i].to_dict()
            for k in box_dict:
                box_dicts[k][i] = box_dict[k]

        return box_dicts
