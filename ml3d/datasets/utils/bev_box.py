from ...vis import BoundingBox3D
import numpy as np


class BEVBox3D(BoundingBox3D):
    """Class that defines a special bounding box for object detection, with only one rotation axis (yaw)."""

    def __init__(self,
                 center,
                 size,
                 yaw,
                 label_class,
                 confidence,
                 world_cam=None,
                 cam_img=None,
                 **kwargs):
        """Creates a bounding box. Front, up, left define the axis of the box
        and must be normalized and mutually orthogonal.

        center: (x, y, z) that defines the center of the box
        yaw: yaw angle of box
        size: (width, height, depth) that defines the size of the box, as
            measured from edge to edge
        label_class: integer specifying the classification label. If an LUT is
            specified in create_lines() this will be used to determine the color
            of the box.
        confidence: confidence level of the box
        world_cam: world to camera transformation
        cam_img: camera to image transformation"""

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
        self.dis_to_cam = np.linalg.norm(self.center)

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.size[2::-1]
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
        """
        Returns box in the common 7-sized vector representation.
        :return box: (7,)
        """
        bbox = np.zeros((7,))
        bbox[0:3] = self.center - [0, 0, self.size[1] / 2]
        bbox[3:6] = np.array(self.size)[[0, 2, 1]]
        bbox[6] = self.yaw
        return bbox

    def to_camera(self):
        """
        Transforms box into camera space.
        :return transformed box: (7,)
        """
        if self.world_cam is None:
            return self.to_xyzwhlr()[[1, 2, 0, 4, 5, 3, 6]]

        bbox = np.zeros((7,))
        bbox[0:3] = self.center - [0, 0, self.size[1] / 2]
        bbox[0:3] = (np.array([*bbox[0:3], 1.0]) @ self.world_cam)[:3]
        bbox[3:6] = self.size[2::-1]
        bbox[6] = self.yaw
        return bbox

    def to_img(self):
        """
        Transforms box into 2d box.
        :return transformed box: (4,)
        """
        if self.cam_img is None:
            return None

        corners = self.generate_corners3d()
        corners = np.concatenate(
            [corners, np.ones((corners.shape[0], 1))], axis=-1)

        bbox_img = np.matmul(corners, self.cam_img)
        bbox_img = bbox_img[:, :2] / bbox_img[:, 2:3]

        minxy = np.min(bbox_img, axis=0)
        maxxy = np.max(bbox_img, axis=0)

        size = maxxy - minxy
        center = minxy + size / 2

        return np.concatenate([center, size])

    def get_difficulty(self):
        """
        General method to compute difficulty, can be overloaded.
        Return difficulty depending on projected height of box.
        """

        if self.cam_img is None:
            return 0

        heights = [40, 25]
        height = self.to_img()[3]
        diff = -1
        for j in range(len(heights)):
            if height > heights[j]:
                diff = j
                break
        return diff

    def to_dict(self):
        """
        Convert data for evaluation:
        """
        return {
            'bbox': self.to_camera(),
            'label': self.label_class,
            'score': self.confidence,
            'difficulty': self.level
        }

    @staticmethod
    def to_dicts(bboxes):
        """
        Convert data for evaluation:

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
