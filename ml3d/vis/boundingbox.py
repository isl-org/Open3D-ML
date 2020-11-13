import numpy as np
import open3d as o3d


class BoundingBox3D:
    next_id = 1

    def __init__(self,
                 center,
                 front,
                 up,
                 left,
                 size,
                 label_class,
                 confidence,
                 meta=None,
                 show_class=False,
                 show_confidence=False,
                 show_meta=None,
                 identifier=None,
                 arrow_length=1.0):
        assert (len(center) == 3)
        assert (len(front) == 3)
        assert (len(up) == 3)
        assert (len(left) == 3)
        assert (len(size) == 3)

        self.center = np.array(center, dtype="float32")
        self.front = np.array(front, dtype="float32")
        self.up = np.array(up, dtype="float32")
        self.left = np.array(left, dtype="float32")
        self.size = size
        self.label_class = label_class
        self.confidence = confidence
        self.meta = meta
        self.show_class = show_class
        self.show_confidence = show_confidence
        self.show_meta = show_meta
        if identifier is not None:
            self.identifier = identifier
        else:
            self.identifier = "box:" + str(BoundingBox3D.next_id)
            BoundingBox3D.next_id += 1
        self.arrow_length = arrow_length

    def __repr__(self):
        s = str(self.identifier) + " (class=" + str(
            self.label_class) + ", conf=" + str(self.confidence)
        if self.meta is not None:
            s = s + ", meta=" + str(self.meta)
        s = s + ")"
        return s

    @staticmethod
    def create_lines(boxes, lut):
        nverts = 14
        nlines = 17
        points = np.zeros((nverts * len(boxes), 3), dtype="float32")
        indices = np.zeros((nlines * len(boxes), 2), dtype="int32")
        colors = np.zeros((nlines * len(boxes), 3), dtype="float32")

        for i in range(0, len(boxes)):
            box = boxes[i]
            pidx = nverts * i
            x = box.size[0] * box.left
            y = box.size[1] * box.up
            z = box.size[2] * box.front
            arrow_tip = box.center + z + box.arrow_length * box.front
            arrow_mid = box.center + z + 0.60 * box.arrow_length * box.front
            head_length = 0.3 * box.arrow_length
            # It seems to be substantially faster to assign directly for the
            # points, as opposed to points[pidx:pidx+nverts] = np.stack((...))
            points[pidx] = box.center + x + y + z
            points[pidx + 1] = box.center - x + y + z
            points[pidx + 2] = box.center - x + y - z
            points[pidx + 3] = box.center + x + y - z
            points[pidx + 4] = box.center + x - y + z
            points[pidx + 5] = box.center - x - y + z
            points[pidx + 6] = box.center - x - y - z
            points[pidx + 7] = box.center + x - y - z
            points[pidx + 8] = box.center + z
            points[pidx + 9] = arrow_tip
            points[pidx + 10] = arrow_mid + head_length * box.up
            points[pidx + 11] = arrow_mid - head_length * box.up
            points[pidx + 12] = arrow_mid + head_length * box.left
            points[pidx + 13] = arrow_mid - head_length * box.left

        # It is faster to break the indices and colors into their own loop.
        for i in range(0, len(boxes)):
            box = boxes[i]
            pidx = nverts * i
            idx = nlines * i
            indices[idx:idx +
                    nlines] = ((pidx, pidx + 1), (pidx + 1, pidx + 2),
                               (pidx + 2, pidx + 3), (pidx + 3, pidx),
                               (pidx + 4, pidx + 5), (pidx + 5, pidx + 6),
                               (pidx + 6, pidx + 7), (pidx + 7, pidx + 4),
                               (pidx + 0, pidx + 4), (pidx + 1, pidx + 5),
                               (pidx + 2, pidx + 6), (pidx + 3, pidx + 7),
                               (pidx + 8, pidx + 9), (pidx + 9, pidx + 10),
                               (pidx + 9, pidx + 11), (pidx + 9,
                                                       pidx + 12), (pidx + 9,
                                                                    pidx + 13))

            if lut is not None:
                label = lut.labels[box.label_class]
                c = (label.color[0], label.color[1], label.color[2])
            else:
                c = (0.5, 0.5, 0.5)

            colors[idx:idx +
                   nlines] = c  # copies c to each element in the range

        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(points)
        lines.lines = o3d.utility.Vector2iVector(indices)
        lines.colors = o3d.utility.Vector3dVector(colors)

        return lines
