""" Tools for computing various statistics on point cloud datasets
    Currently works only with ScanNet
"""

import functools
import logging as log
import numpy as np
from sklearn.cluster import KMeans

NBINS = 10


def compute_scene_stats(mesh_vertices, semantic_labels, instance_labels,
                        instance_bboxes):
    """Compute scene statistics

    Args:
        mesh_vertices: (n_pts, 6) array with each row XYZRGB of a vertex
        semantic_labels: (n_pts, ) array with class label ids
        instance_labels: (n_pts, ) array with instance label ids
        instance_bboxes: (n_pts, 7-10) array with each row
            scenes: [xc, yc, zc, dx, dy, dz, class_id] or
            frames: [xc, yc, zc, dx, dy, dz, yaw, class_id, truncation, n_pts_inside_bbox]

    Returns:
        scene_stats(dict):
    """
    stats = {}
    if mesh_vertices is not None:
        stats['points_per_scene'] = mesh_vertices.shape[0]
        stats['vertices'] = {
            'min': np.amin(mesh_vertices[:, :3], axis=0),
            'max': np.amax(mesh_vertices[:, :3], axis=0)
        }
    if instance_bboxes is not None:
        cls_idx = 6 if instance_bboxes.shape[1] == 7 else 7
        class_id, class_bbox_count = np.unique(instance_bboxes[:, cls_idx],
                                               return_counts=True)
        stats['class_count'] = dict(
            zip(class_id.tolist(), class_bbox_count.tolist()))

        stats['bbox_shape'] = {clsid: [] for clsid in class_id}
        for bbox in instance_bboxes:
            stats['bbox_shape'][bbox[cls_idx]].append(bbox[3:6])
        if instance_bboxes.shape[1] > 8:
            stats['truncation'], _ = np.histogram(instance_bboxes[:, 8],
                                                  bins=NBINS,
                                                  range=(0, 1))
        if instance_bboxes.shape[1] > 9:
            stats['n_pts_bbox'] = instance_bboxes[:, 9]

    if instance_labels is not None and semantic_labels is not None:
        instance_id, n_pts_seg = np.unique(instance_labels, return_counts=True)
        instance_to_class = dict(zip(instance_labels, semantic_labels))
        stats['n_pts_seg'] = {
            class_id: [] for class_id in instance_to_class.values()
        }
        stats['n_pts_seg'][0] = []  # Unlabeled
        for instance, n_pts in zip(instance_id, n_pts_seg):
            stats['n_pts_seg'][instance_to_class[instance]].append(n_pts)

    return stats


def compute_dataset_stats(scene_stats):
    """
    Compute dataset statistics by aggregating data from all scenes.

    Args:
        scene_stats(list(dict)):

    Returns:
        data_stats(dict):
    """
    scene_stats = list(filter(None, scene_stats))  # Remove empty scenes
    data_stats = {}
    if len(scene_stats) == 0:
        return data_stats

    if 'points_per_scene' in scene_stats[0]:
        points_per_scene = [stats['points_per_scene'] for stats in scene_stats]
        data_stats['points_per_scene'] = {
            'min': int(np.min(points_per_scene)),
            'max': int(np.max(points_per_scene)),
            'mean': float(np.mean(points_per_scene)),
            'std': float(np.std(points_per_scene)),
        }
    if 'vertices' in scene_stats[0]:

        scene_size = [
            stats['vertices']['max'] - stats['vertices']['min']
            for stats in scene_stats
        ]
        data_stats['point_cloud'] = {
            'points_min':
                functools.reduce(np.fmin, (stats['vertices']['min']
                                           for stats in scene_stats)).tolist(),
            'points_max':
                functools.reduce(np.fmax, (stats['vertices']['max']
                                           for stats in scene_stats)).tolist(),
            'size_min':
                functools.reduce(np.fmin, scene_size).tolist(),
            'size_max':
                functools.reduce(np.fmax, scene_size).tolist(),
            'size_mean':
                np.mean(scene_size, axis=0).tolist(),
            'size_std':
                np.std(scene_size, axis=0).tolist(),
        }
    if 'class_count' in scene_stats[0]:

        def accumulate_dict(accumulator, new_dict):
            for key, value in new_dict.items():
                accumulator[key] = value + accumulator.get(key, 0)
            return accumulator

        data_stats['class_count'] = functools.reduce(
            accumulate_dict, (stats['class_count'] for stats in scene_stats))

    if 'bbox_shape' in scene_stats[0]:
        class_ids = np.unique([
            clsid for stats in scene_stats
            for clsid in stats['bbox_shape'].keys()
        ]).tolist()

        data_stats['bbox_shape'] = dict(zip(class_ids, [[]] * len(class_ids)))
        class_bboxes = dict(zip(class_ids, [[]] * len(class_ids)))
        for class_id in class_ids:
            class_bboxes[class_id] = np.array([
                bbox for stats in scene_stats
                for bbox in (stats['bbox_shape'][class_id] if class_id in
                             stats['bbox_shape'] else [])
            ])
            data_stats['bbox_shape'][class_id] = {
                'mean': np.mean(class_bboxes[class_id], axis=0).tolist(),
                'std': np.std(class_bboxes[class_id], axis=0).tolist()
            }
        anchor_boxes, mIoU = calc_anchor_boxes(
            class_bboxes, range(len(class_ids) // 2, 2 * len(class_ids) + 1))
        data_stats['anchor_boxes'] = {'mIou': mIoU, 'boxes': anchor_boxes}

    if 'truncation' in scene_stats[0]:
        truncation = np.sum([stats['truncation'] for stats in scene_stats],
                            axis=0).tolist()
        data_stats['truncation'] = {
            (i + 1.0) / NBINS: t for i, t in enumerate(truncation)
        }
    if 'n_pts_bbox' in scene_stats[0]:
        n_pts_bbox = np.concatenate(
            [stats['n_pts_bbox'] for stats in scene_stats])
        hist = np.histogram(n_pts_bbox, bins=NBINS)
        data_stats['n_pts_bbox'] = {
            int(hist[1][k + 1]): int(count) for k, count in enumerate(hist[0])
        }

    if 'n_pts_seg' in scene_stats[0]:
        data_stats['n_pts_seg'] = dict(zip(class_ids, [[]] * len(class_ids)))
        for class_id in class_ids:
            class_n_pts_segs = [
                n_pts for stats in scene_stats
                for n_pts in (stats['n_pts_seg'][class_id] if class_id in
                              stats['n_pts_seg'] else [])
            ]
            data_stats['n_pts_seg'][class_id] = {
                'mean': float(np.mean(class_n_pts_segs)),
                'std': float(np.std(class_n_pts_segs)),
                'min': int(np.min(class_n_pts_segs)),
                'max': int(np.max(class_n_pts_segs))
            }

    return data_stats


def calc_anchor_boxes(bbox_shapes, n_anchors):
    """
    Use k-means clustering on bounding boxes to estimate anchor boxes using
    Euclidean distance on the boundin box size.

    Args:
        bbox_shapes (dict(class: list(bbox shape))): Bounding box shapes for
            all instances of all classes
        n_anchors (list): Number of anchor boxes to estimate.

    Returns:
        anchor_boxes (dict): The estimated anchor boxes for each n_anchor.
        mIoU (dict): The mean IoU metric for each n_anchor.
    """

    all_bboxes = [
        bbox for class_bboxes in bbox_shapes.values() for bbox in class_bboxes
    ]
    all_bbox_cls = [
        cls for cls, class_bboxes in bbox_shapes.items()
        for bbox in class_bboxes
    ]
    mIoU = {}
    anchor_boxes = {}
    for n_clusters in n_anchors:
        kmeans = KMeans(n_clusters=n_clusters).fit(all_bboxes)
        anchor_boxes[n_clusters] = kmeans.cluster_centers_.tolist()
        assigned_anchor_box = kmeans.labels_
        all_IoU = [
            centered_bbox3d_IoU(bbox, anchor_boxes[n_clusters][k])
            for bbox, k in zip(all_bboxes, assigned_anchor_box)
        ]
        mIoU[n_clusters] = float(np.mean(all_IoU))
        class_mIoU = {
            cls:
            np.ma.mean(np.ma.array(all_IoU, mask=np.array(all_bbox_cls) == cls))
            for cls in bbox_shapes.keys()
        }
        anchor_boxes[n_clusters].sort(key=lambda box: box[0] * box[1] * box[2])
        log.debug(f"{n_clusters} clusters:\n{anchor_boxes[n_clusters]}")
        log.info(f"mIoU = {mIoU[n_clusters]}\nclass mIoU = {class_mIoU}")
    return anchor_boxes, mIoU


def centered_bbox3d_IoU(bbox_shape1, bbox_shape2):
    """
    Compute Intersection over Union for a pair of axis aligned 3D bounding boxes
    centered at the origin.

    Args:
        bbox_shape2, bbox_shape1: [length, width, height] of each bounding box

    Returns:
        Intersection Volume / Union Volume
    """
    intersection_volume = np.prod(np.fmin(bbox_shape1, bbox_shape2))
    union_volume = (np.prod(bbox_shape1) + np.prod(bbox_shape2) -
                    intersection_volume)
    return intersection_volume / union_volume
