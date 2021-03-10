""" Tools for computing various statistics on point cloud datasets
    Currently works only with ScanNet
"""

import functools
import numpy as np


def compute_scene_stats(mesh_vertices, semantic_labels, instance_labels,
                        instance_bboxes):
    """Compute scene statistics
    Args:
        mesh_vertices: (n_pts, 6) array with each row XYZRGB of a vertex
        semantic_labels: (n_pts, ) array with class label ids
        instance_labels: (n_pts, ) array with instance label ids
        instance_bboxes: (n_pts, 7) array with each row (xc, yc, zc, dx, dy,
        dz, class_label)
    Returns:
        scene_stats(dict):
    """
    stats = {}
    stats['points_per_scene'] = mesh_vertices.shape[0]
    stats['vertices'] = {
        'min': np.amin(mesh_vertices[:, :3], axis=0),
        'max': np.amax(mesh_vertices[:, :3], axis=0)
    }
    class_id, class_bbox_count = np.unique(instance_bboxes[:, -1],
                                           return_counts=True)
    stats['class_count'] = dict(
        zip(class_id.tolist(), class_bbox_count.tolist()))

    stats['bbox_shape'] = {
        class_id: [] for class_id in np.unique(instance_bboxes[:, -1])
    }
    for bbox in instance_bboxes:
        stats['bbox_shape'][bbox[-1]].append(bbox[3:6])

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
    Compute dataset statistics by aggregating data from all scenes
    Args:
        scene_stats(list(dict)):
    Returns:
        data_stats(dict):
    """
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
        for class_id in class_ids:
            class_bboxes = [
                bbox for stats in scene_stats
                for bbox in (stats['bbox_shape'][class_id] if class_id in
                             stats['bbox_shape'] else [])
            ]
            data_stats['bbox_shape'][class_id] = {
                'mean': np.mean(class_bboxes, axis=0).tolist(),
                'std': np.std(class_bboxes, axis=0).tolist()
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
                'std': float(np.std(class_n_pts_segs))
            }

    return data_stats
