from ...utils.rotate_iou import rotate_iou_gpu_eval, d3_box_overlap_kernel

def bev_iou(pred, target):
    riou = rotate_iou_gpu_eval(
        pred['bbox'][:, [0, 2, 3, 5, 6]],
        target['bbox'][:, [0, 2, 3, 5, 6]], -1)
    return riou

def box3d_iou(pred, target):
    riou = rotate_iou_gpu_eval(
        pred['bbox'][:, [0, 2, 3, 5, 6]],
        target['bbox'][:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(pred['bbox'], target['bbox'], riou, -1)
    return riou