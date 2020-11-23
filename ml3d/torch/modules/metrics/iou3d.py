from ...utils.rotate_iou import rotate_iou_gpu_eval, d3_box_overlap_kernel

def bev_iou(pred, target):
    riou = rotate_iou_gpu_eval(
        pred['bboxes'][:, [0, 2, 3, 5, 6]],
        target['bboxes'][:, [0, 2, 3, 5, 6]], -1)
    return riou

def box3d_iou(pred, target):
    riou = rotate_iou_gpu_eval(
        pred['bboxes'][:, [0, 2, 3, 5, 6]],
        target['bboxes'][:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(pred['bboxes'], target['bboxes'], riou, -1)
    return riou