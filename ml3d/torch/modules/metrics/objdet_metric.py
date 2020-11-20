import gc
import io as sysio
import numpy as np



#@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist']
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno['name'])
    num_dt = len(dt_anno['name'])
    num_valid_gt = 0
    for i in range(num_gt):
        bbox = gt_anno['bbox'][i]
        gt_name = gt_anno['name'][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        elif (current_cls_name == 'Pedestrian'.lower()
              and 'Person_sitting'.lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == 'Car'.lower() and 'Van'.lower() == gt_name):
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        if ((gt_anno['occluded'][i] > MAX_OCCLUSION[difficulty])
                or (gt_anno['truncated'][i] > MAX_TRUNCATION[difficulty])
                or (height <= MIN_HEIGHT[difficulty])):
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
    # for i in range(num_gt):
        if gt_anno['name'][i] == 'DontCare':
            dc_bboxes.append(gt_anno['bbox'][i])
    for i in range(num_dt):
        if (dt_anno['name'][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno['bbox'][i, 3] - dt_anno['bbox'][i, 1])
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes



#@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lidar.
    # TODO: change to use prange for parallel mode, should check the difference
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in numba.prange(N):
        for j in numba.prange(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = (
                    min(boxes[i, 1], qboxes[j, 1]) -
                    max(boxes[i, 1] - boxes[i, 4],
                        qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
    rinc = nms_gpu(boxes[:, [0, 2, 3, 5, 6]], qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


#@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                # delta.append(gt_alphas[i] - dt_alphas[det_idx])
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


#@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i],
                               gt_num:gt_num + gt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def calculate_iou_partly(gt_annos, dt_annos, num_parts=50):
    """Fast iou algorithm. this function can be used independently to do result
    analysis. Must be used in CAMERA coordinate system.

    Args:
        gt_annos (dict): Must from get_label_annos() in kitti_common.py.
        dt_annos (dict): Must from get_label_annos() in kitti_common.py.
        num_parts (int): A parameter for fast calculate algorithm.
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a['name']) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a['name']) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]

        loc = np.concatenate([a['location'] for a in gt_annos_part], 0)
        dims = np.concatenate([a['dimensions'] for a in gt_annos_part], 0)
        rots = np.concatenate([a['rotation_y'] for a in gt_annos_part], 0)
        gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                    axis=1)
        loc = np.concatenate([a['location'] for a in dt_annos_part], 0)
        dims = np.concatenate([a['dimensions'] for a in dt_annos_part], 0)
        rots = np.concatenate([a['rotation_y'] for a in dt_annos_part], 0)
        dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                    axis=1)
        overlap_part = d3_box_overlap(gt_boxes,
                                        dt_boxes).astype(np.float64)

        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate(
            [gt_annos[i]['bbox'], gt_annos[i]['alpha'][..., np.newaxis]], 1)
        dt_datas = np.concatenate([
            dt_annos[i]['bbox'], dt_annos[i]['alpha'][..., np.newaxis],
            dt_annos[i]['score'][..., np.newaxis]
        ], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)


def eval_class(gt_annos,
               dt_annos,
               current_classes,
               difficultys,
               min_overlaps,
               num_parts=200):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.

    Args:
        gt_annos (dict): Must from get_label_annos() in kitti_common.py.
        dt_annos (dict): Must from get_label_annos() in kitti_common.py.
        current_classes (list[int]): 0: car, 1: pedestrian, 2: cyclist.
        difficultys (list[int]): Eval difficulty, 0: easy, 1: normal, 2: hard
        min_overlaps (float): Min overlap. format:
            [num_overlap, class].
        num_parts (int): A parameter for fast calculate algorithm

    Returns:
        dict[str, np.ndarray]: recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    if num_examples < num_parts:
        num_parts = num_examples
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    precision = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    for m, current_class in enumerate(current_classes):
        for idx_l, difficulty in enumerate(difficultys):
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets
            for k, min_overlap in enumerate(min_overlaps[:, m]):
                thresholdss = []
                for i in range(len(gt_annos)):
                    rets = compute_statistics_jit(
                        overlaps[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        dontcares[i],
                        min_overlap=min_overlap,
                        thresh=0.0,
                        compute_fp=False)
                    thresholds = rets[-1]
                    thresholdss += thresholds.tolist()
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx:idx + num_part], 0)
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx:idx + num_part], 0)
                    dc_datas_part = np.concatenate(
                        dontcares[idx:idx + num_part], 0)
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx:idx + num_part], 0)
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx:idx + num_part], 0)
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        total_dc_num[idx:idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        min_overlap=min_overlap,
                        thresholds=thresholds)
                    idx += num_part
                for i in range(len(thresholds)):
                    recall[m, idx_l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, idx_l, k, i] = pr[i, 0] / (
                        pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[m, idx_l, k, i] = np.max(
                        precision[m, idx_l, k, i:], axis=-1)
                    recall[m, idx_l, k, i] = np.max(
                        recall[m, idx_l, k, i:], axis=-1)
    ret_dict = {
        'recall': recall,
        'precision': precision,
        'orientation': aos,
    }

    # clean temp variables
    del overlaps
    del parted_overlaps

    gc.collect()
    return ret_dict


def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps):
    # min_overlaps: [num_minoverlap, num_class]
    difficulties = [0, 1, 2]
    ret = eval_class(gt_annos, dt_annos, current_classes, difficulties,
                        min_overlaps)
    mAP_3d = get_mAP(ret['precision'])
    return mAP_3d


def kitti_eval(gt_annos,
               dt_annos,
               current_classes):
    """KITTI evaluation.

    Args:
        gt_annos (list[dict]): Contain gt information of each sample.
        dt_annos (list[dict]): Contain detected information of each sample.
        current_classes (list[str]): Classes to evaluation.

    Returns:
        tuple: String and dict of evaluation results.
    """

    overlap_0_7 = np.array([0.7, 0.5, 0.5, 0.7, 0.5])
    overlap_0_5 = np.array([0.5, 0.25, 0.25, 0.5, 0.25])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 5]
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, current_classes]
    result = ''

    mAP3d = do_eval(gt_annos, dt_annos, current_classes, min_overlaps)

    ret_dict = {}
    difficulty = ['easy', 'moderate', 'hard']
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        curcls_name = class_to_name[curcls]
        for i in range(min_overlaps.shape[0]):
            # prepare results for print
            result += ('{} AP@{:.2f}, {:.2f}, {:.2f}:\n'.format(
                curcls_name, *min_overlaps[i, j]))
            result += '3d   AP:{:.4f}, {:.4f}, {:.4f}\n'.format(
                *mAP3d[j, :, i])

            # prepare results for logger
            for idx in range(3):
                if i == 0:
                    postfix = f'{difficulty[idx]}_strict'
                else:
                    postfix = f'{difficulty[idx]}_loose'
                prefix = f'KITTI/{curcls_name}'
                ret_dict[f'{prefix}_3D_{postfix}'] = mAP3d[j, idx, i]

    # calculate mAP over all classes if there are multiple classes
    if len(current_classes) > 1:
        # prepare results for print
        result += ('\nOverall AP@{}, {}, {}:\n'.format(*difficulty))
        mAP3d = mAP3d.mean(axis=0)
        result += '3d   AP:{:.4f}, {:.4f}, {:.4f}\n'.format(*mAP3d[:, 0])

        # prepare results for logger
        for idx in range(3):
            postfix = f'{difficulty[idx]}'
            ret_dict[f'KITTI/Overall_3D_{postfix}'] = mAP3d[idx, 0]

    return result, ret_dict

