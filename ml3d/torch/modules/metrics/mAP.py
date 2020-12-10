import numpy as np
from . import box3d_iou, bev_iou

def filter_data(data, labels, diffs=None):
    """Filters the data to fit the given labels and difficulties.
    Args:
        data (dict): Dictionary with the data (as numpy arrays).
            {
                'label':      [...], # expected
                'difficulty': [...]  # if diffs not None
                ...
            }
        labels (number[]): List of labels which should be maintained.
        difficulties (number[]): List of difficulties which should maintained.
            (optional)

    Returns:
        Dictionary with same as format as input, with only the given labels
        and difficulties.
    """
    cond = np.any([data['label'] == label for label in labels], axis=0)
    if diffs is not None and 'difficulty' in data:
        dcond = np.any([np.all([data['difficulty'] >= 0, data['difficulty'] <= diff], axis=0) for diff in diffs], axis=0)
        cond = np.all([cond, dcond], axis=0)
    idx = np.where(cond)[0]

    result = {}
    for k in data:
        result[k] = data[k][idx]
    return result, idx

def flatten_data(data):
    """Converts a list of dictionaries into one dictionary.
    Args:
        data (dict): List of dictionaries with the data (as numpy arrays).
            {
                ...: [...]
            }[]

    Returns:
        Single dictionary with merged lists and additional entry 
        for the original indices.
            {
                ...: [...],
                idx: number[]
            }[]
    """
    res = {}
    res['idx'] = []
    for i, d in enumerate(data):
        l = 0
        for k in d:
            if k not in res:
                res[k] = []
            res[k].extend(d[k])
            l = len(d[k])
        res['idx'].extend([i]*l)
    for k in res:
        res[k] = np.array(res[k])
    return res


def precision_3d(pred, target, classes=[0], difficulties=[0], min_overlap=[0.5], bev=True, similar_classes={}):
    """Computes precision quantities for each predicted box.
    Args:
        pred (dict): Dictionary with the prediction data (as numpy arrays).
            {
                'bbox':       [...],
                'label':      [...],     
                'score':      [...],
                'difficulty': [...],
                ...
            }
        target (dict): Dictionary with the target data (as numpy arrays).
            {
                'bbox':       [...],
                'label':      [...],     
                'score':      [...],
                'difficulty': [...],
                ...
            }
        classes (number[]): List of classes which should be evaluated.
            Default is [0].
        difficulties (number[]): List of difficulties which should evaluated.
            Default is [0].
        min_overlap (number[]): Minimal overlap required to match bboxes.
            One entry for each class expected. Default is [0.5].
        bev (boolean): Use BEV IoU (else 3D IoU is used).
            Default is True.
        similar_classes (dict): Assign classes to similar classes that were not part of the training data so that they are not counted as false negatives.
            Default is {}.

    Returns:
        A tuple with a list of detection quantities 
        (score, true pos., false. pos) for each box
        and a list of the false negatives.
    """
    sim_values = list(similar_classes.values())

    # pre-filter data, remove unknown classes 
    pred = filter_data(pred, classes)[0]
    target = filter_data(target, classes+sim_values)[0]

    f_iou = bev_iou if bev else box3d_iou
    overlap = f_iou(pred, target)

    detection = np.zeros((len(difficulties), len(classes), len(pred['bbox']), 3))
    fns = np.zeros((len(difficulties), len(classes), 1), dtype="int64")
    for j, label in enumerate(classes):
        # filter only with label
        pred_label, pred_idx_l = filter_data(pred, [label])
        target_label, target_idx_l = filter_data(target, [label, similar_classes.get(label)])
        overlap_label = overlap[pred_idx_l][:, target_idx_l]
        for i, diff in enumerate(difficulties):
            # filter with difficulty
            pred_idx = filter_data(pred_label, [label], [diff])[1]
            target_idx = filter_data(target_label, [label], [diff])[1]

            if len(pred_idx) > 0:
                # no matching gt box (filtered preds vs all targets)
                fp = np.all(overlap_label[pred_idx] < min_overlap[j], axis=1).astype("float32")

                # identify all matches (filtered preds vs filtered targets)
                match_cond = np.any(overlap_label[pred_idx][:, target_idx] >= min_overlap[j], axis=-1)
                tp = np.zeros((len(pred_idx),))

                # all matches first fp
                fp[np.where(match_cond)] = 1

                # only best match can be tp
                max_idx = np.argmax(overlap_label[:, target_idx], axis=0)
                max_cond = [idx in max_idx for idx in pred_idx]
                match_cond = np.all([max_cond, match_cond], axis=0)
                tp[match_cond] = 1
                fp[match_cond] = 0

                # no matching pred box (all preds vs filtered targets)
                fns[i, j] = np.sum(np.all(overlap_label[:, target_idx] < min_overlap[j], axis=0))
                detection[i, j, [pred_idx]] = np.stack([pred_label['score'][pred_idx], tp, fp], axis=-1)
            else:
                fns[i, j] = len(target_idx)

    return detection, fns


def sample_thresholds(scores, gt_cnt, sample_cnt=41):
    """Computes equally spaced sample thresholds from given scores
    Args:
        scores (list): list of scores
        gt_cnt (number): amount of gt samples
        sample_cnt (number): amount of samples 
            Default is 41.
    Returns:
        Returns a list of equally spaced samples of the input scores.
    """
    scores = np.sort(scores)[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / gt_cnt
        r_recall = (i + 2) / gt_cnt if i < (len(scores) - 1) else l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        thresholds.append(score)
        current_recall += 1 / (sample_cnt - 1.0)
    return thresholds


def mAP(pred, target, classes=[0], difficulties=[0], min_overlap=[0.5], bev=True, samples=41, similar_classes={}):
    """Computes mAP of the given prediction (11-point interpolation).
    Args:
        pred (dict): List of dictionaries with the prediction data (as numpy arrays).
            {
                'bbox':       [...],
                'label':      [...],     
                'score':      [...],
                'difficulty': [...]
            }[]
        target (dict): List of dictionaries with the target data (as numpy arrays).
            {
                'bbox':       [...],
                'label':      [...],     
                'score':      [...],
                'difficulty': [...]
            }[]
        classes (number[]): List of classes which should be evaluated.
            Default is [0].
        difficulties (number[]): List of difficulties which should evaluated.
            Default is [0].
        min_overlap (number[]): Minimal overlap required to match bboxes.
            One entry for each class expected. Default is [0.5].
        bev (boolean): Use BEV IoU (else 3D IoU is used).
            Default is True.
        samples (number): Count of used samples for mAP calculation.
            Default is 41.
        similar_classes (dict): Assign classes to similar classes that were not part of the training data so that they are not counted as false negatives.
            Default is {}.

    Returns:
        Returns the mAP for each class and difficulty specified.
    """

    cnt = 0
    box_cnts = [0]
    for p in pred:
        cnt += len(p['bbox'])
        box_cnts.append(cnt)
    
    detection = np.zeros((len(difficulties), len(classes), box_cnts[-1], 3))
    fns = np.zeros((len(difficulties), len(classes), 1), dtype='int64')
    for i in range(len(pred)):
        d, f = precision_3d(
            pred=pred[i], target=target[i], classes=classes, 
            difficulties=difficulties, min_overlap=min_overlap, bev=bev)
        detection[:,:,box_cnts[i]:box_cnts[i+1]] = d
        fns += f

    mAP = np.empty((len(difficulties), len(classes), 1))
    for j in range(len(classes)):
        for i in range(len(difficulties)):
            det = detection[i,j,np.argsort(-detection[i,j,:,0])]

            gt_cnt = np.sum(det[:,1]) + fns[i, j]
            thresholds = sample_thresholds(det[np.where(det[:,1] > 0)[0],0], gt_cnt, samples)

            tp_acc = np.zeros((len(thresholds),))
            fp_acc = np.zeros((len(thresholds),))

            for ti in range(len(thresholds)):
                d = det[np.where(det[:,0] >= thresholds[ti])[0]]
                tp_acc[ti] = np.sum(d[:,1])
                fp_acc[ti] = np.sum(d[:,2])

            prec = tp_acc / (tp_acc + fp_acc)
            mAP[i, j] = np.sum(prec) / samples * 100

    return mAP
            
