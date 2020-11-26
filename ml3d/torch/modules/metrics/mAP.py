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


def precision_3d(pred, target, classes=[0], difficulties=[0], min_overlap=[0.5], bev=True):
    """Computes precision quantities for each predicted box.
    Args:
        pred (dict): Dictionary with the prediction data (as numpy arrays).
            {
                'bbox':  [...],
                'label': [...],     
                'score': [...]
            }
        target (dict): Dictionary with the target data (as numpy arrays).
            {
                'bbox':       [...],
                'label':      [...],     
                'score':      [...],
                'difficulty': [...]
            }
        classes (number[]): List of classes which should be evaluated.
            Default is [0].
        difficulties (number[]): List of difficulties which should evaluated.
            Default is [0].
        min_overlap (number[]): Minimal overlap required to match bboxes.
            One entry for each class expected. Default is [0.5].
        bev: Use BEV IoU (else 3D IoU is used).
            Default is True.

    Returns:
        A tuple with a list of detection quantities 
        (score, true pos., false. pos) for each box
        and a list of the false negatives.
    """
    # pre-filter data, remove unknown classes 
    pred = filter_data(pred, classes)[0]
    target = filter_data(target, classes)[0]

    f_iou = bev_iou if bev else box3d_iou
    overlap = f_iou(pred, target)

    detection = np.zeros((len(difficulties), len(classes), len(pred['bbox']), 3))
    fns = np.zeros((len(difficulties), len(classes), 1), dtype="int64")
    for j, label in enumerate(classes):
        # filter only with label
        pred_label, pred_idx_l = filter_data(pred, [label])
        target_label, target_idx_l = filter_data(target, [label])
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
                # at least one match
                if  np.any(match_cond):
                    # all matches fp
                    fp[np.where(match_cond)] = 1

                    # only best match may be tp
                    max_idx = np.argmax(overlap_label[:, target_idx], axis=0)
                    max_cond = np.where([idx in max_idx for idx in pred_idx])
                    tp[max_cond] = 1
                    fp[max_cond] = 0

                # no matching pred box (all preds vs filtered targets)
                fns[i, j] = np.sum(np.all(overlap_label[:, target_idx] < min_overlap[j], axis=0))
                detection[i, j, [pred_idx]] = np.stack([pred['score'][pred_idx], tp, fp], axis=-1)
            else:
                fns[i, j] = len(target_idx)

    return detection, fns


def mAP(pred, target, classes=[0], difficulties=[0], min_overlap=[0.5], bev=True):
    """Computes mAP of the given prediction (11-point interpolation).
    Args:
        pred (dict): List of dictionaries with the prediction data (as numpy arrays).
            {
                'bbox':  [...],
                'label': [...],     
                'score': [...]
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
        bev: Use BEV IoU (else 3D IoU is used).
            Default is True.

    Returns:
        Returns the mAP for each class and difficulty specified.
    """

    cnt = 0
    box_cnts = [0]
    for p in pred:
        cnt += len(p['bbox'])
        box_cnts.append(cnt)
    
    detection = np.ones((len(difficulties), len(classes), box_cnts[-1], 3))
    fns = np.ones((len(difficulties), len(classes), 1), dtype='int64')
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
            tp_acc, fp_acc = 0, 0
            tp_sum = np.sum(det[:,1])
            prec = np.zeros((len(det),))
            recall = np.zeros((len(det),))

            for k, (tp, fp) in enumerate(det[...,1:]):
                tp_acc += tp
                fp_acc += fp
                if tp_acc + fp_acc > 0:
                    prec[k] = tp_acc / (tp_acc + fp_acc)
                if tp_acc + fns[i, j] > 0:
                    recall[k] = tp_acc / (tp_sum + fns[i, j])
                
            AP = 0
            for r in np.linspace(1.0, 0.0, 11):
                p = prec[recall >= r]
                if len(p) > 0:
                    AP += np.max(p)

            mAP[i, j] = 100 * AP / 11

    return mAP
            
