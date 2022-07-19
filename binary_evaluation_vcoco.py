import pickle
import numpy as np
import os.path as osp

def iou(bb1, bb2):
    x1 = bb1[2] - bb1[0]
    y1 = bb1[3] - bb1[1]
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0

    x2 = bb2[2] - bb2[0]
    y2 = bb2[3] - bb2[1]
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0

    xiou = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])
    yiou = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])
    if xiou < 0:
        xiou = 0
    if yiou < 0:
        yiou = 0

    if xiou * yiou <= 0:
        return 0
    else:
        return xiou * yiou / (x1 * y1 + x2 * y2 - xiou * yiou)

def calc_hit(det, gtbox):
    gtbox = gtbox.astype(np.float64)
    hiou  = iou(det[:4], gtbox[:4])
    # if gtbox[4] is None:
    #     return hiou
    oiou  = iou(det[4:], gtbox[4:])
    return min(hiou, oiou)

def calc_binary(keys, bboxes, score):
    hit = []
    idx = np.argsort(score)[::-1]
    gt_bbox = pickle.load(open('./gt_binary_VCOCO.pkl', 'rb'))
    npos = 0
    used = {}
    for key in gt_bbox.keys():
        npos += gt_bbox[key].shape[0]
        used[key] = set()

    for j in range(len(idx)):
        pair_id = idx[j]
        bbox = bboxes[pair_id]
        key = keys[pair_id]

        if key in gt_bbox.keys():
            maxi = 0.0
            k = -1
            for i in range(gt_bbox[key].shape[0]):
                tmp = calc_hit(bbox, gt_bbox[key][i, :])
                if maxi < tmp:
                    maxi = tmp
                    k = i
            if k in used[key] or maxi < 0.5:
                hit.append(0)
            else:
                hit.append(1)
                used[key].add(k)
        else:
            hit.append(0)

    bottom = np.array(range(len(hit))) + 1
    hit = np.cumsum(np.array(hit))

    rec = hit / npos
    prec = hit / bottom
    ap = 0.0
    for i in range(11):
        mask = rec >= (i / 10.0)
        if np.sum(mask) > 0:
            ap += np.max(prec[mask]) / 11.0

    return ap, np.max(rec)
