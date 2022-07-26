import numpy as np
import pickle
import os

rare_index = np.array([ 9, 23,28, 45,51, 56,63, 64,67, 71,77, 78,81, 84,85, 91,100,101,105,108,113,128,136,137,150,159,166,167,169,173,180,182,185,189,190,193,196,199,206,207,215,217,223,228,230,239,240,255,256,258,261,262,263,275,280,281,282,287,290,293,304,312,316,318,326,329,334,335,346,351,352,355,359,365,380,382,390,391,392,396,398,399,400,402,403,404,405,406,408,411,417,419,427,428,430,432,437,440,441,450,452,464,470,475,483,486,499,500,505,510,515,518,521,523,527,532,536,540,547,548,549,550,551,552,553,556,557,561,579,581,582,587,593,594,596,597,598,600,]) - 1
rare = np.zeros(600)
rare[rare_index] += 2

obj_range = [
    (161, 170), (11, 24),   (66, 76),   (147, 160), (1, 10), 
    (55, 65),   (187, 194), (568, 576), (32, 46),   (563, 567), 
    (326, 330), (503, 506), (415, 418), (244, 247), (25, 31), 
    (77, 86),   (112, 129), (130, 146), (175, 186), (97, 107), 
    (314, 325), (236, 239), (596, 600), (343, 348), (209, 214), 
    (577, 584), (353, 356), (539, 546), (507, 516), (337, 342), 
    (464, 474), (475, 483), (489, 502), (369, 376), (225, 232), 
    (233, 235), (454, 463), (517, 528), (534, 538), (47, 54), 
    (589, 595), (296, 305), (331, 336), (377, 383), (484, 488), 
    (253, 257), (215, 224), (199, 208), (439, 445), (398, 407), 
    (258, 264), (274, 283), (357, 363), (419, 429), (306, 313), 
    (265, 273), (87, 92),   (93, 96),   (171, 174), (240, 243), 
    (108, 111), (551, 558), (195, 198), (384, 389), (394, 397), 
    (435, 438), (364, 368), (284, 290), (390, 393), (408, 414), 
    (547, 550), (450, 453), (430, 434), (248, 252), (291, 295), 
    (585, 588), (446, 449), (529, 533), (349, 352), (559, 562)
]


hoi_no_inter_all = [
    10,24,31,46,54,65,76,86,92,96,107,111,129,146,160,170,174,186,194,198,208,214,
    224,232,235,239,243,247,252,257,264,273,283,290,295,305,313,325,330,336,342,348,
    352,356,363,368,376,383,389,393,397,407,414,418,429,434,438,445,449,453,463,474,
    483,488,502,506,516,528,533,538,546,550,558,562,567,576,584,588,595,600
]


def getSigmoid(b,c,d,x,a=6):
    e = 2.718281828459
    return a/(1+e**(b-c*x))+d


def iou_(bb1, bb2, debug = False):
    #  x1, y1, x2, y2
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

    if debug:
        print(x1, y1, x2, y2, xiou, yiou)
        print(x1 * y1, x2 * y2, xiou * yiou)
    if xiou * yiou <= 0:
        return 0
    else:
        return xiou * yiou / (x1 * y1 + x2 * y2 - xiou * yiou)


def calc_hit_(det, gtbox):
    gtbox = gtbox.astype(np.float64)
    hiou = iou_(det[:4], gtbox[:4])
    oiou = iou_(det[4:], gtbox[4:])
    return min(hiou, oiou)

def iou(bb1, bb2, debug = False):
    x1 = bb1[2] - bb1[0]
    y1 = bb1[3] - bb1[1]
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    
    
    x2 = bb2[1] - bb2[0]
    y2 = bb2[3] - bb2[2]
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0
    
    
    xiou = min(bb1[2], bb2[1]) - max(bb1[0], bb2[0])
    yiou = min(bb1[3], bb2[3]) - max(bb1[1], bb2[2])
    if xiou < 0:
        xiou = 0
    if yiou < 0:
        yiou = 0

    if debug:
        print(x1, y1, x2, y2, xiou, yiou)
        print(x1 * y1, x2 * y2, xiou * yiou)
    if xiou * yiou <= 0:
        return 0
    else:
        return xiou * yiou / (x1 * y1 + x2 * y2 - xiou * yiou)

def calc_hit(det, gtbox):
    gtbox = gtbox.astype(np.float64)
    hiou = iou(det[:4], gtbox[:4])
    oiou = iou(det[4:], gtbox[4:])
    return min(hiou, oiou)

def calc_ap(scores, bboxes, keys, hoi_id, begin, gt_path="gt_hoi_py2", nms=False):
    score = scores[:, hoi_id - begin]

    hit = []
    idx = np.argsort(score)[::-1]
    gt_bbox = pickle.load(open('utils/%s/hoi_%d.pkl' % (gt_path, hoi_id), 'rb'), encoding='latin1')

    npos = 0
    used = {}

    if len(gt_bbox.keys()) == 0:
        return np.nan, np.nan

    for key in gt_bbox.keys():
        gt_bbox[key] = np.array(gt_bbox[key])
        npos += gt_bbox[key].shape[0]
        used[key] = set()
    if len(idx) == 0:
        return 0, 0, 0, 0
    
    if nms:
        ### nms
        Key2PairId = {}
        for i in range(min(len(idx), 19999)):
            pair_id = idx[i]
            bbox = bboxes[pair_id, :]
            key  = keys[pair_id]
            
            if score[pair_id] <= 0:
                break
            if key not in Key2PairId.keys():
                Key2PairId[key] = []
            Key2PairId[key].append(pair_id)
        
        sel_all = []
        for key, pair_id_list in Key2PairId.items():
            sel = []
            deleted = []
            for i in range(len(pair_id_list)):
                cur_pair_id = pair_id_list[i]
                if cur_pair_id not in deleted:
                    sel.append(cur_pair_id)
                    for pair_id in pair_id_list[i+1:]:
                        if pair_id not in deleted:
                            if calc_hit_(bboxes[pair_id, :], bboxes[cur_pair_id, :]) >= 0.7:
                                deleted.append(pair_id)
            assert len(pair_id_list) == len(sel) + len(deleted)
            sel_all.extend(sel)
        
        sel_all = np.array(sel_all)
        score, keys, bboxes = score[sel_all], keys[sel_all], bboxes[sel_all]
        idx = np.argsort(score)[::-1]
        ### nms

    for i in range(min(len(idx), 19999)):
        pair_id = idx[i]
        bbox = bboxes[pair_id, :]
        key  = keys[pair_id]

        if key in gt_bbox:
            maxi = 0.0
            k    = -1
            for i in range(gt_bbox[key].shape[0]):
                tmp = calc_hit(bbox, gt_bbox[key][i, :])
                
                if maxi < tmp:
                    maxi = tmp
                    k    = i
            if k in used[key] or maxi < 0.5:
                hit.append(0)
            else:
                hit.append(1)
                used[key].add(k)
        else:
            hit.append(0)
    bottom = np.array(range(len(hit))) + 1
    hit    = np.cumsum(hit)
    rec    = hit / npos
    prec   = hit / bottom
    ap     = 0.0
    for i in range(11):
        mask = rec >= (i / 10.0)
        if np.sum(mask) > 0:
            ap += np.max(prec[mask]) / 11.0

    return ap, np.max(rec)

def get_map(keys, scores, bboxes):
    map  = np.zeros(600)
    mrec = np.zeros(600)
    for i in range(80):
        begin = obj_range[i][0] - 1
        end   = obj_range[i][1]
        for hoi_id in range(begin, end):
            score = scores[i]
            bbox  = bboxes[i]
            key   = keys[i]
            if len(score) == 0:
                continue
            map[hoi_id], mrec[hoi_id] = calc_ap(score, bbox, key, hoi_id, begin)
    return map, mrec


def get_map_wo_nointer(keys, scores, bboxes, gt_path="gt_hoi_py2", nms=False):
    map  = np.zeros(600)
    mrec = np.zeros(600)
    for i in range(80):
        begin = obj_range[i][0] - 1
        end   = obj_range[i][1] 
        for hoi_id in range(begin, end-1):
            score = scores[i]
            bbox  = bboxes[i]
            key   = keys[i]
            if len(score) == 0:
                continue
            map[hoi_id], mrec[hoi_id] = calc_ap(score, bbox, key, hoi_id, begin, gt_path, nms)
        map[end-1], mrec[end-1] = np.nan, np.nan
    return map, mrec


def calc_ap_verb(scores, bboxes, keys, verb_id):
    score = scores[:,verb_id]
    hit = []
    idx = np.argsort(score)[::-1]
    gt_bbox = pickle.load(open('./gt_verb/verb_%d.pkl' % verb_id, 'rb'), encoding='latin1')
    npos = 0
    used = {}

    for key, value in gt_bbox.items():
        gt_bbox[key] = np.array(gt_bbox[key])

    for key in gt_bbox.keys():
        npos += gt_bbox[key].shape[0]
        used[key] = set()
    if len(idx) == 0:
        return 0, 0, 0, 0
    for i in range(len(idx)):
        if (score[idx[i]] == 0):
            break
        pair_id = idx[i]
        bbox = bboxes[pair_id, :]
        key  = keys[pair_id]

        if key in gt_bbox.keys():
            maxi = 0.0
            k    = -1
            for i in range(gt_bbox[key].shape[0]):
                tmp = calc_hit(bbox, gt_bbox[key][i, :])
                
                if maxi < tmp:
                    maxi = tmp
                    k    = i
            if k in used[key] or maxi < 0.5:
                hit.append(0)
            else:
                hit.append(1)
                used[key].add(k)
        else:
            hit.append(0)
    
    bottom = np.array(range(len(hit))) + 1
    hit    = np.cumsum(hit)
    rec    = hit / npos
    prec   = hit / bottom
    ap     = 0.0
    for i in range(11):
        mask = rec >= (i / 10.0)
        if np.sum(mask) > 0:
            ap += np.max(prec[mask]) / 11.0

    return ap, np.max(rec)


def get_map_verb(keys, scores, bboxes, calc_nointeraction = True):
    map  = np.zeros(117)
    mrec  = np.zeros(117)

    for i in range(117):
        if not calc_nointeraction and i==0:
            map[i], mrec[i] = np.nan, np.nan
            continue
        map[i], mrec[i] = calc_ap_verb(scores, bboxes, keys, i)

    return map, mrec