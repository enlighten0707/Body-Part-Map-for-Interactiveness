# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class HungarianMatcherHOI(nn.Module):

    def __init__(self, cost_obj_class: float = 1, cost_verb_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, binary: bool = False, cost_binary: float = 1, match_binary_part=False):
        super().__init__()
        self.cost_obj_class  = cost_obj_class
        self.cost_verb_class = cost_verb_class
        self.cost_bbox       = cost_bbox
        self.cost_giou       = cost_giou
        self.binary          = binary
        self.cost_binary     = cost_binary
        assert cost_obj_class != 0 or cost_verb_class != 0 or cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_obj_logits'].shape[:2]

        out_obj_prob = outputs['pred_obj_logits'].flatten(0, 1).softmax(-1)
        out_sub_bbox = outputs['pred_sub_boxes'].flatten(0, 1)
        out_obj_bbox = outputs['pred_obj_boxes'].flatten(0, 1)
        for v in targets:
            v['obj_labels'] = v['obj_labels'][v['pair2obj']]
            v['obj_boxes']  = v['obj_boxes'][v['pair2obj']]
            v['sub_boxes']  = v['sub_boxes'][v['pair2sub']]
        
        tgt_obj_labels = torch.cat([v['obj_labels'] for v in targets])
        tgt_sub_boxes  = torch.cat([v['sub_boxes'] for v in targets])
        tgt_obj_boxes  = torch.cat([v['obj_boxes'] for v in targets])

        cost_obj_class = -out_obj_prob[:, tgt_obj_labels]
        
        out_verb_prob   = outputs['pred_verb_logits'].flatten(0, 1).sigmoid()
        tgt_verb_labels = torch.cat([v['verb_labels'] for v in targets])
        tgt_verb_labels_permute = tgt_verb_labels.permute(1, 0)
        cost_verb_class = -(out_verb_prob.matmul(tgt_verb_labels_permute) / (tgt_verb_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                            (1 - out_verb_prob).matmul(1 - tgt_verb_labels_permute) / ((1 - tgt_verb_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2
        if self.binary:
            if outputs['pred_binary_logits'].shape[-1] == 2:
                out_binary_prob   = outputs['pred_binary_logits'].flatten(0, 1).sigmoid()
                tgt_binary_labels = torch.cat([v['binary_labels'] for v in targets]).long()
                cost_binary_class = -out_binary_prob[:, tgt_binary_labels]
            else:
                out_binary_prob   = outputs['pred_binary_logits'].flatten(0, 1) # (bz * 64)
                cost_binary_class = -out_binary_prob.unsqueeze(-1)

        cost_sub_bbox = torch.cdist(out_sub_bbox, tgt_sub_boxes, p=1)
        cost_obj_bbox = torch.cdist(out_obj_bbox, tgt_obj_boxes, p=1) * (tgt_obj_boxes != 0).any(dim=1).unsqueeze(0)
        if cost_sub_bbox.shape[1] == 0:
            cost_bbox = cost_sub_bbox
        else:
            cost_bbox = torch.stack((cost_sub_bbox, cost_obj_bbox)).max(dim=0)[0]

        cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))
        cost_obj_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes)) + \
                        cost_sub_giou * (tgt_obj_boxes == 0).all(dim=1).unsqueeze(0)
        if cost_sub_giou.shape[1] == 0:
            cost_giou = cost_sub_giou
        else:
            cost_giou = torch.stack((cost_sub_giou, cost_obj_giou)).max(dim=0)[0]

        C = self.cost_obj_class * cost_obj_class + self.cost_verb_class * cost_verb_class + \
            self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        if self.binary:
            C += self.cost_binary * cost_binary_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v['verb_labels']) for v in targets]
        
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        obj_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    
        return obj_indices, obj_indices, obj_indices, outputs



class CascadedHungarianMatcherHOI(nn.Module):

    def __init__(self, cost_obj_class: float = 1, cost_verb_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, binary: bool = False, cost_binary: float = 1, root_hum: int = 3):
        super().__init__()
        self.cost_obj_class  = cost_obj_class
        self.cost_verb_class = cost_verb_class
        self.cost_bbox       = cost_bbox
        self.cost_giou       = cost_giou
        self.binary          = binary
        self.cost_binary     = cost_binary
        self.num_hum         = root_hum ** 2
        assert cost_obj_class != 0 or cost_verb_class != 0 or cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs, targets):
    
        bs, num_queries = outputs['pred_obj_logits'].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_obj_prob = outputs["pred_obj_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_obj_bbox = outputs["pred_obj_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_obj_labels = torch.cat([v["obj_labels"] for v in targets])
        tgt_obj_boxes  = torch.cat([v["obj_boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_obj_class = -out_obj_prob[:, tgt_obj_labels]
        # Compute the L1 cost between boxes
        cost_obj_bbox = torch.cdist(out_obj_bbox, tgt_obj_boxes, p=1) * (tgt_obj_boxes != 0).any(dim=1).unsqueeze(0)

        # Compute the giou cost betwen boxes
        cost_obj_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes))

        # Final cost matrix
        C = self.cost_bbox * cost_obj_bbox + self.cost_obj_class * cost_obj_class + self.cost_giou * cost_obj_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes   = [len(v["obj_boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        obj_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices] # [(pred_idx, tgt_idx), ...] of length N, tgt_idx
        ######################################
        bs = outputs['pred_sub_boxes'].shape[0]

        out_sub_bbox  = outputs['pred_sub_boxes'].flatten(0, 1)
        out_verb_prob = outputs['pred_verb_logits'].flatten(0, 1).sigmoid()

        tgt_sub_boxes = torch.cat([v['sub_boxes'][v['pair2sub']] for v in targets])
        tgt_verb_labels = torch.cat([v['verb_labels'] for v in targets])
        
        tgt_verb_labels_permute = tgt_verb_labels.permute(1, 0)
        cost_verb_class = -(out_verb_prob.matmul(tgt_verb_labels_permute) / (tgt_verb_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                            (1 - out_verb_prob).matmul(1 - tgt_verb_labels_permute) / ((1 - tgt_verb_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2
        if self.binary:
            out_binary_prob   = outputs['pred_binary_logits'].flatten(0, 1).sigmoid()
            tgt_binary_labels = torch.cat([v['binary_labels'] for v in targets]).long()
            cost_binary_class = -out_binary_prob[:, tgt_binary_labels]

        cost_sub_bbox = torch.cdist(out_sub_bbox, tgt_sub_boxes, p=1)
        cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))

        C = self.cost_verb_class * cost_verb_class + self.cost_bbox * cost_sub_bbox + self.cost_giou * cost_sub_giou
        if self.binary:
            C += self.cost_binary * cost_binary_class
        C = C.view(bs, num_queries, self.num_hum, -1).cpu()

        sizes = [len(v['verb_labels']) for v in targets]
        pair_indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            local_pred_idx, local_tgt_idx = [], []
            pair2obj = targets[i]['pair2obj']
            pred_obj_idx, tgt_obj_idx = obj_indices[i]
            for j in range(len(pred_obj_idx)):
                tmp = torch.where(pair2obj == tgt_obj_idx[j])[0]
                row_ind, col_ind = linear_sum_assignment(c[i][pred_obj_idx[j]][:, tmp])
                local_pred_idx.append(torch.as_tensor(row_ind, dtype=torch.int64) + pred_obj_idx[j] * self.num_hum)
                local_tgt_idx  += tmp[col_ind]
            if len(local_pred_idx) > 0:
                pair_indices.append((torch.cat(local_pred_idx), torch.as_tensor(local_tgt_idx, dtype=torch.int64)))
            else:
                pair_indices.append((torch.as_tensor(local_pred_idx, dtype=torch.int64), torch.as_tensor(local_tgt_idx, dtype=torch.int64)))
        ######################## Needs refinement
        tgt_sub_boxes = torch.cat([v['sub_boxes'] for v in targets])
        cost_sub_bbox = torch.cdist(out_sub_bbox, tgt_sub_boxes, p=1)
        cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))
        C_sub = self.cost_bbox * cost_sub_bbox + self.cost_giou * cost_sub_giou
        C_sub = C_sub.view(bs, num_queries, self.num_hum, -1).cpu()
        sizes = [len(v['sub_boxes']) for v in targets]
        sub_indices = []
        for i, c in enumerate(C_sub.split(sizes, -1)):
            local_pred_idx, local_tgt_idx = [], []
            pred_obj_idx, tgt_obj_idx = obj_indices[i]    # object mapping for i-th image
            pred_pair_idx, tgt_pair_idx = pair_indices[i] # pair mapping for i-th image
            tgt_sub_idx = targets[i]['pair2sub'][tgt_pair_idx] # tgt sub in pair_mapping for i-th image
            pred_pair_idx_obj = pred_pair_idx // self.num_hum  # pred_obj in pair_mapping for i-th image
            pred_pair_idx_hum = pred_pair_idx % self.num_hum   # pred_hum in pair_mapping for i-th image
            for j in range(len(pred_obj_idx)):
                ############### todo
                tmp = torch.zeros(self.num_hum, dtype=torch.int64) 
                tmp[pred_pair_idx_hum[pred_pair_idx_obj == pred_obj_idx[j]]] += 1 # 1 if the pred pair has been matched by pred_obj_idx[j]
                pred_tmp = torch.where(tmp < 1)[0] # the unmatched pairs
                tmp = torch.zeros(c[i].shape[-1], dtype=torch.int64)
                tmp[tgt_sub_idx[pred_pair_idx_obj == pred_obj_idx[j]]] += 1       # 1 if the tgt sub has been matched by pred_obj_idx[j]
                tgt_tmp = torch.where(tmp < 1)[0]  # the unmatched subs
                row_ind, col_ind = linear_sum_assignment(c[i][pred_obj_idx[j]][pred_tmp, :][:, tgt_tmp])
                local_pred_idx.append(torch.as_tensor(pred_tmp[row_ind], dtype=torch.int64) + pred_obj_idx[j] * self.num_hum)
                local_tgt_idx  += tgt_tmp[col_ind]
            if len(local_pred_idx) > 0:
                sub_indices.append((torch.cat([torch.cat(local_pred_idx).to(pred_pair_idx.device), pred_pair_idx]), torch.cat([torch.as_tensor(local_tgt_idx, dtype=torch.int64, device=tgt_sub_idx.device), tgt_sub_idx])))
            else:
                sub_indices.append((pred_pair_idx, tgt_sub_idx))
        
        return obj_indices, sub_indices, pair_indices


def build_matcher(args):
    if args.hoi:
        if args.cascaded_matcher:
            return CascadedHungarianMatcherHOI(cost_obj_class=args.set_cost_obj_class, cost_verb_class=args.set_cost_verb_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou, binary=args.binary, cost_binary=args.set_cost_binary, root_hum=args.root_hum)
        else:
            return HungarianMatcherHOI(cost_obj_class=args.set_cost_obj_class, cost_verb_class=args.set_cost_verb_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou, binary=args.binary, cost_binary=args.set_cost_binary, match_binary_part=("binary_6v_labels" in args.losses))
    else:
        return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
