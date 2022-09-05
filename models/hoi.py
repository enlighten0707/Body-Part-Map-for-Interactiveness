# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from cmath import cos
from scipy.optimize import linear_sum_assignment

import torch
from torch import nn
import torch.nn.functional as F

from queue import Queue
from collections import OrderedDict
from itertools import zip_longest
import numpy as np
import math

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

class DETR_PartMap(nn.Module):

    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False, binary=False, depth=False, depth_cat=100, spmap=False, root_hum=3, ref=False, extract=False, config = None):
        super().__init__()
        
        self.config = config
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.binary   = binary
        self.depth    = depth
        self.spmap    = spmap
        self.ref      = ref
        self.extract  = extract

        ### Interactiveness
        if self.binary:
            self.binary_class_embed = nn.Linear(hidden_dim, 2)

    def get_mask_from_query(self, outputs_coord, img_size, mask):

        outputs_obj_coord_xyxy = box_cxcywh_to_xyxy(outputs_coord) # (bz, num_query, 4), outputs_obj_class: [cx, cy, w, h], ratio
        outputs_obj_coord_xyxy = outputs_obj_coord_xyxy.detach()
        outputs_obj_coord_xyxy[..., 0] = outputs_obj_coord_xyxy[..., 0] * img_size[..., 1].unsqueeze(-1) // 32 # (bz, num_query) * (bz)
        outputs_obj_coord_xyxy[..., 1] = outputs_obj_coord_xyxy[..., 1] * img_size[..., 0].unsqueeze(-1) // 32
        outputs_obj_coord_xyxy[..., 2] = outputs_obj_coord_xyxy[..., 2] * img_size[..., 1].unsqueeze(-1) // 32
        outputs_obj_coord_xyxy[..., 3] = outputs_obj_coord_xyxy[..., 3] * img_size[..., 0].unsqueeze(-1) // 32
        outputs_obj_coord_xyxy = outputs_obj_coord_xyxy.long()
        mask_object = []
        for i in range(mask.shape[0]): # bz
            for j in range(self.num_queries):
                w1, h1, w2, h2 = outputs_obj_coord_xyxy[i, j]
                w2 = torch.minimum(w2, torch.tensor(mask.shape[2]-1).cuda().long())
                h2 = torch.minimum(h2, torch.tensor(mask.shape[1]-1).cuda().long())

                cur_mask_object = torch.zeros((mask.shape[1], mask.shape[2]), dtype=bool).cuda()
                if h1 > 0:
                    cur_mask_object = cur_mask_object.index_fill(0, torch.arange(0, h1).cuda(), True)
                if h2+1 < cur_mask_object.shape[0]:
                    cur_mask_object = cur_mask_object.index_fill(0, torch.arange(h2+1, cur_mask_object.shape[0]).cuda(), True)
                if w1 > 0:
                    cur_mask_object = cur_mask_object.index_fill(1, torch.arange(0, w1).cuda(), True)
                if w2+1 < cur_mask_object.shape[1]:
                    cur_mask_object = cur_mask_object.index_fill(1, torch.arange(w2+1, cur_mask_object.shape[1]).cuda(), True)

                mask_object.append(cur_mask_object)
        mask_object = torch.stack(mask_object, 0) # (bz * num_query, H/32, W/32)
        mask_object = mask_object.view(mask.shape[0], self.num_queries, mask_object.shape[-2], mask_object.shape[-1])
        return mask_object

    def forward(self, samples: NestedTensor, depth: NestedTensor = None, spmap: NestedTensor = None, mask_part = None, img_size=None, targets=None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples) # samples: [bz, 3, H, W], [bz, H, W]

        features, pos = self.backbone(samples) # pos: (bz, 256, H/32, W/32)
        src, mask = features[-1].decompose() # src: (bz, 2048, H/32, W/32), mask: (bz, H/32, W/32)
        assert mask is not None
        
        if mask_part is not None:
            mask_part = mask_part.decompose()[0] # mask_part: (bz, 6=num_part, H/32, W/32)
            mask_part = mask_part[..., :mask.shape[1], :mask.shape[2]] # mask_part: (bz, 6=num_part, H/32, W/32)

        # box decoder
        hs_box, memory = self.transformer.forward_box(self.input_proj(src), mask, self.query_embed.weight, pos[-1]) # hs_obj, hs_hum, hs_verb: (6=num_decoder, bz, 64=num_query, 256)
        hs_obj = hs_hum = hs_box
        outputs_obj_class  = self.obj_class_embed(hs_obj) # (6, bz, num_query, 80)
        outputs_obj_coord  = self.obj_bbox_embed(hs_obj).sigmoid() # (6, bz, num_query, 4)
        outputs_sub_coord  = self.sub_bbox_embed(hs_hum).sigmoid() # (6, bz, num_query, 4)

        # get mask from query
        mask_object = self.get_mask_from_query(outputs_obj_coord[-1], img_size, mask)
        mask_human = self.get_mask_from_query(outputs_sub_coord[-1], img_size, mask)

        bs = src.shape[0]
        matched_6v = torch.zeros((bs, self.num_queries, 6)).cuda() # (bz, num_query, 6)

        # binary_decoder
        if self.binary:
            hs_binary, util_value = self.transformer.forward_binary(memory, mask, pos[-1], hs_box, \
                                            mask_part, mask_object, mask_human, self.num_queries, matched_6v)
            
        # prediction
        if "verb_labels" in self.config.losses:
             outputs_verb_class = self.verb_class_embed(hs_binary)
        else:
            outputs_verb_class = torch.zeros((6, pos.shape[0], self.num_queries, self.config.num_verb_classes)).cuda()
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1],}    
        outputs_binary_class, outputs_obj_class_ref, outputs_obj_coord_ref = None, None, None

        pred_binary_logits = None
        if self.binary:
            pred_part_binary = util_value # (bz, num_query, 6, 2)
            pred_binary_logits = self.binary_class_embed(hs_binary) # (1, bz, num_query, 256) -> (1, bz, num_query, 2)
            out['pred_binary_logits'] = pred_binary_logits[-1] # (bz, num_query, 2)
            out['pred_part_binary_logits'] = pred_part_binary.permute(2, 0, 1, 3) # (6, bz, num_query, 2)

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_sub_coord, outputs_obj_coord, pred_binary_logits, outputs_verb_class)
        return out
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_sub_coord, outputs_obj_coord, outputs_binary_class=None, outputs_verb_class=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        return [{'pred_obj_logits': a,  'pred_sub_boxes': c, 'pred_obj_boxes': d, 'pred_binary_logits': e, 'pred_verb_logits':f}
                for a, c, d, e, f in zip_longest(outputs_obj_class[:-1], 
                                outputs_sub_coord[:-1], outputs_obj_coord[:-1], outputs_binary_class[:-1], outputs_verb_class[:-1])]

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, verb_loss_type, dataset_file, alpha, obj_reweight, verb_reweight, use_static_weights, queue_size, p_obj, p_verb, extract=False, binary_loss_type="bce"):
        super().__init__()

        assert verb_loss_type == 'bce' or verb_loss_type == 'focal'

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.verb_loss_type = verb_loss_type
        self.extract = extract
        self.binary_loss_type = binary_loss_type
        
        self.alpha = alpha

        if dataset_file == 'hico':
            self.obj_nums_init = [1811, 9462, 2415, 7249, 1665, 3587, 1396, 1086, 10369, 800, \
                                  287, 77, 332, 2352, 974, 470, 1386, 4889, 1675, 1131, \
                                  1642, 185, 92, 717, 2228, 4396, 275, 1236, 1447, 1207, \
                                  2949, 2622, 1689, 2345, 1863, 408, 5594, 1178, 562, 1479, \
                                  988, 1057, 419, 1451, 504, 177, 1358, 429, 448, 186, \
                                  121, 441, 735, 706, 868, 1238, 1838, 1224, 262, 517, \
                                  5787, 200, 529, 1337, 146, 272, 417, 1277, 31, 213, \
                                  7, 102, 102, 2424, 606, 215, 509, 529, 102, 572]
        else:
            self.obj_nums_init = [5397, 238, 332, 321, 5, 6, 45, 90, 59, 20, \
                                  13, 5, 6, 313, 28, 25, 46, 277, 20, 16, \
                                  154, 0, 7, 13, 356, 191, 458, 66, 337, 1364, \
                                  1382, 958, 1166, 68, 258, 221, 1317, 1428, 759, 201, \
                                  190, 444, 274, 587, 124, 107, 102, 37, 226, 16, \
                                  30, 22, 187, 320, 222, 465, 893, 213, 56, 322, \
                                  306, 13, 55, 834, 23, 104, 38, 861, 11, 27, \
                                  0, 16, 22, 405, 50, 14, 145, 63, 9, 11]
        self.obj_nums_init.append(3 * sum(self.obj_nums_init))  # 3 times fg for bg init

        if dataset_file == 'hico':
            self.verb_nums_init = [67, 43, 157, 321, 664, 50, 232, 28, 5342, 414, \
                                   49, 105, 26, 78, 157, 408, 358, 129, 121, 131, \
                                   275, 1309, 3, 799, 2338, 128, 633, 79, 435, 1, \
                                   905, 19, 319, 47, 816, 234, 17958, 52, 97, 648, \
                                   61, 1430, 13, 1862, 299, 123, 52, 328, 121, 752, \
                                   111, 30, 293, 6, 193, 32, 4, 15421, 795, 82, \
                                   30, 10, 149, 24, 59, 504, 57, 339, 62, 38, \
                                   472, 128, 672, 1506, 16, 275, 16092, 757, 530, 380, \
                                   132, 68, 20, 111, 2, 160, 3209, 12246, 5, 44, \
                                   18, 7, 5, 4815, 1302, 69, 37, 25, 5048, 424, \
                                   1, 235, 150, 131, 383, 72, 76, 139, 258, 464, \
                                   872, 360, 1917, 1, 3775, 1206, 1]
        else:
            self.verb_nums_init = [4001, 4598, 1989, 488, 656, 3825, 367, 367, 677, 677, \
                                   700, 471, 354, 498, 300, 313, 300, 300, 622, 458, \
                                   500, 498, 489, 1545, 133, 142, 38, 116, 388]
        self.verb_nums_init.append(3 * sum(self.verb_nums_init))
        
        self.obj_reweight       = obj_reweight
        self.verb_reweight      = verb_reweight
        self.use_static_weights = use_static_weights
        
        Maxsize = queue_size

        if self.obj_reweight:
            self.q_obj = Queue(maxsize=Maxsize)
            self.p_obj = p_obj
            self.obj_weights_init = self.cal_weights(self.obj_nums_init, p=self.p_obj)

        if self.verb_reweight:
            self.q_verb = Queue(maxsize=Maxsize)
            self.p_verb = p_verb
            self.verb_weights_init = self.cal_weights(self.verb_nums_init, p=self.p_verb)

    def cal_weights(self, label_nums, p=0.5):
        num_fgs = len(label_nums[:-1])
        weight = np.zeros(num_fgs + 1)
        num_all = sum(label_nums[:-1])
        
        bottom = np.array(label_nums[:num_fgs])
        idx    = np.where(bottom > 0)[0]
        weight[idx] = np.power(num_all / bottom[idx], p)
        weight = weight / np.mean(weight[weight > 0])

        weight[-1] = np.power(num_all / label_nums[-1], p) if label_nums[-1] != 0 else 0

        weight = torch.FloatTensor(weight).cuda()
        return weight
    
    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        src_logits = outputs['pred_obj_logits']
        if src_logits is None:
            return None

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        if not self.obj_reweight:
            obj_weights = self.empty_weight
        elif self.use_static_weights:
            obj_weights = self.obj_weights_init
        else:
            obj_label_nums_in_batch = [0] * (self.num_obj_classes + 1)
            for target_class in target_classes:
                for label in target_class:
                    obj_label_nums_in_batch[label] += 1

            if self.q_obj.full():
                self.q_obj.get()
            self.q_obj.put(np.array(obj_label_nums_in_batch))
            accumulated_obj_label_nums = np.sum(self.q_obj.queue, axis=0)
            obj_weights = self.cal_weights(accumulated_obj_label_nums, p=self.p_obj)

            aphal = min(math.pow(0.999, self.q_obj.qsize()), 0.9)
            obj_weights = aphal * self.obj_weights_init + (1 - aphal) * obj_weights
        
        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, obj_weights)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        if pred_logits is None:
            return None
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses
        
    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        src_logits = outputs['pred_verb_logits']
        if src_logits is None:
            return None

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        if not self.verb_reweight:
            verb_weights = None
        elif self.use_static_weights:
            verb_weights = self.verb_weights_init
        else:
            verb_label_nums_in_batch = [0] * (self.num_verb_classes + 1)
            for target_class in target_classes:
                for label in target_class:
                    label_classes = torch.where(label > 0)[0]
                    if len(label_classes) == 0:
                        verb_label_nums_in_batch[-1] += 1
                    else:
                        for label_class in label_classes:
                            verb_label_nums_in_batch[label_class] += 1

            if self.q_verb.full():
                self.q_verb.get()
            self.q_verb.put(np.array(verb_label_nums_in_batch))
            accumulated_verb_label_nums = np.sum(self.q_verb.queue, axis=0)
            verb_weights = self.cal_weights(accumulated_verb_label_nums, p=self.p_verb)

            aphal = min(math.pow(0.999, self.q_verb.qsize()),0.9)
            verb_weights = aphal * self.verb_weights_init + (1 - aphal) * verb_weights
        
        if self.verb_loss_type == 'bce':
            loss_verb_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes)
        elif self.verb_loss_type == 'focal':
            src_logits = src_logits.sigmoid()
            loss_verb_ce = self._neg_loss(src_logits, target_classes, weights=verb_weights, alpha=self.alpha)

        losses = {'loss_verb_ce': loss_verb_ce}
        return losses
    
    def loss_binary_labels(self, outputs, targets, indices, num_interactions):
        src_logits = outputs['pred_binary_logits'] # (bz, 64=num_query)
        idx                 = self._get_src_permutation_idx(indices) # [(num_pair, ), (num_pair,)]
        target_classes_b    = torch.cat([t['binary_labels'][J] for t, (_, J) in zip(targets, indices)]) # (num_pair)
        target_classes      = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_b

        if src_logits.shape[-1] == 2:
            loss_binary_ce      = F.cross_entropy(src_logits.transpose(1, 2), target_classes) # (bz, 2, num_query), (bz, num_query) -> (bz, num_query)
        else:
            loss_binary_ce      = F.binary_cross_entropy(src_logits, target_classes.float()) # (bz, num_query)

        losses = {'loss_binary_ce': loss_binary_ce}
        return losses

    def loss_binary_consistency(self, outputs, targets, indices, num_interactions):

        binary_logits = outputs['pred_binary_logits'] # (bz, 64=num_query, 2)
        if binary_logits.shape[-1] == 2:
            binary_logits = F.softmax(binary_logits, -1)[..., 1] # (bz, 64=num_query)

        binary_part_logits = outputs['pred_part_binary_logits'] # (6=num_part, bz, 64=num_query, 2)
        if binary_part_logits.shape[-1] == 2:
            binary_part_logits = F.softmax(binary_part_logits, -1)[..., 1]  # (6=num_part, bz, 64=num_query)
        binary_part_logits = torch.max(binary_part_logits, 0)[0] # (bz, 64=num_query)
        losses = {'loss_binary_consistency': F.mse_loss(binary_logits, binary_part_logits)}
        
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        if outputs['pred_sub_boxes'] is None or outputs['pred_obj_boxes'] is None:
            return None
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses
        
    def loss_sub_boxes(self, outputs, targets, indices, num_interactions):
        if outputs['pred_sub_boxes'] is None:
            return None
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
        return losses
    
    def loss_obj_boxes(self, outputs, targets, indices, num_interactions):
        if outputs['pred_obj_boxes'] is None:
            return None
        idx = self._get_src_permutation_idx(indices)
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_obj_boxes.shape[0] == 0:
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def _neg_loss(self, pred, gt, weights=None, alpha=0.25):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        if weights is not None:
            pos_loss = pos_loss * weights[:-1]
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        num_pos  = pos_inds.float().sum()
        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'verb_labels': self.loss_verb_labels,
            'sub_obj_boxes': self.loss_sub_obj_boxes,
            'sub_boxes': self.loss_sub_boxes,
            'obj_boxes': self.loss_obj_boxes,
            'binary_labels': self.loss_binary_labels,
            "binary_consistency": self.loss_binary_consistency,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        obj_indices_final, sub_indices_final, pair_indices_final, outputs_without_aux = self.matcher(outputs_without_aux, targets)
        outputs.update(outputs_without_aux)
        if self.extract:
            output = {'indices': pair_indices_final}
            if 'aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    if None in aux_outputs.values():
                        obj_indices, sub_indices, pair_indices = obj_indices_final, sub_indices_final, pair_indices_final
                    else:
                        obj_indices, sub_indices, pair_indices = self.matcher(aux_outputs, targets)
                    output.update({'indices_{}'.format(i): pair_indices})
            return output
        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if 'obj' in loss:
                losses.update(self.get_loss(loss, outputs, targets, obj_indices_final, num_interactions))
            elif 'sub' in loss:
                losses.update(self.get_loss(loss, outputs, targets, sub_indices_final, num_interactions))
            elif loss == "binary_labels":
                losses.update(self.get_loss(loss, outputs, targets, sub_indices_final, num_interactions))
            else:
                losses.update(self.get_loss(loss, outputs, targets, pair_indices_final, num_interactions))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if None in aux_outputs.values():
                    obj_indices, sub_indices, pair_indices = obj_indices_final, sub_indices_final, pair_indices_final
                else:
                    obj_indices, sub_indices, pair_indices, aux_outputs = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss in ["binary_att", "binary_labels", "binary_6v_labels", "binary_part_cardinality", "binary_consistency", "pvp76_labels"]:
                        continue
                    kwargs = {}
                    if loss == 'obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'obj' in loss: 
                        l_dict = self.get_loss(loss, aux_outputs, targets, obj_indices, num_interactions, **kwargs)
                    elif 'sub' in loss:
                        l_dict = self.get_loss(loss, aux_outputs, targets, sub_indices, num_interactions, **kwargs)
                    else:
                        l_dict = self.get_loss(loss, aux_outputs, targets, pair_indices, num_interactions, **kwargs)
                    if l_dict is not None:
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

        return losses


class PostProcessHOI(nn.Module):

    def __init__(self, subject_category_id, binary=False, pnms=-1, aux_outputs=False, root_hum=3):
        super().__init__()
        self.subject_category_id = subject_category_id
        self.binary = binary
        self.pnms   = pnms
        self.aux_outputs = aux_outputs
        self.num_hum = root_hum ** 2

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['pred_obj_logits'], outputs['pred_verb_logits'], outputs['pred_sub_boxes'], outputs['pred_obj_boxes']
        if self.binary:
            out_bin_logits = outputs['pred_binary_logits']
            if out_bin_logits.shape[-1] == 2:
                binary_scores = F.softmax(out_bin_logits, -1)[..., 1]
            else:
                binary_scores = out_bin_logits

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        verb_scores = out_verb_logits.sigmoid()

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for idx in range(len(obj_scores)):
            os, ol, vs, sb, ob = obj_scores[idx], obj_labels[idx], verb_scores[idx], sub_boxes[idx], obj_boxes[idx]
            
            if os.shape[0] != vs.shape[0]:
                os = os[:, None].expand(-1, self.num_hum).flatten(0, 1)
                ob = ob[:, None, ...].expand(-1, self.num_hum, -1).flatten(0, 1)
                ol = ol[:, None].expand(-1, self.num_hum).flatten(0, 1)
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            vs = vs * os.unsqueeze(1)
            
            if self.binary:
                bs = binary_scores[idx]
                if outputs["pred_part_binary_logits"].shape[-1] == 2:
                    binary_scores_6v = F.softmax(outputs["pred_part_binary_logits"], -1)[..., 1]
                else:
                    binary_scores_6v = outputs["pred_part_binary_logits"]
                bs_6v = binary_scores_6v[:, idx] # (6, num_query)
            #     vs = vs * bs.unsqueeze(1)

            ids = torch.arange(b.shape[0])
            
            if self.pnms > 0:
                sub_iou, _ = box_iou(sb, sb)
                obj_iou, _ = box_iou(ob, ob)
                pair_iou   = torch.min(sub_iou, obj_iou)
                ori_mapping    = {}
                for i in range(vs.shape[0]):
                    if ol[i] not in ori_mapping:
                        ori_mapping[int(ol[i])] = {}
                    ori_mapping[int(ol[i])][i] = 1
                for i in range(vs.shape[1]):
                    mapping = {}
                    mapping.update(ori_mapping)
                    nms_idx = torch.argsort(vs[:, i]).cpu().numpy().tolist()[::-1]
                    for j in nms_idx:
                        if j in mapping[int(ol[j])]:
                            mapping[int(ol[j])].pop(j)
                            if len(mapping[int(ol[j])]) > 0:
                                sel  = torch.Tensor(list(mapping[int(ol[j])].keys()))
                                dump = torch.where(pair_iou[j, sel] > self.pnms)[0]
                                vs[i, sel[dump]] -= 1.
                                for k in dump:
                                    mapping[int(ol[j])].pop(k)

            results[-1].update({'verb_scores': vs.to('cpu'), 'verb_logits': out_verb_logits[idx].to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                'obj_ids': ids[ids.shape[0] // 2:], 'obj_scores': os.to('cpu')})
            if self.binary:
                results[-1].update({'binary_scores': bs.to('cpu')})
                results[-1].update({'binary_scores_6v': bs_6v.to('cpu')})

        if self.aux_outputs:
            for layer in range(len(outputs['aux_outputs'])):
                out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['aux_outputs'][layer]['pred_obj_logits'], outputs['aux_outputs'][layer]['pred_verb_logits'], outputs['aux_outputs'][layer]['pred_sub_boxes'], outputs['aux_outputs'][layer]['pred_obj_boxes']
                if self.binary:
                    out_bin_logits = outputs['aux_outputs'][layer]['pred_binary_logits']
                    if out_bin_logits.shape[-1] == 2:
                        binary_scores = F.softmax(out_bin_logits, -1)[..., 1]
                    else:
                        binary_scores = out_bin_logits

                assert len(out_obj_logits) == len(target_sizes)
                assert target_sizes.shape[1] == 2

                obj_prob = F.softmax(out_obj_logits, -1)
                obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

                verb_scores = out_verb_logits.sigmoid()

                img_h, img_w = target_sizes.unbind(1)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
                sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
                sub_boxes = sub_boxes * scale_fct[:, None, :]
                obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
                obj_boxes = obj_boxes * scale_fct[:, None, :]
                for idx in range(len(obj_scores)):
                    os, ol, vs, sb, ob = obj_scores[idx], obj_labels[idx], verb_scores[idx], sub_boxes[idx], obj_boxes[idx]
                    sl = torch.full_like(ol, self.subject_category_id)
                    l = torch.cat((sl, ol))
                    b = torch.cat((sb, ob))
                    results[idx].update({'labels' + f'_{layer}': l.to('cpu'), 'boxes' + f'_{layer}': b.to('cpu')})
                    
                    vs = vs * os.unsqueeze(1)
                    if self.binary:
                        bs = binary_scores[idx]
                    #     vs = vs * bs.unsqueeze(1)
                    
                    if self.pnms > 0:
                        sub_iou, _ = box_iou(sb, sb)
                        obj_iou, _ = box_iou(ob, ob)
                        pair_iou   = torch.min(sub_iou, obj_iou)
                        ori_mapping    = {}
                        for i in range(vs.shape[0]):
                            if ol[i] not in ori_mapping:
                                ori_mapping[int(ol[i])] = {}
                            ori_mapping[int(ol[i])][i] = 1
                        for i in range(vs.shape[1]):
                            mapping = {}
                            mapping.update(ori_mapping)
                            nms_idx = torch.argsort(vs[:, i]).cpu().numpy().tolist()[::-1]
                            for j in nms_idx:
                                if j in mapping[int(ol[j])]:
                                    mapping[int(ol[j])].pop(j)
                                    if len(mapping[int(ol[j])]) > 0:
                                        sel  = torch.Tensor(list(mapping[int(ol[j])].keys()))
                                        dump = torch.where(pair_iou[j, sel] > self.pnms)[0]
                                        vs[i, sel[dump]] -= 1.
                                        for k in dump:
                                            mapping[int(ol[j])].pop(k)

                    results[idx].update({'verb_scores' + f'_{layer}': vs.to('cpu'), 'obj_scores' + f'_{layer}': os.to('cpu')})
                    if self.binary:
                        results[idx].update({'binary_scores' + f'_{layer}': bs.to('cpu')})

        return results
