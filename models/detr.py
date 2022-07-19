# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from queue import Queue
import numpy as np
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .hoi import (SetCriterionHOI, PostProcessHOI, DETR_PartMap)
from .transformer import build_transformer


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.obj_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor, depth=None, spmap=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding bx.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.obj_class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, dataset_file, alpha, obj_reweight, use_static_weights, queue_size, p_obj):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

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
            raise
        self.obj_nums_init.append(3 * sum(self.obj_nums_init))  # 3 times fg for bg init
        self.obj_reweight       = obj_reweight
        self.use_static_weights = use_static_weights
        Maxsize = queue_size
        if self.obj_reweight:
            self.q_obj = Queue(maxsize=Maxsize)
            self.p_obj = p_obj
            self.obj_weights_init = self.cal_weights(self.obj_nums_init, p=self.p_obj)

    def cal_weights(self, label_nums, p=0.5):
        num_fgs = len(label_nums[:-1])
        weight = [0] * (num_fgs + 1)
        num_all = sum(label_nums[:-1])

        for index in range(num_fgs):
            if label_nums[index] == 0: continue
            weight[index] = np.power(num_all/label_nums[index], p)

        weight = np.array(weight)
        weight = weight / np.mean(weight[weight>0])

        weight[-1] = np.power(num_all/label_nums[-1], p) if label_nums[-1] != 0 else 0

        weight = torch.FloatTensor(weight).cuda()
        return weight
        
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        if not self.obj_reweight:
            obj_weights = self.empty_weight
        elif self.use_static_weights:
            obj_weights = self.obj_weights_init
        else:
            obj_label_nums_in_batch = [0] * (self.num_classes + 1)
            for target_class in target_classes:
                for label in target_class:
                    obj_label_nums_in_batch[label] += 1

            if self.q_obj.full(): self.q_obj.get()
            self.q_obj.put(np.array(obj_label_nums_in_batch))
            accumulated_obj_label_nums = np.sum(self.q_obj.queue, axis=0)
            obj_weights = self.cal_weights(accumulated_obj_label_nums, p=self.p_obj)

            aphal = min(math.pow(0.999, self.q_obj.qsize()), 0.9)
            obj_weights = aphal * self.obj_weights_init + (1 - aphal) * obj_weights

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_mutual_exclusive(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']
        loss_me   = 0.
        for i in range(src_boxes.shape[0]):
            tmp = box_ops.box_cxcywh_to_xyxy(src_boxes[i])
            s, _ = box_ops.box_iou(tmp, tmp)
            loss_me += s.nansum() - torch.nansum(torch.diag(s))
            
        loss_me /= (src_boxes.shape[0] * src_boxes.shape[1] * src_boxes.shape[1])
        losses = {'loss_me': loss_me}
        return losses
    
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

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'exclusive': self.loss_mutual_exclusive,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(out_logits.device)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s.to('cpu'), 'labels': l.to('cpu'), 'boxes': b.to('cpu')} for s, l, b in zip(scores, labels, boxes)]

        return results


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


def build(args):
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    if args.hoi:
        model = DETR_PartMap(
            backbone,
            transformer,
            num_obj_classes=args.num_obj_classes,
            num_verb_classes=args.num_verb_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            binary=args.binary,
            depth=args.depth,
            depth_cat=args.depth_cat, 
            spmap=args.spmap, 
            root_hum=args.root_hum, 
            ref=args.ref,
            extract=args.extract,
            config = args,
        )
    else:
        model = DETR(
            backbone,
            transformer,
            num_classes=args.num_obj_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
        )
        if args.masks:
            model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {}
    if args.hoi:
        weight_dict['loss_obj_ce']   = args.obj_loss_coef
        weight_dict['loss_verb_ce']  = args.verb_loss_coef
        weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
        weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
        weight_dict['loss_sub_giou'] = args.giou_loss_coef
        weight_dict['loss_obj_giou'] = args.giou_loss_coef
        if args.binary:
            weight_dict['loss_binary_ce'] = args.bin_loss_coef
            for part_id in range(6):
                weight_dict['loss_binary_v%d_ce'%part_id] = 0.1 * args.bin_loss_coef
            weight_dict["loss_binary_consistency"] = args.bin_loss_coef
            weight_dict["loss_pvp76_labels"] = 0.2 * args.bin_loss_coef
    else:
        weight_dict['loss_ce']   = 1
        weight_dict['loss_bbox'] = args.bbox_loss_coef
        weight_dict['loss_giou'] = args.giou_loss_coef
        weight_dict['loss_me']   = args.excl_loss_coef
        if args.masks:
            weight_dict["loss_mask"] = args.mask_loss_coef
            weight_dict["loss_dice"] = args.dice_loss_coef
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.hoi:
        losses = args.losses
        assert len(losses) >= 4
        criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses, verb_loss_type=args.verb_loss_type, dataset_file=args.dataset_file, alpha=args.alpha, obj_reweight=args.obj_reweight, verb_reweight=args.verb_reweight, use_static_weights=args.use_static_weights, queue_size=args.queue_size, p_obj=args.p_obj, p_verb=args.p_verb, extract=args.extract, binary_loss_type = args.binary_loss_type)
    else:
        losses = ['labels', 'boxes', 'cardinality', 'exclusive']
        if args.masks:
            losses += ["masks"]
        criterion = SetCriterion(args.num_obj_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses, dataset_file=args.dataset_file, alpha=args.alpha, obj_reweight=args.obj_reweight, use_static_weights=args.use_static_weights, queue_size=args.queue_size, p_obj=args.p_obj)
    criterion.to(device)
    if args.hoi:
        postprocessors = {'hoi': PostProcessHOI(args.subject_category_id, args.binary, args.pnms, args.aux_outputs)}
    else:
        postprocessors = {'bbox': PostProcess()}
        if args.masks:
            postprocessors['segm'] = PostProcessSegm()
            if args.dataset_file == "coco_panoptic":
                is_thing_map = {i: i <= 90 for i in range(201)}
                postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
