# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch 
from torch.utils.data import DataLoader, DistributedSampler
import os, yaml, re, pickle

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import train_one_epoch, evaluate_hoi
from models import build_model
from binary_evaluation import calc_binary as calc_binary_hico
from binary_evaluation_vcoco import calc_binary as calc_binary_vcoco


os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# export CUDA_VISIBLE_DEVICES=0,1,2,3; python -m torch.distributed.launch --nproc_per_node=4 main.py --config_path configs/qpic_res50.yml
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_path', type=str, default='configs/qpic_res50.yml',
                        help="Path of the configuration")
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    # xinpeng added
    parser.add_argument('--binary', dest='binary', action='store_true',
                        help="Binary interactiveness detection")
    parser.add_argument('--cascaded', dest='cascaded', action='store_true',
                        help="Cascaded decoder")
    parser.add_argument('--triple_cascaded', dest='triple_cascaded', action='store_true',
                        help="Triple Cascaded decoder")
    parser.add_argument('--dec_layers_hum', default=3, type=int,
                        help="Number of human decoding layers in the cascaded transformer")
    parser.add_argument('--dec_layers_box', default=3, type=int,
                        help="Number of box decoding layers in the cascaded transformer")
    parser.add_argument('--dec_layers_verb', default=3, type=int,
                        help="Number of verb decoding layers in the cascaded transformer")
                        

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # HOI
    parser.add_argument('--hoi', action='store_true',
                        help="Train for HOI if the flag is provided")
    parser.add_argument('--num_obj_classes', type=int, default=80,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help="Number of verb classes")
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--subject_category_id', default=0, type=int)
    parser.add_argument('--verb_loss_type', type=str, default='focal',
                        help='Loss type for the verb classification')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_obj_class', default=1, type=float,
                        help="Object class coefficient in the matching cost")
    parser.add_argument('--set_cost_verb_class', default=1, type=float,
                        help="Verb class coefficient in the matching cost")
    # xinpeng added
    parser.add_argument('--set_cost_binary', default=1, type=float,
                        help="Binary class coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--obj_loss_coef', default=1, type=float)
    parser.add_argument('--verb_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    # xinpeng added
    parser.add_argument('--bin_loss_coef', default=1, type=float)
    parser.add_argument('--excl_loss_coef', default=1, type=float)
    parser.add_argument('--alpha', default=0.5, type=float, help='focal loss alpha')
    parser.add_argument('--obj_reweight', action='store_true')
    parser.add_argument('--verb_reweight', action='store_true')
    parser.add_argument('--use_static_weights', action='store_true', 
                        help='use static weights or dynamic weights, default use dynamic')
    parser.add_argument('--queue_size', default=4704, type=float,
                        help='Maxsize of queue for obj and verb reweighting, default 1 epoch')
    parser.add_argument('--p_obj', default=0.7, type=float,
                        help='Reweighting parameter for obj')
    parser.add_argument('--p_verb', default=0.7, type=float,
                        help='Reweighting parameter for verb')
    parser.add_argument('--cascaded_matcher', action='store_true',
                        help='Whether to use cascaded matcher')
    parser.add_argument('--ref', action='store_true',
                        help='Whether to refine object detection')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--hoi_path', type=str)
    parser.add_argument('--label_nms', default=-1, type=float, help="Label NMS threshold, -1 for no NMS")
    parser.add_argument('--pnms', default=-1, type=float, help="Pair NMS threshold, -1 for no NMS")

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    # xinpeng added
    parser.add_argument('--depth', action='store_true')
    parser.add_argument('--depth_cat', default=100, type=int)
    parser.add_argument('--spmap', action='store_true')
    parser.add_argument('--root_hum', default=3, type=int)
    parser.add_argument('--aux_outputs', action='store_true')
    parser.add_argument('--freeze_mode', default=0, type=int)
    parser.add_argument('--extract', action='store_true')

    # xiaoqian added
    parser.add_argument('--binary_mode', default=None, type=str)
    parser.add_argument('--losses', default=None, type=list)
    parser.add_argument('--binary_nms', default=False, type=bool)
    parser.add_argument('--binary_loss_type', default="bce", type=str)
    parser.add_argument('--use_hake', default=False, type=bool)
    parser.add_argument('--hard_mask', default=False, type=bool)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser

def make_options(args):
    loader = yaml.FullLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    config = yaml.load(open(args.config_path), Loader=loader)
    dic = vars(args)
    all(map( dic.pop, config))
    dic.update(config)
    return args

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)
    
    device = torch.device(args.device)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        if args.freeze_mode:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])], output_device=int(os.environ["LOCAL_RANK"]), find_unused_parameters=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])], output_device=int(os.environ["LOCAL_RANK"]))
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    print("freeze_mode", args.freeze_mode)
    if args.freeze_mode == 1:
        for name, p in model.named_parameters():
            if 'decoder' not in name and 'verb_class_embed' not in name and 'obj_class_embed' not in name and 'sub_bbox_embed' not in name and 'obj_bbox_embed' not in name and 'binary_class_embed' not in name:
                p.requires_grad = False
    elif args.freeze_mode == 2:
        for name, p in model.named_parameters():
            if 'verb_class_embed' not in name:
                p.requires_grad = False
    elif args.freeze_mode == 3:
        for name, p in model.named_parameters():
            if 'verb_decoder' not in name and 'verb_class_embed' not in name:
                p.requires_grad = False
    elif args.freeze_mode == 4:
        print([name for name, p in model.named_parameters() if 'verb' in name])
        for name, p in model.named_parameters():
            if 'verb' not in name:
                p.requires_grad = False
    elif args.freeze_mode == 5:
        for name, p in model.named_parameters():
            if 'backbone' in name:
                p.requires_grad = False
    elif args.freeze_mode == 6:
        for name, p in model.named_parameters():
            if 'binary_class_embed' not in name:
                p.requires_grad = False
    
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if not args.hoi:
        if args.dataset_file == "coco_panoptic":
            # We also evaluate AP during panoptic training, on original coco DS
            coco_val = datasets.coco.build("val", args)
            base_ds = get_coco_api_from_dataset(coco_val)
        else:
            base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    bst_state = None
    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        bst_state = checkpoint['model']
        bst_state = {k:v for k,v in bst_state.items() if k in model_without_ddp.state_dict().keys()}
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("missing_keys", missing_keys)
        print("unexpected_keys", unexpected_keys)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    elif args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        if args.num_queries < 100:
            checkpoint['model']['query_embed.weight'] = checkpoint['model']['query_embed.weight'][:args.num_queries]
        remapping = []
        for key in checkpoint['model'].keys():
            if "transformer.decoder" in key:
                remapping.append((key, key.replace("transformer.decoder", "transformer.box_decoder")))
            if "transformer.interaction_decoder" in key:
                remapping.append((key, key.replace("transformer.interaction_decoder", "transformer.verb_decoder")))
            if ".bbox_embed" in key:
                remapping.append((key, key.replace("transformer.bbox_embed", "transformer.obj_bbox_embed")))
        for k1, k2 in remapping:
            checkpoint['model'][k2] = checkpoint['model'].pop(k1)
        bst_state = checkpoint['model']
        bst_state = {k:v for k,v in bst_state.items() if k in model_without_ddp.state_dict().keys()}
        if args.freeze_mode == 3:
            bst_state = {k:v for k,v in bst_state.items() if 'verb_class_embed' not in k}
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(bst_state, strict=False)
        print("missing_keys", missing_keys)
        print("unexpected_keys", unexpected_keys)
    
    if args.extract:
        # for epoch in range(args.epochs):
        # extract_stats = extract_feature(model, criterion, data_loader_train, device, 0, args.output_dir)
        extract_stats = extract_feature(model, criterion, data_loader_val, device, 0, args.output_dir)
        return

    if args.eval:
        if args.hoi:
            test_stats, preds = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val, args.subject_category_id, device)
            if 'AP_full' in test_stats:
                ap_full = test_stats.pop('AP_full')
            if args.output_dir:
                utils.save_on_master({'stats': test_stats, 'preds': preds}, output_dir / "result.pth")
            if args.binary and utils.is_main_process():
                if args.dataset_file == 'hico':
                    idx, res = preds
                    bin_preds = {'keys': [], 'scores': [], 'bboxes': [], \
                        'scores_v0':[], 'scores_v1':[], 'scores_v2':[], 'scores_v3':[], 'scores_v4':[], 'scores_v5':[]}
                    anno  = json.load(open('data/hico_20160224_det/annotations/test_hico.json'))
                    for i in range(len(res)):
                        pnum = res[i]['verb_scores'].shape[0]
                        bin_preds['keys'].append(np.array([int(anno[idx[i]["id"]]['file_name'][-10:-4])] * pnum))
                        bin_preds['bboxes'].append(np.concatenate([res[i]['boxes'][np.arange(pnum)], res[i]['boxes'][np.arange(pnum, pnum * 2)]], axis=1))
                        bin_preds['scores'].append(res[i]['binary_scores'])
                        for k in range(6):
                            bin_preds['scores_v%d'%k].append(res[i]['binary_scores_6v'][k])
                    bin_preds['scores'] = np.concatenate(bin_preds['scores'])
                    bin_preds['keys'] = np.concatenate(bin_preds['keys'])
                    bin_preds['bboxes'] = np.concatenate(bin_preds['bboxes'])
                    for k in range(6):
                        bin_preds['scores_v%d'%k] = np.concatenate(bin_preds['scores_v%d'%k])
                    bin_ap, bin_rec = calc_binary_hico(bin_preds['keys'], bin_preds['bboxes'], bin_preds['scores'])
                    print("binary ap:", 100 * bin_ap)
                    pickle.dump(bin_preds, open(os.path.join(output_dir, "bst_result_binary.pkl"), "wb"))
                elif args.dataset_file == 'vcoco':
                    idx, res = preds
                    bin_preds = {'keys': [], 'scores': [], 'bboxes': [], 'obj_labels': [], 'obj_scores':[], \
                        'scores_v0':[], 'scores_v1':[], 'scores_v2':[], 'scores_v3':[], 'scores_v4':[], 'scores_v5':[]}

                    for i in range(len(res)):
                        pnum = res[i]['verb_scores'].shape[0]
                        bin_preds['keys'].append(np.array([idx[i]['img_id']] * pnum))
                        bin_preds['bboxes'].append(np.concatenate([res[i]['boxes'][np.arange(pnum)], res[i]['boxes'][np.arange(pnum, pnum * 2)]], axis=1))
                        bin_preds['scores'].append(res[i]['binary_scores'].numpy())
                        bin_preds['obj_labels'].append((res[i]["labels"][pnum:]).numpy())
                        bin_preds['obj_scores'].append((res[i]["obj_scores"]).numpy())

                    bin_preds['scores'] = np.concatenate(bin_preds['scores'])
                    bin_preds['keys'] = np.concatenate(bin_preds['keys'])
                    bin_preds['bboxes'] = np.concatenate(bin_preds['bboxes'])
                    bin_preds['obj_labels'] = np.concatenate(bin_preds['obj_labels'])
                    bin_preds['obj_scores'] = np.concatenate(bin_preds['obj_scores'])

                    idx = np.where(bin_preds['obj_labels']<80)[0]
                    bin_ap, bin_rec = calc_binary_vcoco(bin_preds['keys'][idx], bin_preds['bboxes'][idx], bin_preds['scores'][idx])
                    
                    print("binary ap:", 100 * bin_ap)
                    pickle.dump(bin_preds, open(os.path.join(output_dir, "bst_result_binary.pkl"), "wb"))

            return
        elif args.dataset_file == 'hico':
            test_stats, preds = evaluate_box(args.dataset_file, model, postprocessors, data_loader_val, args.subject_category_id, device)
            if args.output_dir:
                utils.save_on_master({'stats': test_stats, 'preds': preds}, output_dir / "result.pth")
            return
        else:
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                                  data_loader_val, base_ds, device, args.output_dir)
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            return
    
    bst_ap_verb_wise = None
    bst_ap_full = None
    if args.freeze_mode == 2:
        test_stats, _ = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val, args.subject_category_id, device)
        if args.dataset_file == 'hico':
            bst_ap_full = test_stats.pop('AP_full')
            bst_ap_verb_wise = {}
            for (h, o, v) in bst_ap_full.keys():
                if v not in bst_ap_verb_wise:
                    bst_ap_verb_wise[v] = []
                bst_ap_verb_wise[v].append(bst_ap_full[(h, o, v)])
            for v in bst_ap_verb_wise.keys():
                bst_ap_verb_wise[v] = np.mean(bst_ap_verb_wise[v])
        elif args.dataset_file == 'vcoco':
            bst_ap_verb_wise = test_stats
            bst_ap_verb_wise.pop('mAP_all')
            bst_ap_verb_wise.pop('mAP_thesis')
    
    print("Start training")
    bst = 0.0
    bst_binary = 0.0
    start_time = time.time()
    # print(args.start_epoch, args.epochs)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.output_dir,
            args.clip_max_norm,
            )
        lr_scheduler.step()
        
        test_stats = {}
        ap_full = None
        if args.hoi:
            test_stats, preds = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val, args.subject_category_id, device)
            coco_evaluator = None
            if args.dataset_file == 'hico':
                ap_full = test_stats.pop('AP_full')
            if args.binary:
                if args.dataset_file == 'hico':
                    idx, res = preds
                    bin_preds = {'keys': [], 'scores': [], 'bboxes': [], \
                        'scores_v0':[], 'scores_v1':[], 'scores_v2':[], 'scores_v3':[], 'scores_v4':[], 'scores_v5':[]}
                    anno  = json.load(open('data/hico_20160224_det/annotations/test_hico.json'))
                    for i in range(len(res)):
                        pnum = res[i]['verb_scores'].shape[0]
                        bin_preds['keys'].append(np.array([int(anno[idx[i]["id"]]['file_name'][-10:-4])] * pnum))
                        bin_preds['bboxes'].append(np.concatenate([res[i]['boxes'][np.arange(pnum)], res[i]['boxes'][np.arange(pnum, pnum * 2)]], axis=1))
                        bin_preds['scores'].append(res[i]['binary_scores'])
                        for k in range(6):
                            bin_preds['scores_v%d'%k].append(res[i]['binary_scores_6v'][k])
                    bin_preds['scores'] = np.concatenate(bin_preds['scores'])
                    bin_preds['keys'] = np.concatenate(bin_preds['keys'])
                    bin_preds['bboxes'] = np.concatenate(bin_preds['bboxes'])
                    for k in range(6):
                        bin_preds['scores_v%d'%k] = np.concatenate(bin_preds['scores_v%d'%k])
                    bin_ap, bin_rec = calc_binary_hico(bin_preds['keys'], bin_preds['bboxes'], bin_preds['scores'])
                elif args.dataset_file == 'vcoco':
                    idx, res = preds
                    bin_preds = {'keys': [], 'scores': [], 'bboxes': [], 'obj_labels': [], \
                        'scores_v0':[], 'scores_v1':[], 'scores_v2':[], 'scores_v3':[], 'scores_v4':[], 'scores_v5':[]}

                    for i in range(len(res)):
                        pnum = res[i]['verb_scores'].shape[0]
                        bin_preds['keys'].append(np.array([idx[i]['img_id']] * pnum))
                        bin_preds['bboxes'].append(np.concatenate([res[i]['boxes'][np.arange(pnum)], res[i]['boxes'][np.arange(pnum, pnum * 2)]], axis=1))
                        bin_preds['scores'].append(res[i]['binary_scores'].numpy())
                        bin_preds['obj_labels'].append((res[i]["labels"][pnum:]).numpy())

                    bin_preds['scores'] = np.concatenate(bin_preds['scores'])
                    bin_preds['keys'] = np.concatenate(bin_preds['keys'])
                    bin_preds['bboxes'] = np.concatenate(bin_preds['bboxes'])
                    bin_preds['obj_labels'] = np.concatenate(bin_preds['obj_labels'])

                    idx = np.where(bin_preds['obj_labels']<80)[0]
                    bin_ap, bin_rec = calc_binary_vcoco(bin_preds['keys'][idx], bin_preds['bboxes'][idx], bin_preds['scores'][idx])
                    
                if utils.is_main_process():
                    test_stats["mAP_binary"] = bin_ap
                    print("binary ap:", 100 * bin_ap)
        elif args.dataset_file == 'hico':
            test_stats, _ = evaluate_box(args.dataset_file, model, postprocessors, data_loader_val, args.subject_category_id, device)
            coco_evaluator = None
        else:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            )
        
        if args.freeze_mode == 2:
            cur_state = model_without_ddp.state_dict()
            ap_verb_wise = {}
            if args.dataset_file == 'hico':
                for (h, o, v) in ap_full.keys():
                    if v not in ap_verb_wise:
                        ap_verb_wise[v] = []
                    ap_verb_wise[v].append(ap_full[(h, o, v)])
                for v in ap_verb_wise.keys():
                    ap_verb_wise[v] = np.mean(ap_verb_wise[v])
                    if ap_verb_wise[v] > bst_ap_verb_wise[v]:
                        bst_ap_verb_wise[v] = ap_verb_wise[v]
                        bst_state['verb_class_embed.weight'][v, ...] = cur_state['verb_class_embed.weight'][v, ...]
                        bst_state['verb_class_embed.bias'][v]        = cur_state['verb_class_embed.bias'][v]
                        for k in bst_ap_full.keys():
                            if v == k[2]:
                                bst_ap_full[k] = bst_ap_verb_wise[v]
                test_stats['mAP'] = np.mean(list(bst_ap_full.values()))
            else:
                ap_verb_wise = test_stats
                ap_verb_wise.pop('mAP_all')
                ap_verb_wise.pop('mAP_thesis')
                verb_classes = ['hold_obj', 'stand', 'sit_instr', 'ride_instr', 'walk', 'look_obj', 'hit_instr', 'hit_obj',
                                'eat_obj', 'eat_instr', 'jump_instr', 'lay_instr', 'talk_on_phone_instr', 'carry_obj',
                                'throw_obj', 'catch_obj', 'cut_instr', 'cut_obj', 'run', 'work_on_computer_instr',
                                'ski_instr', 'surf_instr', 'skateboard_instr', 'smile', 'drink_instr', 'kick_obj',
                                'point_instr', 'read_obj', 'snowboard_instr']
                thesis_map_indices = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 24, 25, 27, 28]
                for i, v in enumerate(verb_classes):
                    if test_stats['AP_{}'.format(v)] > bst_ap_verb_wise['AP_{}'.format(v)]:
                        bst_ap_verb_wise['AP_{}'.format(v)] = ap_verb_wise['AP_{}'.format(v)]
                        bst_state['verb_class_embed.weight'][i, ...] = cur_state['verb_class_embed.weight'][i, ...]
                        bst_state['verb_class_embed.bias'][i]        = cur_state['verb_class_embed.bias'][i]
                test_stats['mAP_all']    = np.mean(list(bst_ap_verb_wise.values()))
                test_stats['mAP_thesis'] = np.mean([bst_ap_verb_wise['AP_{}'.format(v)] for cid, v in enumerate(verb_classes) if cid in thesis_map_indices])
                
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            if 'mAP' in test_stats and test_stats['mAP'] > bst:
                checkpoint_paths.append(output_dir / f'checkpoint_bst.pth')
                bst = test_stats['mAP']
            if 'mAP_thesis' in test_stats and test_stats['mAP_thesis'] > bst:
                checkpoint_paths.append(output_dir / f'checkpoint_bst.pth')
                bst = test_stats['mAP_thesis']
            if 'mAP_binary' in test_stats and test_stats['mAP_binary'] > bst_binary:
                checkpoint_paths.append(output_dir / f'checkpoint_bst_binary.pth')
                bst_binary = test_stats['mAP_binary']
                if utils.is_main_process():
                    pickle.dump(bin_preds, open(os.path.join(output_dir, "bst_result_binary.pkl"), "wb"))
            for checkpoint_path in checkpoint_paths:
                if args.freeze_mode == 2:
                    utils.save_on_master({
                        'model': bst_state,
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
                else:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = make_options(parser.parse_args())
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        f = open(os.path.join(args.output_dir, "config.yaml"), "w")
        for key, val in vars(args).items():
            f.write("%s : %s\n" %(key, val))
        f.close()
    main(args)
