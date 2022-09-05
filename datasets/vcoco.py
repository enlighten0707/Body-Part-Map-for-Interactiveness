# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from pathlib import Path
from PIL import Image
import json
import numpy as np
import pickle

import torch
import torch.utils.data
import torchvision

import datasets.transforms as T

class VCOCO(torch.utils.data.Dataset):

    def __init__(self, img_set, img_folder, anno_file, transforms, num_queries, use_keypoint=True):
        self.img_set = img_set
        self.img_folder = img_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms

        self.num_queries = num_queries

        self._valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                               82, 84, 85, 86, 87, 88, 89, 90)
        self._valid_verb_ids = range(29)

        self.use_keypoint = use_keypoint
        self.part_bboxes_annotations = pickle.load(open("./data/share_label_10w.pkl", "rb"))
        # self.labels_6v_dict = pickle.load(open("./data/labels_6v_vcoco.pkl", "rb"))
        
        self.part6_to_10v = {
            "foot": [0, 3],
            "leg": [1, 2],
            "hip" : [4],
            "hand": [6, 9],
            "arm": [7, 8], 
            "head": [5],
        }
        self.v10_to_part6 = {}
        for part, values in self.part6_to_10v.items():
            for value in values:
                self.v10_to_part6[value] = part
        
        if self.img_set == "train":
            self.single_ids=[]
            self.multi_ids=[]
            for idx, img_anno in enumerate(self.annotations):
                categorys=[obj['category_id'] for obj in img_anno['annotations']]
                num_sub=categorys.count(1)
                if num_sub==1:
                    self.single_ids.append(idx)
                else:
                    self.multi_ids.append(idx)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_anno = self.annotations[idx]

        img = Image.open(self.img_folder / img_anno['file_name']).convert('RGB')
        w, h = img.size

        if self.img_set == 'train' and len(img_anno['annotations']) > self.num_queries:
            img_anno['annotations'] = img_anno['annotations'][:self.num_queries]

        boxes = [obj['bbox'] for obj in img_anno['annotations']]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        if self.img_set == 'train':
            # Add index for confirming which boxes are kept after image transformation
            classes = [(i, self._valid_obj_ids.index(obj['category_id'])) for i, obj in enumerate(img_anno['annotations'])]
        else:
            classes = [self._valid_obj_ids.index(obj['category_id']) for obj in img_anno['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        if self.img_set == 'train':
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]

            target['boxes'] = boxes
            target['labels'] = classes
            target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            if self._transforms is not None:
                img, target = self._transforms(img, target)

            kept_box_indices = [label[0] for label in target['labels']]

            target['labels'] = target['labels'][:, 1]

            obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
            sub_obj_pairs = []
            obj_used, pair2obj = {}, []
            sub_used, pair2sub = {}, []
            labels_6v = []
            for hoi in img_anno['hoi_annotation']:
                if hoi['subject_id'] not in kept_box_indices or \
                   (hoi['object_id'] != -1 and hoi['object_id'] not in kept_box_indices):
                    continue
                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                if sub_obj_pair in sub_obj_pairs:
                    verb_labels[sub_obj_pairs.index(sub_obj_pair)][self._valid_verb_ids.index(hoi['category_id'])] = 1
                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                    verb_label[self._valid_verb_ids.index(hoi['category_id'])] = 1
                    sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                    if hoi['object_id'] == -1:
                        obj_box = torch.zeros((4,), dtype=torch.float32)
                    else:
                        obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                    verb_labels.append(verb_label)
                    if hoi['object_id'] not in obj_used:
                        obj_used[hoi['object_id']] = len(obj_boxes)
                        obj_boxes.append(obj_box)
                        if hoi['object_id'] == -1:
                            obj_labels.append(torch.tensor(len(self._valid_obj_ids)))
                        else:
                            obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])
                    if hoi['subject_id'] not in sub_used:
                        sub_used[hoi['subject_id']] = len(sub_boxes)
                        sub_boxes.append(sub_box)
                    pair2obj.append(obj_used[hoi['object_id']])
                    pair2sub.append(sub_used[hoi['subject_id']])
                    # if img_anno['file_name'] in self.labels_6v_dict.keys() and sub_obj_pair in self.labels_6v_dict[img_anno['file_name']].keys():
                    #     labels_6v.append(self.labels_6v_dict[img_anno['file_name']][sub_obj_pair])
                    # else:
                    #     labels_6v.append(np.zeros(6))

            if len(sub_obj_pairs) == 0:
                target['obj_labels']  = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['sub_boxes']   = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes']   = torch.zeros((0, 4), dtype=torch.float32)
                target['pair2obj']    = torch.zeros((0,), dtype=torch.int64)
                target['pair2sub']    = torch.zeros((0,), dtype=torch.int64)
                target['binary_labels'] = torch.zeros((0,), dtype=torch.int64)
                # target['binary_labels_6v']   = torch.zeros((0, 6), dtype=torch.float32)
            else:
                target['obj_labels']  = torch.stack(obj_labels)
                target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                target['sub_boxes']   = torch.stack(sub_boxes)
                target['obj_boxes']   = torch.stack(obj_boxes)
                target['pair2obj']    = torch.as_tensor(pair2obj, dtype=torch.int64)
                target['pair2sub']    = torch.as_tensor(pair2sub, dtype=torch.int64)
                target['binary_labels'] = torch.ones(target['verb_labels'].shape[0], dtype=torch.int64)
                # target['binary_labels_6v']   = torch.as_tensor(labels_6v, dtype=torch.float32)
        else:
            if self._transforms is not None:
                img, target = self._transforms(img, target)
            
            target['boxes'] = boxes
            target['labels'] = classes
            target['id'] = idx
            target['img_id'] = int(img_anno['file_name'].rstrip('.jpg').split('_')[2])

            hois = []
            for hoi in img_anno['hoi_annotation']:
                hois.append((hoi['subject_id'], hoi['object_id'], self._valid_verb_ids.index(hoi['category_id'])))
            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)

        if self.use_keypoint:
            mask = {}
            if img_anno['file_name'] in self.part_bboxes_annotations.keys():
                for part in self.part6_to_10v.keys():
                    mask[part] = torch.ones((int(target["size"][0] // 32) + 1, int(target["size"][1] // 32) + 1), dtype=bool)
                for item in self.part_bboxes_annotations[img_anno['file_name']]:
                    # 0-Right_foot, 1-Right_leg, 2-Left_leg, 3-Left_foot, 4-Hip, 5-Head, 6-Right_hand, 7-Right_arm, 8-Left_arm, 9-Left_hand.
                    # part_bboxes = item[-1]["part_bbox"] # len=10
                    part_bboxes = item[5]
                    for i in range(10):
                        cur_part = self.v10_to_part6[i]
                        hh_min = int((part_bboxes[i]["y1"] * target["size"][0] / target["orig_size"][0]) // 32)
                        hh_max = int((part_bboxes[i]["y2"] * target["size"][0] / target["orig_size"][0]) // 32)
                        ww_min = int((part_bboxes[i]["x1"] * target["size"][1] / target["orig_size"][1]) // 32)
                        ww_max = int((part_bboxes[i]["x2"] * target["size"][1] / target["orig_size"][1]) // 32)
                        hh_min = max(hh_min, 0)
                        hh_max = min(hh_max, int(target["size"][0] // 32))
                        ww_min = max(ww_min, 0)
                        ww_max = min(ww_max, int(target["size"][1] // 32))
                        for xx in range(hh_min, hh_max + 1):
                            for yy in range(ww_min, ww_max + 1):
                                mask[cur_part][xx, yy] = False # False for effective patch
            else:
                for part in self.part6_to_10v.keys():
                    mask[part] = torch.zeros((int(target["size"][0] // 32) + 1, int(target["size"][1] // 32) + 1), dtype=bool)
            target["mask_part"] = torch.stack([mask[part] for part in self.part6_to_10v.keys()], 0) # (6, H/32, W/32)
            # print(torch.min(target["mask_part"].flatten(1, 2), -1)[0]

        return img, target

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)


# Add color jitter to coco transforms
def make_vcoco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.hoi_path)
    assert root.exists(), f'provided HOI path {root} does not exist'
    PATHS = {
        'train': (root / 'images' / 'train2014', root / 'annotations' / 'trainval_vcoco.json'),
        'val': (root / 'images' / 'val2014', root / 'annotations' / 'test_vcoco.json')
    }
    CORRECT_MAT_PATH = root / 'annotations' / 'corre_vcoco.npy'

    img_folder, anno_file = PATHS[image_set]
    dataset = VCOCO(image_set, img_folder, anno_file, transforms=make_vcoco_transforms(image_set),
                    num_queries=args.num_queries)
    if image_set == 'val':
        dataset.load_correct_mat(CORRECT_MAT_PATH)
    return dataset
