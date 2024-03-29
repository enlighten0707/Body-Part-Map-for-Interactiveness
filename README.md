# Body-Part Map for Interactiveness
This repo contains the official implementation of our paper:

**Mining Cross-Person Cues for Body-Part Interactiveness Learning in HOI Detection (ECCV 2022)**

Xiaoqian Wu*, Yong-Lu Li*, Xinpeng Liu, Junyi Zhang, Yuzhe Wu, and Cewu Lu

[[Paper](https://arxiv.org/pdf/2207.14192v2.pdf)] 

In this paper, we focus on learning human body-part interactiveness from a previously overlooked *global* perspective. We construct **body-part saliency maps** to mine informative cues from not only the targeted person, but also *other persons* in the image.

Note: Our method *does not* depend on extra supervision. The main model of our method is trained **without extra PaSta labels**.

![](./assets/intro.jpg)

## Dependencies
```
python==3.9
pytorch==1.9
torchvision==0.10.1
```
## Data preparation
For HICO-DET&V-COCO, download the pre-calculated pose keypoint files [here](https://drive.google.com/drive/folders/16fYJ5trvMzA6ZjHIJVHcPkgTLbiZSMtl?usp=sharing), and put them into `data` folder. They are used for body-part saliency map calculation.

HICO-DET dataset can be downloaded [here](https://drive.google.com/file/d/1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk/view). After finishing downloading, unpack `hico_20160224_det.tar.gz` into `data` folder. We use the annotation files provided by the PPDM authors. The annotation files can be downloaded from [here](https://drive.google.com/drive/folders/1WI-gsNLS-t0Kh8TVki1wXqc3y2Ow1f2R).

For training, download the COCO pre-trained DETR [here](https://drive.google.com/drive/folders/16fYJ5trvMzA6ZjHIJVHcPkgTLbiZSMtl?usp=sharing) and put it into `params` folder.

## Training
```
python -m torch.distributed.launch --nproc_per_node=4 main.py --config_path configs/interactiveness_train_hico_det.yml
```
## Evaluation
```
python -m torch.distributed.launch --nproc_per_node=4 main.py --config_path configs/interactiveness_eval_hico_det.yml
```
## Results
The result file can be downloaded from [here](https://drive.google.com/drive/folders/1UydzhAbgsUG4jHK27Oi8m3vOF8pTtjz4?usp=share_link).
Then replace `exp` folder with the downloaded dir, and run [notebooks/eval.ipynb](./notebooks/eval.ipynb) for final interactiveness/HOI mAP. 

## Visualization of Attention Results
First extract attention weights
```
python -m torch.distributed.launch --nproc_per_node=4 main.py --config_path configs/interactiveness_train_hico_det.yml --extract
```
Then run [notebooks/att.ipynb](./notebooks/att.ipynb).

## Citation
```
@inproceedings{wu2022mining,
  title={Mining Cross-Person Cues for Body-Part Interactiveness Learning in HOI Detection},
  author={Wu, Xiaoqian and Li, Yong-Lu and Liu, Xinpeng and Zhang, Junyi and Wu, Yuzhe and Lu, Cewu},
  booktitle={ECCV},
  year={2022}
}
```
