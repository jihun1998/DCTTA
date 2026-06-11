# DCTTA
Official repository for ICCV 2025 paper: "**DC-TTA: Divide-and-Conquer Framework for Test-Time Adaptation of Interactive Segmentation**".
This code is implemented based on [ClickSEG repository](https://github.com/XavierCHEN34/ClickSEG).

## Prerequisite
* Tested on Ubuntu 20.04, with Python 3.10.14, PyTorch 2.3.0 with 1 GPUs.
* [CAMO](https://github.com/ltnghia/CAMO) dataset: Download dataset and place under ./datasets folder.
* Please install [SAM](https://github.com/facebookresearch/segment-anything) and download vit_b version as ./sam_checkpoints/sam_vit_b_01ec64.pth

## Training
* For operating DC-TTA on CAMO datasets use following command:
```
CUDA_VISIBLE_DEVICES=0 python run_colseg_sam.py --method final_dctta --tta_lr 1e-5 --datasets CAMO --tta --mm --exp [EXP_NAME]
```

## Citation
If our code be useful for you, please consider citing our ICCV 2025 paper using the following BibTeX entry.
```
@inproceedings{kim2025dc,
  title={DC-TTA: Divide-and-Conquer Framework for Test-Time Adaptation of Interactive Segmentation},
  author={Kim, Jihun and Kwon, Hoyong and Kweon, Hyeokjun and Jeong, Wooseong and Yoon, Kuk-Jin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={23279--23289},
  year={2025}
}
```
