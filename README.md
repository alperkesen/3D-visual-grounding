# 3D-visual-grounding

TUM - Advanced Deep Learning for Computer Vision Project (Summer Semester 2021)

Group: Ekrem Alper Kesen, Yoonha Choe

Improve ScanRefer architecture with BRNet object detector, self-attention, cross-modal attention and DGCNN.

## Installation

Check the original [ScanRefer](https://github.com/daveredrum/ScanRefer) model for setup.

## Usage

In order to train the model:
```shell
python scripts/train.py --use_multiview --use_normal --use_brnet --use_dgcnn --use_self_attn --use_cross_attn
```

In order to evaluate the model:
```shell
python scripts/eval.py --folder <folder_name> --reference --no_nms --force --repeat 5 --use_multiview --use_normal --use_brnet --use_dgcnn --use_self_attn --use_cross_attn
```
