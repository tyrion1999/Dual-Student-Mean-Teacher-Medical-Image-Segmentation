# [Decoupling Mean Teacher via Dual Students: A Prototype-based Framework for Semi-Supervised Medical Segmentation](https://github.com/tyrion1999/Dual-Student-Mean-Teacher-Medical-Image-Segmentation)

by **Yang Zuo**, **Renfeng Zhang**, **Xiurui Guo**, **Chunmeng Kang**, **Lei Lyu**

---

### Introduction

This repository provides the official implementation of the paper:

> **Decoupling mean teacher via Dual students: a prototype-based framework for semi-supervised medical segmentation**

The paper has been published in *Complex & Intelligent Systems*:

- Journal: *Complex & Intelligent Systems*  
- Volume: **11**, Issue: **8**, Article: **353**, Year: **2025**  
- DOI: **10.1007/s40747-025-01964-z**  
- Online: https://link.springer.com/article/10.1007/s40747-025-01964-z  

Our DS-MT framework introduces:
- A **dual-student mean teacher architecture** with decoupled parameter updates;
- A **prototype-based pseudo-label denoising scheme** that evaluates voxel-wise label reliability via distances to class prototypes;
- A training strategy that jointly improves segmentation performance and robustness under limited labeled data.

---

### Requirements

This repository is based on:

- **Python 3.8.0**
- **PyTorch 1.8.0**
- **CUDA 11.1**

All experiments in our paper were conducted on a single **NVIDIA GeForce RTX 3090** GPU.

---

### Datasets

You can download the datasets used in our experiments from the following links:

- **Left Atrium (LA) dataset**
  - URL: https://pan.baidu.com/s/16A5GrrmvkJ1Z_3kzaitoTw  
  - CODE: `1234`

- **Pancreas-CT dataset**
  - URL: https://pan.baidu.com/s/1GEQ2gbPULW2vQwrRMPFXbg  
  - CODE: `1234`

- **ACDC dataset**
  - URL: https://pan.baidu.com/s/1jziQRdTk2cjQBX4K2pCjXA  
  - CODE: `1234`

---

### Usage

#### 1. Train the model

For example, to train DS-MT with **10% labeled data** on the **Left Atrium** dataset:

```bash
# e.g., for 10% labels on LA
python ./code/train.py \
  --root_path '/home/stu/zy/data/LA' \
  --model1 'vnet_student1' \
  --model2 'vnet_student2' \
  --batch_size 4 \
  --base_lr 0.01 \
  --patch_size [112,112,80] \
  --seed 1337 \
  --gpu 0 \
  --image_size [80,112,112] \
  --labeled_bs 2 \
  --labeled_num 8 \
  --total_sample 80 \
  --ema_decay 0.99 \
  --consistency_type mse \
  --consistency 0.1 \
  --consistency_rampup 40.0
```
### Citation

If you find this repository or our paper useful in your research, please consider citing:

### APA

Zuo, Y., Zhang, R., Guo, X., Kang, C., & Lyu, L. (2025). Decoupling mean teacher via Dual students: a prototype-based framework for semi-supervised medical segmentation. Complex & Intelligent Systems, 11(8), 353. https://doi.org/10.1007/s40747-025-01964-z

### BibTeX
@article{Zuo2025DSMT,
  title   = {Decoupling mean teacher via Dual students: a prototype-based framework for semi-supervised medical segmentation},
  author  = {Zuo, Yang and Zhang, Renfeng and Guo, Xiurui and Kang, Chunmeng and Lyu, Lei},
  journal = {Complex \& Intelligent Systems},
  year    = {2025},
  volume  = {11},
  number  = {8},
  pages   = {353},
  doi     = {10.1007/s40747-025-01964-z}
}
Acknowledgements

Our code is adapted from the following excellent repositories:

MC-Net https://github.com/ycwu1997/MC-Net

UAMT https://github.com/yulequan/UA-MT

SASSNet https://github.com/kleinzcy/SASSnet

DTC https://github.com/HiLab-git/DTC

URPC https://github.com/HiLab-git/SSL4MIS?utm_source=chatgpt.com

SSL4MIS  https://github.com/HiLab-git/SSL4MIS?utm_source=chatgpt.com

We sincerely thank the authors of these works and hope that our DS-MT framework can further promote research in semi-supervised medical image segmentation.
