
# [Dual-Student-Mean-Teacher-Medical-Image-Segmentation](https://github.com/tyrion1999/Dual-Student-Mean-Teacher-Medical-Image-Segmentation)
by Yang Zuo, Xiurui Guo, Chunmeng Kang, Guohui Cai, Xiang Liu, Lei Lyu

### Introduction
This repository is for our paper: Dual-Student-Mean-Teacher-Medical-Image-Segmentation

### Requirements
This repository is based on PyTorch 1.8.0, CUDA 11.1 and Python 3.8.0; All experiments in our paper were conducted on a single NVIDIA GeForce RTX 3090 GPU.

### Usage
1. Train the model;

```
# e.g., for 10% labels on LA
python ./code/train.py 
--root_path '/home/stu/zy/data/LA' 
--model1 'vnet_student1' 
--model2 'vnet_student2' 
--batch_size 4 
--base_lr 0.01 
--patch_size [112,112,80]
--seed 1337
--gpu 0
--image_size [80,112,112]

--labeled_bs 2
--labeled_num 8
--total_sample 80

--ema_decay 0.99
--consistency_type mse
--consistency 0.1
--consistency_rampup 40.0


```
2. Test the model;

```
# e.g., for 20% labels on LA
python ./code/test_3d.py 
--root_path 'home/stu/zy/data/LA' 
--model 'vnet' 
--gpu 0Test the biou;
```


### Acknowledgements:
Our code is origin from [MC-Net](https://github.com/ycwu1997/MC-Net), [UAMT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [DTC](https://github.com/HiLab-git/DTC), [URPC](https://github.com/HiLab-git/SSL4MIS) and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.