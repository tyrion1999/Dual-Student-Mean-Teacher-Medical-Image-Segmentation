# [Dual-Student-Mean-Teacher-Medical-Image-Segmentation](https://github.com/tyrion1999/Dual-Student-Mean-Teacher-Medical-Image-Segmentation)
by Yang Zuo, Xiurui Guo, Chunmeng Kang, Guohui Cai, Xiang Liu, Lei Lyu

### Introduction
This repository is for our paper: Prototype-Denoising Dual Student Mean Teacher for Voxel-wise Medical Image Segmentation

### Requirements
This repository is based on:
- **PyTorch 1.8.0**
- **CUDA 11.1**
- **Python 3.8.0**

All experiments in our paper were conducted on a single NVIDIA GeForce RTX 3090 GPU.

### Datasets:
You can download the datasets used in our experiments from the following links:

- **Left Atrium dataset:**
  - URL: https://pan.baidu.com/s/16A5GrrmvkJ1Z_3kzaitoTw 
  - CODE: 1234

- **Pancreas-CT dataset:**
  - URL: https://pan.baidu.com/s/1GEQ2gbPULW2vQwrRMPFXbg 
  - CODE: 1234

- **ACDC dataset:**
  - URL: https://pan.baidu.com/s/1jziQRdTk2cjQBX4K2pCjXA 
  - CODE: 1234


### Usage
1. Train the model;
You can train the model with the following command. For example, to use 10% labeled data on the Left Atrium dataset:
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
To test the model, use the following command. For example, for 20% labeled data on the Left Atrium dataset:

```
# e.g., for 20% labels on LA
python ./code/test_3d.py 
--root_path 'home/stu/zy/data/LA' 
--model 'vnet' 
--gpu 0Test the biou;
```
<!--
### Citation

```
If you find our work useful or inspiring for your research, please consider citing our paper:
@article{YourPaper2024,
  title={Dual-Student-Mean-Teacher-Medical-Image-Segmentation},
  author={Yang Zuo and Xiurui Guo and Chunmeng Kang and Guohui Cai and Xiang Liu and Lei Lyu},
  journal={Journal Name},
  year={2024},
  volume={XX},
  pages={XXX--XXX},
  doi={DOI_NUMBER}
}
```
-->
### Acknowledgements:
Our code is origin from [MC-Net](https://github.com/ycwu1997/MC-Net), [UAMT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [DTC](https://github.com/HiLab-git/DTC), [URPC](https://github.com/HiLab-git/SSL4MIS) and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.
