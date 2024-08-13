# -*- coding: utf-8 -*-
import os
import argparse
import torch
# from networks.vnet_sdf import VNet
# from val_3D import test_all_case
from networks.vnet import VNet, VNet_Student1,VNet_Student2
from networks.vnet import VNet
# from networks.vnet import VNet
from thop import profile

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# net = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).cuda()
net = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).cuda()
# input = torch.randn(1, 1, 112, 112,80).cuda()
input = torch.randn(1, 1, 96, 96,96).cuda()
# 计算 FLOPs
flops, param = profile(net, inputs=(input,))
FLOPs = round(flops / 1e9, 2)
params = round(param / 1e6, 2)
print(params, FLOPs)