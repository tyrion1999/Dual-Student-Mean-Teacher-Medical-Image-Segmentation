import argparse
import logging
import math
import os
import random
import shutil
import sys
import time

import unfoldNd
from torch.nn.functional import one_hot
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm


from dataloaders import utils
from dataloaders.la_heart import (LA_heart, CenterCrop, RandomCrop,RandomRotFlip, ToTensor,TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from val_3D import test_all_case


def save_combined_models(model1, model2, file_path):
    state_dict = {
        'model1': model1.state_dict(),
        'model2': model2.state_dict()
    }
    torch.save(state_dict, file_path)
    print(f"Combined model saved to {file_path}")

# Function to load both models from one file


class DSCLoss(nn.Module):
    def __init__(self, num_classes=2, inter_weight=0.5, intra_weights=None, device='cuda', is_3d=False):
        super(DSCLoss, self).__init__()
        if intra_weights is not None:
            intra_weights = torch.tensor(intra_weights).to(device)
        self.ce_loss = nn.CrossEntropyLoss(weight=intra_weights)
        self.num_classes = num_classes
        self.intra_weights = intra_weights
        self.inter_weight = inter_weight
        self.device = device
        self.is_3d = is_3d

    def dice_loss(self, prediction, target, weights=None):
        """Calculating the dice loss
        Args:
            prediction = predicted image
            target = Targeted image
        Output:
            dice_loss"""
        smooth = 1e-5

        prediction = torch.softmax(prediction, dim=1)
        batchsize = target.size(0)
        num_classes = target.size(1)
        prediction = prediction.view(batchsize, num_classes, -1)
        target = target.view(batchsize, num_classes, -1)

        intersection = (prediction * target)

        dice = (2. * intersection.sum(2) + smooth) / (prediction.sum(2) + target.sum(2) + smooth)
        dice_loss = 1 - dice.sum(0) / batchsize
        if weights is not None:
            weighted_dice_loss = dice_loss * weights
            return weighted_dice_loss.mean()
        return dice_loss.mean()

    def forward(self, pred, label):
        """Calculating the loss and metrics
            Args:
                prediction = predicted image
                target = Targeted image
                metrics = Metrics printed
                bce_weight = 0.5 (default)
            Output:
                loss : dice loss of the epoch """
        # if label.shape[0] != pred.shape[0]:
        #     label = label.unsqueeze(0).repeat(pred.shape[0], 1, 1, 1, 1)

        # label = label.float()
        cel = self.ce_loss(pred, label)
        if self.is_3d:
            label_onehot = one_hot(label, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).contiguous()
        else:
            label_onehot = one_hot(label, num_classes=self.num_classes).permute(0, 3, 1, 2).contiguous()
        dicel = self.dice_loss(pred, label_onehot, self.intra_weights)
        loss = cel * (1 - self.inter_weight) + dicel * self.inter_weight
        return loss


parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=2, help='number of target categories')
parser.add_argument('--intra_weights', type=list, default=[1., 1.], help='inter classes weighted coefficients in the loss function')
parser.add_argument('--root_path', type=str, default='/root/autodl-tmp/data/LA', help='Name of Experiment')
#parser.add_argument('--root_path', type=str, default='/home/stu/zy/data/LA', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='LA_ACMT_改阈值_0.1', help='experiment_name')
parser.add_argument('--model1', type=str,  default='vnet_student1', help='model_name')
parser.add_argument('--model2', type=str,  default='vnet_student2', help='model_name')
parser.add_argument('--max_iterations', type=int, default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,  help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[112, 112, 80], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0',help='gpu id')
parser.add_argument('--image_size', type=int, default=[80, 112, 112], help='the size of images for training and testing')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=16,  help='labeled data')
parser.add_argument('--total_sample', type=int, default=80, help='total samples')
# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
# PD
parser.add_argument('--uncertainty_th', type=float,  default=0.1, help='threshold')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        
def getPrototype(features, mask, class_confidence):
    # 使features与掩码mask的大小相匹配
    fts = F.interpolate(features, size=mask.shape[-3:], mode='trilinear')  
    mask_new = mask.unsqueeze(1)
    # 接着，将调整后的特征图fts与扩展后的掩码mask_new逐元素相乘，得到被掩码覆盖的特征图。
    # 这一步的目的是将特征图中属于对象区域的像素保留下来，而背景区域的像素则被置零。
    masked_features = torch.mul(fts, mask_new)
    # 计算被掩码覆盖的特征图中每个通道的加权平均值
    # 这一步的目的是从被掩码覆盖的特征图中提取出属于对象区域的特征表示，同时根据类别置信度对不同通道的贡献进行加权平均。
    masked_fts = torch.sum(masked_features*class_confidence, dim=(2, 3, 4)) /((mask_new*class_confidence).sum(dim=(2, 3, 4)) + 1e-5)  # bs x C
    return masked_fts


def calDist(fts, mask, prototype):
    """
    Calculate the distance between features and prototypes
    """
    # 调整大小。使fts和mask形状保持一致。
    fts_adj_size = F.interpolate(fts, size=mask.shape[-3:], mode='trilinear')
    prototype_new = prototype.unsqueeze(2)
    prototype_new = prototype_new.unsqueeze(3)
    prototype_new = prototype_new.unsqueeze(4)
    # 计算特征与prototype之间距离的平方。
    dist = torch.sum(torch.pow(fts_adj_size - prototype_new, 2), dim=1, keepdim=True)
    return dist

def masked_entropy_loss(p, mask, C=2):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    y1 = mask * y1
    ent = torch.mean(y1)
    return ent

def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    patch_size = args.patch_size
    max_iterations = args.max_iterations
    num_classes = 2

    def create_model1(ema=False):
        # Network definition
        net = net_factory_3d(net_type=args.model1, in_chns=1, class_num=num_classes)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    def create_model2(ema=False):
        # Network definition
        net = net_factory_3d(net_type=args.model2, in_chns=1, class_num=num_classes)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    best_performance1 = float('-inf')  # Initialize with a very low value
    best_performance2 = float('-inf')  # Initialize with a very low value

    model1 = create_model1()
    model2 = create_model2()
    ema_model = create_model1(ema=True)

    db_train = LA_heart(base_dir=train_data_path, split='train',num=None, transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, args.total_sample))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    model1.train()
    model2.train()
    ema_model.train()
    d, h, w = args.image_size[0] // 8, args.image_size[1] // 8, args.image_size[2] // 8  #
    unfolds = unfoldNd.UnfoldNd(kernel_size=(d, h, w), stride=(d, h, w)).to(device)
    folds = unfoldNd.FoldNd(output_size=(args.image_size[0], args.image_size[1], args.image_size[2]), kernel_size=(d, h, w), stride=(d, h, w)).to(device)
    criterion_dsc = DSCLoss(num_classes=args.num_classes, intra_weights=[1.,1.], inter_weight=1., device=device, is_3d=True)
    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)
    # unfolds  = unfoldNd.UnfoldNd(kernel_size=(d, h, w), stride=(d, h, w)).to(device)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label'], # volume_batch[4,1,112,112,80] label_batch:[4,112,112,80]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda() #
            # 获取无标签数据
            unlabeled_volume_batch = volume_batch[args.labeled_bs:] # unlabeled:[2,1,112,112,80]
            # 生成和无标签数据一样形状的噪声
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2) # noise:[2,1,112,112,80]
            # 为无标签数据添加上噪声
            noisy_ema_inputs = unlabeled_volume_batch + noise # [2,1,112,112,80]
            ema_inputs = unlabeled_volume_batch # (2,1,112,112,80)

            # 有标签数据的输出
            outputs1 = model1(volume_batch) # (4,2,112,112,80)
            outputs2 = model2(volume_batch) # (4,2,112,112,80)
            # 对模型的输出进行softmax操作，得到概率分布
            outputs_soft1 = torch.softmax(outputs1, dim=1) # (4,2,112,112,80)
            outputs_soft2 = torch.softmax(outputs2, dim=1) # (4,2,112,112,80)
            # 将概率分布转换为独热编码
            outputs_onehot1 = torch.argmax(outputs_soft1, dim=1) # (4,112,112,80)
            outputs_onehot2 = torch.argmax(outputs_soft2, dim=1) # (4,112,112,80)
            # 使用无梯度计算，获取EMA模型在输入上的输出
            with torch.no_grad():
                ema_output = ema_model(ema_inputs) # (2,2,112,112,80)
                ema_output_soft = torch.softmax(ema_output, dim=1) # (2,2,112,112,80)
                # 获取特征图的中心信息
                LQ_fts = ema_model.featuremap_center # (2,32,56,56,40)
                # 获取带有噪声的EMA模型输出
                noisy_ema_output = ema_model(noisy_ema_inputs) # (2,2,112,112,80)
                noisy_ema_output_soft = torch.softmax(noisy_ema_output, dim=1) # (2,2,112,112,80)

			# P-Err 策略
            # 生成10次EMA模型的预测结果和特征图
            ema_preds = torch.zeros([10, ema_output.shape[0], ema_output.shape[1], ema_output.shape[2], ema_output.shape[3], ema_output.shape[4]]).cuda() # (10,2,2,112,112,80)
            ema_fts = torch.zeros([10, LQ_fts.shape[0], LQ_fts.shape[1], LQ_fts.shape[2], LQ_fts.shape[3], LQ_fts.shape[4]]).cuda() # (10,2,32,56,56,40)
            for i in range(10): 
                with torch.no_grad():
                    #  获取EMA模型在无标签数据上的预测结果
                    ema_preds[i,...] = ema_model(unlabeled_volume_batch) # (10,2,2,112,112,80)
                    # 获取EMA模型的特征
                    ema_fts[i,...] = ema_model.featuremap_center # (10,2,32,56,56,40)
            # 对EMA模型的预测结果进行sigmoid操作
            ema_preds = torch.sigmoid(ema_preds/2.0) # (10,2,2,112,112,80)
            # 计算EMA模型预测结果的标准差
            uncertainty_map = torch.std(ema_preds,dim=0) # (2,2,112,112,80)
            # 计算EMA模型特征图的平均值
            ema_ft = torch.mean(ema_fts,dim=0) # (2,32,56,56,40)

            # 2. 不确定性校正标签
            # 创建确定性掩码
            #_, _, d, w, h = unlabeled_volume_batch.shape # d:10 w:14 h:14
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1) # (4,1,112,112,80)
            uncertainty_map = torch.softmax(uncertainty_map, dim=1) # (2,2,112,112,80)
            uncertainty_map = torch.mean(uncertainty_map, dim=0) # (2,112,112,80)
            uncertainty = -1.0 * torch.sum(uncertainty_map * torch.log(uncertainty_map + 1e-6), dim=1, keepdim=True) # (2,1,112,80)
            uncertainty = uncertainty / math.log(2) # (2,1,112,80)

            # 生成确定性掩码
            certain_mask = torch.zeros_like(uncertainty) # (2,1,112,112,80)
            # 根据设定的不确定性阈值，生成确定性掩码 certain_mask
            certain_mask[uncertainty < args.uncertainty_th] = 1.0
            # 将确定性掩码应用到 EMA 模型输出的概率分布上，以纠正标签
            rect_ema_output_soft = certain_mask * ema_output_soft # (2,2,112,112,80)
            # 获取修正后的独热编码表示
            rect_ema_output_onehot = torch.argmax(rect_ema_output_soft, dim=1) # (2,112,112,80)

            # 3. 原型生成和距离计算
            # 计算对象类别的置信度
            obj_confidence = ema_output_soft[:, 1, ...].unsqueeze(1) # (2,1,112,112,80)
            # 根据矩形修正后的EMA输出的独热编码和对象置信度计算对象原型
            obj_prototype = getPrototype(ema_ft, rect_ema_output_onehot, obj_confidence) # (2,32)
            # 计算EMA特征图到对象原型的距离
            distance_f_obj = calDist(ema_ft, rect_ema_output_onehot, obj_prototype) # (2,1,112,112,80)

            # 计算背景类别的置信度
            bg_confidence = ema_output_soft[:, 0, ...].unsqueeze(1) # (2,1,112,112,80)
            # 将所有背景类别（即类别为 0 的部分）标记为 True，其他类别标记为 False。
            rect_bg_ema_output_onehot = (rect_ema_output_onehot == 0) # (2,1,112,112,80)
            # 根据矩形修正后的EMA输出的独热编码中背景类别的掩码和背景置信度计算背景原型
            bg_prototype = getPrototype(ema_ft, rect_bg_ema_output_onehot, bg_confidence) # (2,32)
            # 计算EMA特征图到背景原型的距离
            distance_f_bg = calDist(ema_ft, rect_bg_ema_output_onehot, bg_prototype) # (2,1,112,112,80)

            # 4. 比较并生成选择掩码
            # 创建用于背景和对象选择的掩码张量
            selection_mask_bg = torch.zeros(distance_f_bg.shape).cuda() # (2,1,112,112,80)
            selection_mask_obj = torch.zeros(distance_f_obj.shape).cuda() # (2,1,112,112,80)
            # 根据对象和背景的距离计算结果，生成背景和对象选择掩码
            selection_mask_bg[distance_f_obj>distance_f_bg] = 1.0 # (2,1,112,112,80)
            selection_mask_obj[distance_f_obj<distance_f_bg] = 1.0 # (2,1,112,112,80)
            # 将背景掩码和对象掩码按照通道维度进行拼接，得到最终的选择掩码
            selection_mask = torch.cat((selection_mask_bg, selection_mask_obj), dim=1) # (2,2,112,112,80)

            # 5.将选择掩码应用到无标签数据的独热编码中，以生成无标签数据的标签
            unlabel_labelbatch_two_channel_1 = torch.zeros_like(outputs1[args.labeled_bs:]).scatter_(dim=1,index= outputs_onehot1[args.labeled_bs:].unsqueeze(dim=1),src=torch.ones_like(outputs1[args.labeled_bs:])) # (2,2,112,112,80)
            unlabel_labelbatch_two_channel_2 = torch.zeros_like(outputs2[args.labeled_bs:]).scatter_(dim=1,index= outputs_onehot2[args.labeled_bs:].unsqueeze(dim=1),src=torch.ones_like(outputs2[args.labeled_bs:])) # (2,2,112,112,80)
            # 创建一个形状为 (batch_size, 2, height, width, depth) 的全零张量 obj_mask，用于表示每个像素属于背景还是对象的标签。
            obj_mask_1 = torch.zeros([outputs_onehot1[args.labeled_bs:].shape[0], 2, outputs_onehot1[args.labeled_bs:].shape[1],  outputs_onehot1[args.labeled_bs:].shape[2],  outputs_onehot1[args.labeled_bs:].shape[3]]).cuda() # (2,2,112,112,80)
            obj_mask_2 = torch.zeros([outputs_onehot2[args.labeled_bs:].shape[0], 2, outputs_onehot2[args.labeled_bs:].shape[1],  outputs_onehot2[args.labeled_bs:].shape[2],  outputs_onehot2[args.labeled_bs:].shape[3]]).cuda() # (2,2,112,112,80)
            # 将选择掩码中为 1 的像素（即对象区域）标记为属于对象，其余标记为属于背景
            obj_mask_1[selection_mask==unlabel_labelbatch_two_channel_1] = 1.0 # (2,2,112,112,80)
            obj_mask_2[selection_mask==unlabel_labelbatch_two_channel_2] = 1.0 # (2,2,112,112,80)
            # 创建背景掩码 bg_mask，obj_mask_1中哪些位置的元素值为 0
            # 换句话说，如果 obj_mask1 中的元素值为 0，则 bg_mask1 中对应位置的元素值为 True；
            # 如果 obj_mask1 中的元素值不为 0，则 bg_mask1 中对应位置的元素值为 False。
            bg_mask1 = (obj_mask_1 == 0) # (2,2,112,112,80)
            bg_mask2 = (obj_mask_2 == 0) # (2,2,112,112,80)
            # 进一步，创建一个背景掩码 bg_mask，表示选择掩码中属于背景区域的像素。
            # 这一步将 obj_mask 中为 0 的像素（即背景区域）标记为 True，其余像素标记为 False
            consistency_weight = get_current_consistency_weight(iter_num//150) # float:0.0006737946999085467


            # 模型在无标签数据上的预测结果与带有噪声的预测结果之间的均方误差
            consistency_dist_1 = losses.softmax_mse_loss(outputs1[args.labeled_bs:], noisy_ema_output)  # (batch, 2, 112,112,80)
            consistency_dist_2 = losses.softmax_mse_loss(outputs2[args.labeled_bs:], noisy_ema_output)  # (batch, 2, 112,112,80)
            # consistency_loss = torch.sum(noisy_mask*consistency_dist)/(torch.sum(noisy_mask)+1e-16)
            consistency_loss = torch.sum(torch.mean(bg_mask1 * consistency_dist_1)) + torch.sum(torch.mean(bg_mask2 * consistency_dist_2))
            consistency_loss = consistency_loss / (torch.sum(torch.mean(bg_mask1.float())) + torch.sum(torch.mean(bg_mask2.float())) + 1e-16)

            # 模型在无标签数据上的预测结果与无噪声的预测结果之间的交叉熵损失
            entropy_loss_1 = torch.mean(masked_entropy_loss(outputs_soft1[args.labeled_bs:], obj_mask_1, C=2))
            entropy_loss_2 = torch.mean(masked_entropy_loss(outputs_soft2[args.labeled_bs:], obj_mask_2, C=2))

            consistency_loss += entropy_loss_1 + entropy_loss_2

            # Supervised loss
            loss_ce1 = ce_loss(outputs1[:args.labeled_bs], label_batch[:args.labeled_bs][:])
            loss_ce2 = ce_loss(outputs2[:args.labeled_bs], label_batch[:args.labeled_bs][:])
            loss_ce = torch.mean(loss_ce1) + torch.mean(loss_ce2)

            loss_dice1 = dice_loss(outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_dice2 = dice_loss(outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_dice = torch.mean(loss_dice1) + torch.mean(loss_dice2)

            lambda_ = 0.2

            # 混合伪标签监督策略
            pred_t_logits = ema_model(ema_inputs)  # (2,2,112,112,80)
            pred_u = ema_model(noisy_ema_inputs)  # (2,2,112,112,80)
            pred_u_logits = pred_u  # (2,2,112,112,80)
            pred_u_probs = torch.softmax(pred_u_logits, dim=1)  # (2,2,112,112,80)
            pred_u_pseudo = torch.argmax(pred_u_probs, dim=1)  # (2,112,112,80)
            pred_u_conf = pred_u_probs.max(dim=1)[0].clone()  # (2,112,112,80)

            pred_u1A1 = model1(volume_batch)  # (4,2,112,112,80)
            pred_u1A1_logits = pred_u1A1  # (4,2,112,112,80)
            pred_u1A1_probs = torch.softmax(pred_u1A1_logits, dim=1)  # (4,2,112,112,80)
            pred_u1A1_pseudo = torch.argmax(pred_u1A1_probs, dim=1)  # (4,112,112,80)
            pred_u1A1_conf = pred_u1A1_probs.max(dim=1)[0].clone()  # (4,112,112,80)

            pred_u2A2 = model2(volume_batch)  # (4,2,112,112,80)
            pred_u2A2_logits = pred_u2A2  # (4,2,112,112,80)
            pred_u2A2_probs = torch.softmax(pred_u2A2_logits, dim=1)  # (4,2,112,112,80)
            pred_u2A2_pseudo = torch.argmax(pred_u2A2_probs, dim=1)  # (4,112,112,80)
            pred_u2A2_conf = pred_u2A2_probs.max(dim=1)[0].clone()  # (4,112,112,80)
            # 交叉验证
            loss_x = (criterion_dsc(pred_u1A1_logits, pred_u2A2_pseudo.detach()) + criterion_dsc(pred_u2A2_logits, pred_u1A1_pseudo.detach())) / 2.
            # 不交叉验证
            # loss_x = (criterion_dsc(pred_u1A1_logits, pred_u1A1_pseudo.detach()) + criterion_dsc(pred_u2A2_logits, pred_u2A2_pseudo.detach())) / 2.

            #loss_u = (criterion_dsc(pred_u1A1_logits, pred_u_pseudo.detach()) + criterion_dsc(pred_u2A2_logits,   pred_u_pseudo.detach())) / 2.
            supervised_loss = (loss_ce + loss_dice + loss_x  * lambda_ ) / 2
            #supervised_loss = (loss_ce + loss_dice + loss_x * 0.1 * lambda_ + loss_u * 0.1 * lambda_) / 2

            loss = supervised_loss + consistency_weight * consistency_loss
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            update_ema_variables(model1, ema_model, args.ema_decay, iter_num)
            update_ema_variables(model2, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_

            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',consistency_weight, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num % 200 == 0:
                pass

            if iter_num >= 800 and iter_num % 100 == 0:
                model1.eval()
                avg_metric = test_all_case(
                    model1, args.root_path, test_list="test.list", num_classes=2, patch_size=args.patch_size,
                    stride_xy=18, stride_z=4)
                # print('DICE',avg_metric[0])
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model1))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score',  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95',avg_metric[0, 1], iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
                model1.train()
                model2.train()
                # 加权保存2个模型的权重
                # model1.eval()
                # model2.eval()
                # avg_metric1 =  test_all_case(model1, args.root_path, test_list="test.list", num_classes=2, patch_size=args.patch_size, stride_xy=18, stride_z=4)
                # avg_metric2 =  test_all_case(model2, args.root_path, test_list="test.list", num_classes=2, patch_size=args.patch_size, stride_xy=18, stride_z=4)
                # avg_metric = (avg_metric1 + avg_metric2) / 2  # Combine the metrics for saving decision
                #
                # if avg_metric.mean() > best_performance1 and avg_metric.mean() > best_performance2:
                #     best_performance1 = avg_metric1
                #     best_performance2 = avg_metric2
                #     save_mode_path = os.path.join(snapshot_path, 'best_combined_model.pth')
                #     save_combined_models(model1, model2, save_mode_path)

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(args.exp, args.labeled_num, args.model1)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('..', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)


