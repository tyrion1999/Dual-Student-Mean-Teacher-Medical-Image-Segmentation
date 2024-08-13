import argparse
import os
import shutil
from glob import glob

import torch
from networks.net_factory_3d import net_factory_3d
from test_3D_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/stu/zy/data/LA', help='Name of Experiment')
#parser.add_argument('--root_path', type=str, default='/home/stu/zy/data/PAN', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='LA_ablation_全是三线性差值', help='experiment_name')
# parser.add_argument('--model', type=str,  default='vnet_student1', help='model_name')
parser.add_argument('--model', type=str,  default='vnet', help='model_name')
parser.add_argument('--gpu', type=str, default='0',  help='gpu id')


def Inference(FLAGS):
    snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
    num_classes = 2
    test_save_path = "../model/{}/{}_DS-MT LA 全改成最近邻".format(FLAGS.exp, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    print('生成的nii.gz文件保存于：',test_save_path)
    os.makedirs(test_save_path)
    net = net_factory_3d(net_type=FLAGS.model, in_chns=1, class_num=num_classes).cuda()
    # save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    # save_mode_path = "/home/stu/zy/AC-MT-main/model/_ACMT_/vnet_student1/vnet_student1_best_model.pth"
    save_mode_path = "/home/stu/zy/AC-MT-main/model/LA_ACMT_改阈值_0.5_8/vnet_student1/iter_700_dice_0.8134.pth"
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric, std = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.list", num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4, test_save_path=test_save_path)
    return avg_metric, std


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    metric = Inference(FLAGS)
    print('dice, jc, hd, asd:', metric)
