import argparse
import os
import parser
from time import perf_counter
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets
from Dataset.transform import ToTensor,Normalize,Compose,UnNormalize,RandomCrop,RandomColorjitter,CenterCrop
from torch.utils.data import DataLoader
from Dataset.au_dataset import AuDataset
import numpy as np
import torch.nn.functional as F
from process.engine import train_one_epoch, evalutate
from PIL import Image
import cv2
import matplotlib.pyplot as plt

save_path = "/media/ljy/ubuntu_disk/jhy_code/resnet-for-au/checkpoint/0605_20e_lr1e-4.pth"
img_path = "/media/ljy/新加卷1/FEAFA-a/2.outputgong2_2_00004832_crop.jpg"
label_path = img_path.replace('jpg', 'auw')
if __name__ == '__main__':
    # transform_val = Compose([
    #     CenterCrop(224),
    #     ToTensor(),
    #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    # model = torchvision.models.resnet50()
    # num_features = model.fc.in_features
    # model.fc = nn.Sequential(
    #     nn.Linear(num_features, 24),
    #     nn.Sigmoid()  # 使用 Sigmoid 激活函数将输出限制在0-1之间
    #     )
    # model.load_state_dict(torch.load(save_path))
    # model.cuda()
    # #import ipdb;ipdb.set_trace()
    # img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
    # f = open(label_path, 'r')
    # lis = f.readline().split(' ')
    # label = list(map(float, lis))
    # f.close()
    # img, label = transform_val(img, label)
    # img = img.unsqueeze(0)
    # img = img.cuda()
    # label = label.cuda()
    # label = label.unsqueeze(0)
    # output = model(img)


    # mse_loss = torch.nn.MSELoss()
    # loss = mse_loss(output, label)
    # mae_value = torch.mean(torch.abs(label-output))   
    # print("the output is :", output)
    # print("the label is :", label)
    # print("the mse is  :", loss)
    # print("the mae_value is:", mae_value)
    cv2.imshow('Image', cv2.imread(img_path))  # 显示图像
    cv2.waitKey(0)  # 等待按键退出
    cv2.destroyAllWindows()  # 关闭窗口