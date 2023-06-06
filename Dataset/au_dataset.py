from random import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import cv2
from Dataset.Landmark_helper import Landmark_helper
from Dataset.face_aligner import FaceAligner

class AuDataset(Dataset):
    def __init__(self, data_path, transform=None):  # data_path: train/test
        super(AuDataset, self).__init__()
        data_folders = os.listdir(data_path)
        # imgs_path = []
        # labels_path = []
        path_and_name = []
        s = set()
        for folder in data_folders:
            item_list = os.listdir(os.path.join(data_path,folder))
            pre_name = None
            for item in item_list:
                if os.path.join(data_path,folder,item.split('.')[0]) != pre_name:
                    path_and_name.append(os.path.join(data_path,folder,item.split('.')[0]))  # 路径名加编号，不带后缀
                    pre_name = os.path.join(data_path,folder,item.split('.')[0])
                # if item.split('.')[-1] == 'auw':
                #     labels_path.append(os.path.join(data_path,folder,item))
                # else:
                #     imgs_path.append(os.path.join(data_path,folder,item))
        # self.imgs_path = imgs_path
        # self.labels_path = labels_path
        self.transform = transform
        self.path_and_name = path_and_name 
        #import ipdb;ipdb.set_trace()
        self.face_aligner = FaceAligner()
        
    def __getitem__(self, idx):

        #get image
        img_path_and_name = self.path_and_name[idx]
        img_path = img_path_and_name + '.jpg'
        assert os.path.exists(img_path), "no img found in {}".format(img_path)
        img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
        利用特征点检测器处理得到特征点
        landmark_helper = Landmark_helper(Method_type='dlib')
        landmark, flag = landmark_helper.detect_facelandmark(img)
        #如果检测到人脸
        if flag:
            assert(landmark.shape==(68,2)),'landmark shape is wrong {:}'.format(landmark.shape)
            img, new_landmarks=self.face_aligner.align(img, landmark)

        # get label
        label_path = img_path.replace('jpg','auw')    #根据img_path 找到对应的label_path
        f = open(label_path, 'r')
        lis = f.readline().split(' ')
        label = list(map(float, lis))
        f.close()
        # transform
        if self.transform:
            img, label = self.transform(img, label)
        assert label.shape[0]==24, 'label shape is {:}'.format(label.shape)
        return img.cuda(), label.cuda()
                 
    def __len__(self):
        return len(self.path_and_name)
    
    def zero_label_num(self):
        count = 0
        for i in self.path_and_name:
            label_path = i + '.auw'
            f = open(label_path, 'r')
            lis = f.readline().split(' ')
            label = list(map(float, lis))
            f.close()
            # transform
            label = torch.tensor(label)
            assert label.shape[0]==24, 'label shape is {:}'.format(label.shape)
            if torch.sum(label) == 0:
                count += 1
                #print(i)
        return count, len(self.path_and_name)