import random
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
    def __init__(self, data_path, transform=None, iscrop=''):  # data_path: train/test
        super(AuDataset, self).__init__()
        data_folders = os.listdir(data_path) #["xxxoutput","xxxoutput"...]
        imgs_path = []
        labels_path = []
        loss = 0
        drop = 0
        #path_and_name = []
        for folder in data_folders:
            item_list = os.listdir(os.path.join(data_path,folder))
            for item in item_list:
                if item.split('.')[-1] == 'auw': #先找label
                    label_path = os.path.join(data_path,folder,item)  #"../././xxxxx.auw"
                    #然后读入对应的crop_img
                    label=np.loadtxt(label_path,dtype=float)
                    img_path_jpg = os.path.join(data_path,folder,item.split('.')[0]+iscrop+'.jpg')   #".././xxxxx_crop.jpg"
                    img_path_png = os.path.join(data_path,folder,item.split('.')[0]+iscrop+'.png')
                    if np.min(label)==0:
                        p=random.uniform(0,1)
                        if p < 0.5:
                            drop += 1
                            continue
                        else:
                            if os.path.exists(img_path_jpg):
                                imgs_path.append(img_path_jpg)
                                labels_path.append(label_path)
                            elif  os.path.exists(img_path_png):
                                imgs_path.append(img_path_png)
                                labels_path.append(label_path)
                            else:
                                #print(img_path_jpg)
                                #print("{} dose not exist".format(img_path))
                                loss += 1
                # if os.path.join(data_path,folder,item.split('.')[0]) != pre_name:
                #     path_and_name.append(os.path.join(data_path,folder,item.split('.')[0]))  # 路径名加编号，不带后缀
                #     pre_name = os.path.join(data_path,folder,item.split('.')[0])

        self.transform = transform
        #self.path_and_name = path_and_name 
        #import ipdb;ipdb.set_trace()
        self.face_aligner = FaceAligner()
        self.labels_path = labels_path
        self.imgs_path = imgs_path
        self.loss_num = loss
        self.iscrop = iscrop
        self.drop = drop
        # print(data_path+" all :", len(self.labels_path))
        print(data_path+ " drop :", self.drop)
        
    def __getitem__(self, idx):

        #get image
        label_path = self.labels_path[idx]
        img_path = self.imgs_path[idx]
        assert os.path.exists(img_path), "no img found in {}".format(img_path)
        img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
        #利用特征点检测器处理得到特征点
        if not self.iscrop:
            landmark_helper = Landmark_helper(Method_type='dlib')
            landmark, flag = landmark_helper.detect_facelandmark(img)
            #如果检测到人脸
            if flag:
                assert(landmark.shape==(68,2)),'landmark shape is wrong {:}'.format(landmark.shape)
                img, new_landmarks=self.face_aligner.align(img, landmark)
        
        # get label
        label=np.loadtxt(label_path,dtype=float)
        
        #print(label)
        # transform
        if self.transform:
            img, label = self.transform(img, label)
        assert label.shape[0]==24, 'label shape is {:}'.format(label.shape)
        return img, label.float()
                 
    def __len__(self):
        return len(self.labels_path)
    
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