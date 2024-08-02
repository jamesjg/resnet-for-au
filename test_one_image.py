import argparse
import os
import parser
import random
from time import perf_counter
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets
from Dataset.transform import ToTensor,Normalize,Compose,UnNormalize,RandomCrop,RandomColorjitter,CenterCrop,Resize
from torch.utils.data import DataLoader
from Dataset.au_dataset import AuDataset
import numpy as np
import torch.nn.functional as F
from process.engine import train_one_epoch, evalutate
from PIL import Image
import cv2
import matplotlib.pyplot as plt

save_path = "/media/ljy/ubuntu_disk1/jhy_code/resnet-for-au/checkpoint/0613_detectfaceandtransformthenrandresize_dropout0_8_weight_decay-2_alldata_hardsigmoid/best.pth"
img_path = "/media/ljy/新加卷/FEAFA+/FEAFA/FEAFA_train/PV057.output/00000994_crop.jpg"
label_path = "/media/ljy/新加卷/FEAFA+/FEAFA/FEAFA_train/PV057.output/00000994.auw"
if __name__ == '__main__':
    
    torch.manual_seed(42)
    random.seed(42)
    torch.cuda.manual_seed(42)  # gpu
    np.random.seed(42)  # numpy
    box_dir = '/media/ljy/ubuntu_disk1/jhy_code/resnet-for-au/face_box_info'
    box_file_name = img_path.split('/')[-2]+img_path.split('/')[-1].split('.')[0]+'.txt'
    box_path = os.path.join(box_dir, box_file_name)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    transform_val = Compose([
        Resize(224),
        #CenterCrop(224),
        ToTensor(),
        Normalize(mean =[0.4743, 0.3539, 0.3249],std = [0.2697, 0.2238, 0.2154]),
        ])
    model = torchvision.models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(),
        nn.Linear(num_features, 24),
        nn.Hardsigmoid()  # 使用 Sigmoid 激活函数将输出限制在0-1之间
        )
    model.load_state_dict(torch.load(save_path))
    model.cuda()
    model.eval()
    with torch.no_grad():
        #import ipdb;ipdb.set_trace()
        assert os.path.exists(img_path), "{} dose not exist".format(img_path)
        img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)

        box_info = np.loadtxt(box_path)
        if len(box_info)<4:
            os.remove(box_path)
            xmin, ymin, weight, height = (0, 0, img.shape[0], img.shape[1])
        else:
            xmin, ymin, weight, height = int(box_info[0]), int(box_info[1]), int(box_info[2]), int(box_info[3])
        xmax = xmin + weight
        ymax = ymin + height
        img = img[xmin:xmax, ymin:ymax]
        print(xmin,ymin, xmax, ymax)

        label = np.loadtxt(label_path,dtype=float)
        img, label = transform_val(img, label)

        label = label.float()
        img = img.unsqueeze(0)
        img = img.cuda()

        ori_img=UnNormalize(mean =[0.4743, 0.3539, 0.3249],std = [0.2697, 0.2238, 0.2154])(img)
        torchvision.utils.save_image(ori_img[0],'images/test_image.jpg',normalize=True)
        label = label.cuda()
        label = label.unsqueeze(0)
        output = model(img)


        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(output, label)
        mae_value = torch.mean(torch.abs(label-output))   
    print("the output is :", output)
    print("the label is :", label)
    print("the loss is  :", loss)
    print("the mae_value is:", mae_value)
    # cv2.imshow('Image', cv2.imread(img_path))  # 显示图像
    # cv2.waitKey(0)  # 等待按键退出
    # cv2.destroyAllWindows()  # 关闭窗口