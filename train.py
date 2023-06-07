import argparse
import os
import parser
from time import perf_counter
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets
from Dataset.transform import ToTensor,Normalize,Compose,UnNormalize,RandomCrop,RandomColorjitter,CenterCrop,Resize
from torch.utils.data import DataLoader, ConcatDataset
from Dataset.au_dataset import AuDataset
import numpy as np
import torch.nn.functional as F
from process.engine import train_one_epoch, evalutate
from PIL import Image
import matplotlib.pyplot as plt
#from timm import create_model
from Dataset.get_landmarks import align_face
from utils.get_logger import create_logger
def get_mean_std_value(loader):
    '''
    求数据集的均值和标准差
    :param loader:
    :return:
    '''
    data_sum,data_squared_sum,num_batches = 0,0,0

    for data,_ in loader:
        # data: [batch_size,channels,height,width]
        # 计算dim=0,2,3维度的均值和，dim=1为通道数量，不用参与计算
        data_sum += torch.mean(data,dim=[0,2,3])    # [batch_size,channels,height,width]
        # 计算dim=0,2,3维度的平方均值和，dim=1为通道数量，不用参与计算
        data_squared_sum += torch.mean(data**2,dim=[0,2,3])  # [batch_size,channels,height,width]
        # 统计batch的数量
        num_batches += 1
    # 计算均值
    mean = data_sum/num_batches
    # 计算标准差
    std = (data_squared_sum/num_batches - mean**2)**0.5
    return mean,std

def main(args):
    logger = create_logger(args.log_dir, model_name='resnet18')
    transform_train = Compose([
        #RandomCrop(224),
        Resize(224),
        RandomColorjitter(),
	    ToTensor(),
	    #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        Normalize(mean =[0.4743, 0.3539, 0.3249],std = [0.2697, 0.2238, 0.2154]),
        ])
    transform_val = Compose([
        #CenterCrop(224),
        Resize(224),
        ToTensor(),
        #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        Normalize(mean =[0.4743, 0.3539, 0.3249],std = [0.2697, 0.2238, 0.2154]),
        ])
    
    file_names = ['FEAFA','Disfa'] #分别是FEAFA-A,FEAFA-B的存储目录名
    train_file_path = [os.path.join(args.file_path, i, i+'_train') for i in file_names]
    val_file_path = [os.path.join(args.file_path, i, i+'_test') for i in file_names]
    logger.info("train_file_path:"+ str(train_file_path))
    logger.info("val_file_path:"+ str(val_file_path))
    train_dataset_list = [AuDataset(data_path=i, transform=transform_train, iscrop=args.iscrop) for i in train_file_path] 
    val_dataset_list =[AuDataset(data_path=i, transform=transform_val, iscrop=args.iscrop) for i in val_file_path]
    train_dataset = ConcatDataset(train_dataset_list)
    val_dataset = ConcatDataset(val_dataset_list)
    
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    logger.info("train_num : {}, val_num : {}".format(train_num, val_num))
    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True,          
                            num_workers=args.num_workers)      
    
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)
    #加载模型，使用预训练模型，但是最后一层重新训练
    # print('==> Computing mean and std..')
    # mean,std = get_mean_std_value(train_loader)
    # print('mean = {},std = {}'.format(mean,std))
    model = torchvision.models.resnet18().cuda()
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 24),
        nn.Sigmoid()  # 使用 Sigmoid 激活函数将输出限制在0-1之间
        )
    model = model.cuda()
    # model_dict = model.state_dict() 
    # ori_model = torchvision.models.resnet50(pretrained=True)
    # pretrained_dict = {k: v for k, v in ori_model.state_dict().items() if k in model_dict and (v.shape == model_dict[k].shape)}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict, strict=False)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    mse_loss = torch.nn.MSELoss()
    lr_scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[4,8], gamma=0.1, verbose=True)

    # begin to train
    logger.info("------start training------")
    min_val_loss = np.inf
    
    best_epoch = 0
    best_mae = 1
    best_acc = 0
    best_loss = 1
    best_model = None
    
    for epoch in range(args.epochs):
        #print("lr =  {}".format(lr_scheduler.get_lr()))
        
        model.train()
        loss = 0.0
        train_loss, train_acc = train_one_epoch(train_loader,  model, optimizer, mse_loss, epoch, args, logger)
        val_loss, val_acc, val_mae = evalutate(val_loader,  model, mse_loss, epoch, args, logger)
        lr_scheduler.step()
        if min_val_loss >= val_loss:
            min_val_loss = val_loss
            best_epoch = epoch
            logger.info("now the best epoch is {}, val_loss = {:.5f}, val_mae = {:.5f}, acc = {:.5f} ".format(epoch, val_loss, val_mae, val_acc))
            best_model = model
            best_mae = val_mae
            best_loss = val_loss
            best_acc = val_acc
            
            
    print("------finish training------")
    #logger.info("the best epoch is " + str(best_epoch))
    logger.info("the best epoch is {}, val_loss = {:.5f}, val_mae = {:.5f}, acc = {:.5f} ".format(best_epoch, best_loss, best_mae, best_acc))
    torch.save(best_model.state_dict(), args.save_path)
    # img = Image.open(r"C:\Users\James\Desktop\huijin.png").convert('RGB')
    # img = transform_val(img).cuda()
    # print(model(img.unsqueeze(0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--file_path',type=str,default="/media/ljy/新加卷1/FEAFA+")
    parser.add_argument('--num_class',type=int,default=24)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--print_fq', type=int, default=20,
                        )
    parser.add_argument('--save_path',type=str,default="/media/ljy/ubuntu_disk/jhy_code/resnet-for-au/checkpoint/0607_10e_milestone4_8_100loss_drop0.pth")
    parser.add_argument('--log_dir',type=str,default="/media/ljy/ubuntu_disk/jhy_code/resnet-for-au/log/")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--iscrop", type=str, default="_crop")
    args, _ = parser.parse_known_args()
    main(args)