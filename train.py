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
from utils.cal_less_more import pred_less_and_more
from utils.lr_scheduler import build_scheduler
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def get_each_au_num(loader):
    matrix = torch.zeros((6,24)).cuda()
    for  imgs, labels in loader:
        bs, _ = labels.shape  #bs, 24
        labels = labels.cuda()
        for i in range(bs):
            matrix[0] += torch.eq(labels[i,:], 0)
            matrix[1] += torch.gt(labels[i,:], 0) & torch.lt(labels[i,:], 0.2) | torch.eq(labels[i,:], 0.2)
            matrix[2] += torch.gt(labels[i,:], 0.2) & torch.lt(labels[i,:], 0.4) | torch.eq(labels[i,:], 0.4)
            matrix[3] += torch.gt(labels[i,:], 0.4) & torch.lt(labels[i,:], 0.6) | torch.eq(labels[i,:], 0.6)
            matrix[4] += torch.gt(labels[i,:], 0.6) & torch.lt(labels[i,:], 0.8) | torch.eq(labels[i,:], 0.8)
            matrix[5] += torch.gt(labels[i,:], 0.8) & torch.lt(labels[i,:], 1.0) | torch.eq(labels[i,:], 1.0)
    return matrix
def get_mean_std_value(loader):
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
    
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)  # gpu
    np.random.seed(args.random_seed)  # numpy

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

    logger = create_logger(args.log_dir, model_name='resnet18')
    transform_train = Compose([
        #RandomCrop(224, pad_if_needed=True, fill=0),
        Resize((224,224)),
        RandomColorjitter(),
	    ToTensor(),
	    #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        Normalize(mean =[0.4743, 0.3539, 0.3249],std = [0.2697, 0.2238, 0.2154]),
        ])
    transform_val = Compose([
        #CenterCrop(224),
        Resize((224,224)),
        ToTensor(),
        #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        Normalize(mean =[0.4743, 0.3539, 0.3249],std = [0.2697, 0.2238, 0.2154]),
        ])
    
    file_names = ['FEAFA','Disfa'] #分别是FEAFA-A,FEAFA-B的存储目录名
    train_file_path = [os.path.join(args.file_path, i, i+'_train') for i in file_names]
    val_file_path = [os.path.join(args.file_path, i, i+'_test') for i in file_names]
    logger.info("train_file_path:"+ str(train_file_path))
    logger.info("val_file_path:"+ str(val_file_path))
    train_dataset_list = [AuDataset(data_path=i, transform=transform_train, iscrop=args.iscrop, mode='Train') for i in train_file_path] 
    val_dataset_list =[AuDataset(data_path=i, transform=transform_val, iscrop=args.iscrop, mode='Test') for i in val_file_path]
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
    if args.model == 'vit':
        model = torchvision.models.vit_b_16(num_classes=24,dropout=0.5)
        model.heads = nn.Sequential(nn.Linear(model.hidden_dim, 24), nn.Sigmoid())
    if args.model == 'swin':
        model = torchvision.models.swin_t(num_classes=24, dropout=0.8)
        model.heads = nn.Sequential(nn.Linear(model.hidden_dim, 24), nn.Sigmoid())
    if args.model == 'resnet':
        model = torchvision.models.resnet18()
        num_features = model.fc.in_features
        print(num_features)
        model.fc = nn.Sequential(
            nn.Dropout(0.8),
            nn.Linear(num_features, 24), 
            nn.Hardsigmoid()  # 使用 Sigmoid 激活函数将输出限制在0-1之间 
            )
    model = model.cuda()
    # model_dict = model.state_dict() 
    # ori_model = torchvision.models.resnet50(pretrained=True)
    # pretrained_dict = {k: v for k, v in ori_model.state_dict().items() if k in model_dict and (v.shape == model_dict[k].shape)}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict, strict=False)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    if args.criterion == 'mse':
         criterion = torch.nn.MSELoss()
    elif args.criterion == 'bce':
        criterion = torch.nn.BCELoss()
    elif args.criterion == 'smoothl1':
        criterion = torch.nn.SmoothL1Loss()
    #lr_scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=args.milestone, gamma=0.1, verbose=True)
    lr_scheduler = build_scheduler(args, optimizer, len(train_loader))
    print("-----------------")
    print("lr_scheduler = {}".format(lr_scheduler))
    print("-----------------")

    for k in args.__dict__:
        logger.info("{} : {}".format(k, args.__dict__[k]))

    # begin to train
    logger.info("------start training------")
    min_val_loss = np.inf
    
    best_epoch = 0
    best_mae = 1
    best_acc = 0
    best_loss = 1
    best_icc = 0
    best_model = None
    if not os.path.exists(args.save_path ):
        os.mkdir(args.save_path)
    for epoch in range(args.epochs):
        # print("lr =  {}".format(lr_scheduler._get_lr()))
        
        model.train()
        loss = 0.0
        train_loss, train_acc = train_one_epoch(train_loader,  model, optimizer, criterion, lr_scheduler, epoch, args, logger)
        val_loss, val_acc, val_mae, val_icc, val_pl, val_pm, pred, label = evalutate(val_loader,  model, criterion, epoch, args, logger)
        pred_less_and_more(pred, label)
        #lr_scheduler.step()
        if min_val_loss >= val_loss:
            min_val_loss = val_loss
            best_epoch = epoch
            logger.info("now the best epoch is {}, val_loss = {:.5f}, val_mae = {:.5f}, acc = {:.5f}, icc = {:.5f} ".format(epoch, val_loss, val_mae, val_acc, val_icc))
            best_model = model
            best_mae = val_mae
            best_loss = val_loss
            best_acc = val_acc
            best_icc = val_icc
        # if epoch %2 == 0:
        #     torch.save(model.state_dict(), args.save_path+"/epoch{}.pth".format(epoch))
    print("------finish training------")
    logger.info("the best epoch is {}, val_loss = {:.5f}, val_mae = {:.5f}, acc = {:.5f}, icc = {:.5f}, pl = {:.5f}, pm = {:.5f} ".format(best_epoch, best_loss, best_mae, best_acc, best_icc, val_pl, val_pm))
    torch.save(best_model.state_dict(), args.save_path+"/best.pth")
    pred_less_and_more(pred, label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--file_path',type=str,default="/media/ljy/新加卷1/FEAFA+")
    parser.add_argument('--model',type=str,default="resnet")
    parser.add_argument('--num_class',type=int,default=24)


    parser.add_argument('--lr_scheduler',type=str,default="cosine")
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--warmup_epochs',type=int,default=2)
    parser.add_argument('--decay_epochs',type=int,default=2)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial base learning rate.')
    parser.add_argument('--warmup_lr', type=float, default=5e-7,
                        help='Initial warmup learning rate.')
    parser.add_argument('--min_lr', type=float, default=5e-6,
                        help='min learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay (L2 loss on parameters).')
    

    parser.add_argument('--print_fq', type=int, default=20,
                        )
    parser.add_argument('--save_path',type=str,default="/media/ljy/ubuntu_disk1/jhy_code/resnet-for-au/checkpoint/0706_swin_dropout0.8")
    parser.add_argument('--log_dir',type=str,default="/media/ljy/ubuntu_disk1/jhy_code/resnet-for-au/log/")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--iscrop", type=str, default="_crop")
    parser.add_argument("--criterion", type=str, default="mse")
    args = parser.parse_args()
    main(args)