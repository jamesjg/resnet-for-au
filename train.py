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
import matplotlib.pyplot as plt
#from timm import create_model
from Dataset.get_landmarks import align_face
from process.help import create_logger
def main(args):

    transform_train = Compose([
        RandomCrop(224),
        RandomColorjitter(),
	    ToTensor(),
	    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    transform_val = Compose([
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    train_dataset = AuDataset(data_path=os.path.join(args.file_path, 'train'), 
                                        transform = transform_train) 
    val_dataset = AuDataset(data_path=os.path.join(args.file_path, 'val'),  
                                        transform = transform_val) 
    
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    zero_train = train_dataset.zero_label_num()
    zero_val = val_dataset.zero_label_num()
    print('train_num:', train_num, 'val_num:', val_num)   
    print('zero_label_train:', zero_train, 'zero_val:', zero_val)
    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True,          
                            num_workers=0)      
    
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=0)
    
    #加载模型，使用预训练模型，但是最后一层重新训练
    model = torchvision.models.resnet50().cuda()
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
    lr_scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[15], gamma=0.1)
    # logger, final_output_dir, tb_log_dir = create_logger(
    #     cfg, '{:}_lr{:}_{:}_{:}'.format(args.model,lr,optimizer,'mse'), 'train')
    # writer_dict = {
    #     'writer': SummaryWriter(log_dir=tb_log_dir),
    #     'train_global_steps': 0,
    #     'valid_global_steps': 0,
    # }
    # begin to train
    print("------start training------")
    min_val_loss = np.inf
    
    best_epoch = 0
    best_mae = 1
    best_auc = 0
    
    # count_labels = torch.zeros(24).cuda()
    # for _, (_, labels) in enumerate(train_dataset):
    #     count_labels += labels>0
    # print(count_labels)
    # exit()
    best_model = None
    
    for epoch in range(args.epochs):
        #print("lr =  {}".format(lr_scheduler.get_lr()))
        
        model.train()
        loss = 0.0
        train_loss, train_acc = train_one_epoch(train_loader,  model, optimizer, mse_loss, epoch, args)
        val_loss, val_acc, val_mae = evalutate(val_loader,  model, mse_loss, epoch, args)
        lr_scheduler.step()
        if min_val_loss >= val_loss:
            min_val_loss = val_loss
            best_epoch = epoch
            print("now the best epoch is {}, val_loss = {:.5f}, acc = {:.5f} ".format(epoch, val_loss, val_acc))
            best_model = model
    print("------finish training------")
    print("the best epoch is " + str(best_epoch))
    torch.save(best_model.state_dict(), args.save_path)
    # img = Image.open(r"C:\Users\James\Desktop\huijin.png").convert('RGB')
    # img = transform_val(img).cuda()
    # print(model(img.unsqueeze(0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--file_path',type=str,default="E:\dataset\\face\FEAFA-A")
    parser.add_argument('--num_class',type=int,default=24)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--print_fq', type=int, default=20,
                        )
    parser.add_argument('--save_path',type=str,default="E:\project\\resnet-for-au\checkpoint\\0605_20e.pth")
    
    args, _ = parser.parse_known_args()
    main(args)