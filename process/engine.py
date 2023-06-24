import torch
from Dataset.transform import UnNormalize
from utils.misc import AverageMeter
import time
import torchvision
def train_one_epoch(train_loader, model, optimizer, criterion, epoch,  args, logger):
    batch_time = AverageMeter()  #一个batch中模型运行时间
    data_time = AverageMeter()  #一个batch中加载数据时间
    loss = AverageMeter()
    mae = AverageMeter()
    acc = AverageMeter()
    model.train()
    batch_start_time = time.time()

    for idx, (imgs, labels) in enumerate(train_loader):
        if epoch == 0 and idx <= 10:
            ori_imgs=UnNormalize(mean =[0.4743, 0.3539, 0.3249],std = [0.2697, 0.2238, 0.2154])(imgs)
            torchvision.utils.save_image(ori_imgs[0],'images/image_{:}.jpg'.format(idx),normalize=True)
        imgs = imgs.cuda()
        labels = labels.cuda()
        bs = labels.shape[0]
        data_time.update(time.time() - batch_start_time)
        optimizer.zero_grad()
        output = model(imgs)
        loss_train = criterion(100*output, 100*labels) #这个batch的平均损失, 每个样本在每个AU上的平均均方损失
        loss_train.backward()
        optimizer.step()

        loss.update(loss_train.item(), bs)
        batch_time.update(time.time() - batch_start_time)
        batch_start_time = time.time()

        mae_array = torch.mean(torch.abs(labels-output), dim=1)  #batchsize个样本的每个AU的平均绝对值损失
        mae_value = torch.mean(mae_array)   #每个样本在每个AU上的平均绝对值损失
        predict_true_array = torch.where(mae_array < 0.08, 1, 0)
        acc_value = torch.sum(predict_true_array) / bs

        mae.update(mae_value, bs)
        acc.update(acc_value, bs)
        
        # global_steps = writer_dict['train_global_steps']
        #import ipdb;ipdb.set_trace()
        if idx % args.print_fq == 0 or idx + 1 == len(train_loader):
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Speed {speed:.1f} samples/s\t' \
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                    'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                    'MAE {mae.val:.5f} ({mae.avg:.5f})\t' \
                    'ACC {acc.val:.5f} ({acc.avg:.5f})\t' .format (
                epoch, idx, len (train_loader), batch_time=batch_time,
                speed=bs / batch_time.val,
                data_time=data_time, loss=loss, mae=mae,acc=acc)
            logger.info (msg)

        # writer = writer_dict['writer']
        # writer.add_scalar('train_loss', loss.val, global_steps)
        # writer.add_scalar('train_acc', mae.val, global_steps)
        # writer_dict['train_global_steps'] = global_steps + 1

    return acc.avg, mae.avg

def evalutate(val_loader, model, criterion, epoch, args, logger):
    batch_time = AverageMeter()  #一个batch中模型运行时间
    data_time = AverageMeter()  #一个batch中加载数据时间
    loss = AverageMeter()
    mae = AverageMeter()
    acc = AverageMeter()
        
    batch_start_time = time.time()
    print("evaluating...")
    with torch.no_grad():
        model.eval()
        # true_positives = torch.zeros(24).cuda()
        # predicted_positives = torch.zeros(24).cuda()
        # actual_positives = torch.zeros(24).cuda()
        for idx, (imgs, labels) in enumerate(val_loader):
            if epoch == 0 and idx <= 5 :
                ori_imgs=UnNormalize(mean =[0.4743, 0.3539, 0.3249],std = [0.2697, 0.2238, 0.2154])(imgs)
                torchvision.utils.save_image(ori_imgs[0],'images/val_image_{:}.jpg'.format(idx),normalize=True)
            # imgs = imgs.cuda()
            # labels = labels.cuda()
            # bs = labels.shape[0]
            # output = model(imgs)
            # loss_val = criterion(100*output, 100*labels)
            # loss.update(loss_val.item(), bs)
            # batch_time.update(time.time() - batch_start_time)
            # batch_start_time = time.time()

            # mae_array = torch.mean(torch.abs(labels-output), dim=1)  #batchsize个样本的每个AU的平均绝对值损失
            # mae_value = torch.mean(mae_array)   #每个样本在每个AU上的平均绝对值损失
            # predict_true_array = torch.where(mae_array < 0.08, 1, 0)
            # acc_value = torch.sum(predict_true_array) / bs

            # mae.update(mae_value, bs)
            # acc.update(acc_value, bs)
            # # global_steps = writer_dict['train_global_steps']
            # if idx % args.print_fq == 0 or idx + 1 == len(val_loader):
            #     msg = 'Epoch: [{0}][{1}/{2}]\t' \
            #           'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
            #           'Speed {speed:.1f} samples/s\t' \
            #           'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
            #           'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
            #           'MAE {mae.val:.5f} ({mae.avg:.5f})\t' \
            #           'ACC {acc.val:.5f} ({acc.avg:.5f})\t' .format (
            #         epoch, idx, len (val_loader), batch_time=batch_time,
            #         speed=bs / batch_time.val,
            #         data_time=data_time, loss=loss, mae=mae,acc=acc)
            #     logger.info (msg)



            # val_loss += loss_val
            # mae = torch.mean(torch.abs(labels-output), dim=1)
            # acc_array = torch.where(mae < 0.08, 1, 0)

            # labels = labels > 0
            # output = output > 0
            # true_positives += torch.sum(labels & output, dim=0).cuda()
            # predicted_positives += torch.sum(output, dim=0).cuda()
            # actual_positives += torch.sum(labels, dim=0).cuda()
        # precision = true_positives / (predicted_positives + 1e-7)
        # recall = true_positives / (actual_positives + 1e-7)
        # f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        # average_f1 = f1.mean()
        # acc = right_preds / num_data
        # val_loss = val_loss / len(val_loader)
        # print("epochs: %d, val_loss:  %.5f, average_acc: %.5f"%(epoch, val_loss, acc))
        # print("f1:", f1)
        #print("average acc: ", torch.mean(acc))



        #return loss.avg, acc.avg, mae.avg