# -*- coding: utf-8 -*-
import torch
import Net
import torch.nn as nn
import torch.optim as optim
from data_pipe import get_test_data, get_train_data
import matplotlib.pyplot as plt
from utils import visualize, visualize_cos, get_time,visualize_center
import random
import numpy as np
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter

plt.switch_backend('agg')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(17)

name = 'QAMFace'  ## choose loss type
assert name in ['raw', 'CenterFace', 'NormFace', 'SphereFace', 'ArcFace', 'QAMFace']
## Set save path
logDir = os.path.join('./log', name, get_time())
imgDir = name
saveDir = os.path.join('./save/', name)
os.makedirs(logDir, exist_ok=True)
os.makedirs(saveDir, exist_ok=True)

loss_Dict = {'raw': Net.Linear, 'CenterFace': Net.centerloss, 'NormFace': Net.NormFace, 'SphereFace': Net.SphereFace,
             'ArcFace': Net.ArcMarginProduct, 'QAMFace': Net.QAMFace}

## For tensorboard visualize
writer = SummaryWriter(logDir)

## Hyper parameters
batch_size = 96
learning_rate = 1e-2
num_epoches = 20

## Data Loader
train_loader = get_train_data(batch_size)
test_loader = get_test_data(batch_size)

## Init Model
EbeddingNet = Net.EModel()
EbeddingNet.cuda()
head = loss_Dict[name]()
head.cuda()

myCrossEntropyLoss = nn.CrossEntropyLoss()


def train_mnist(epoch, optimizer_nn):
    train_loss_nn = 0
    train_acc = 0
    EbeddingNet.train()  # 切换到训练模式
    for images, label in train_loader:
        images = images.cuda()
        label = label.cuda()
        # 前向传播
        feature = EbeddingNet(images, epoch)
        #out = head(feature)
        out,loss = head(feature,label)
        #loss = myCrossEntropyLoss(out, label)
        # 反向传播
        optimizer_nn.zero_grad()
        loss.backward()
        optimizer_nn.step()
        # 记录误差
        train_loss_nn += loss.item()  #############data[0]->item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = float(num_correct) / images.shape[0]
        train_acc += acc
    return train_loss_nn, train_acc


def tes_mnist(epoch):
    # 在测试集上检验效果
    feature_list = []
    label_list = []
    with torch.no_grad():
        eval_loss = 0
        eval_acc = 0
        EbeddingNet.eval()  # 将模型改为预测模式
        with torch.no_grad():
            for images, label in test_loader:
                images = images.cuda()
                label = label.cuda()
                feature = EbeddingNet(images, epoch)
                #out = head(feature)
                out, loss = head(feature,label)
                feature_list.append(feature)
                label_list.append(label)
                #loss = myCrossEntropyLoss(out, label)
                eval_loss += loss.item()
                _, pred = out.max(1)
                num_correct = (pred == label).sum().item()
                acc = float(num_correct) / images.shape[0]
                eval_acc += acc
    feats = torch.cat(feature_list, 0)
    labels = torch.cat(label_list, 0)
    if name=='CenterFace':
        visualize_center(head.center,feats,labels,epoch,writer,name)
    else:
        visualize(feats, labels, epoch, writer, name)
    visualize_cos(feats, labels, epoch, writer, head, name)
    return eval_acc, eval_loss


def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 5


def main():
    # 开始训练
    torch.manual_seed(1)
    optimizer_nn = optim.Adam([
        {'params': EbeddingNet.parameters(), 'lr': learning_rate},
        {'params': head.parameters(), 'lr': learning_rate},
    ])
    milestone = [20, 40, 60]
    for epoch in tqdm(range(80)):
        if epoch in milestone:
            schedule_lr(optimizer_nn)
        train_loss_nn, train_acc = train_mnist(epoch, optimizer_nn)
        eval_acc, eval_loss = tes_mnist(epoch)

        writer.add_scalar('Train Loss_nn', train_loss_nn / len(train_loader), epoch)
        writer.add_scalar('Train Acc', train_acc / len(train_loader), epoch)
        writer.add_scalar('eval Loss', eval_loss / len(test_loader), epoch)
        writer.add_scalar('eval acc', eval_acc / len(test_loader), epoch)
        print(
            'epoch: {}, Train Loss_nn: {:.6f},  Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f},'
                .format(epoch, train_loss_nn / len(train_loader),
                        train_acc / len(train_loader),
                        eval_loss / len(test_loader), eval_acc / len(test_loader)))

    torch.save(EbeddingNet, saveDir + '.pkl')


if __name__ == '__main__':
    main()
