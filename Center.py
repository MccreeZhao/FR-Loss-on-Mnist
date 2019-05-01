# -*- coding: utf-8 -*-
import torch
import Net
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from torch.nn import DataParallel
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

plt.switch_backend('agg')
import pickle
import time
import os
from tqdm import tqdm

logDir = './log_center'
imgDir = './Center'

from tensorboardX import SummaryWriter

if os.path.exists(logDir):
    pass
else:
    os.mkdir(logDir)
writer = SummaryWriter(logDir)

batch_size = 96
learning_rate = 1e-2
num_epoches = 20
dataTransform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])
     ]
)

train_dataset = datasets.MNIST(
    root='./MNIST_data', train=True, transform=dataTransform, download=True
)
test_dataset = datasets.MNIST(
    root='./MNIST_data', train=False, transform=dataTransform, download=True
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

activation = {}

EbeddingNet = Net.EModel()
EbeddingNet.cuda()
head = Net.Linear()
head.cuda()

myCrossEntropyLoss = nn.CrossEntropyLoss()

lamda = 0.2
mycenterloss = Net.centerloss()

def visualize(centers,feat, labels, epoch):
    colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
              '#ff00ff', '#990000', '#999900', '#009900', '#009999']

    fig = Figure(figsize=(6, 6), dpi=100)
    fig.clf()
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    feat = feat.data.cpu().numpy()
    labels = labels.data.cpu().numpy()

    for i in range(10):
        ax.scatter(feat[labels == i, 0], feat[labels == i, 1], c=colors[i], s=1)
        ax.text(centers[i, 0], centers[i, 1], 'c'+str(i), color='black', fontsize=12)
    ax.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    ax.text(0, 0, "epoch=%d" % epoch)
    canvas.draw()

    if (os.path.exists(imgDir)):
        pass
    else:
        os.makedirs(imgDir)
    fig.savefig(imgDir + '/epoch=%d.jpg' % epoch)
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    tt = transforms.ToTensor()
    timg = tt(img)
    timg.unsqueeze(0)
    writer.add_image('center', timg, epoch)


def visualize_cos(feat, labels, epoch):
    colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
              '#ff00ff', '#990000', '#999900', '#009900', '#009999']

    fig = Figure(figsize=(6, 6), dpi=100)
    fig.clf()
    canvas = FigureCanvas(fig)
    ax = fig.gca()


    feat = feat / feat.norm(2, 1).unsqueeze(1).repeat(1, 2)
    feat = feat.data.cpu().numpy()
    labels = labels.data.cpu().numpy()

    weight = head.state_dict()['weight'].t()
    weight = weight / weight.norm(2, 1).unsqueeze(1).repeat(1, 2)
    weight = weight.data.cpu().numpy()

    for i in range(10):
        ax.scatter(feat[labels == i, 0], feat[labels == i, 1], c=colors[i], s=1)
        ax.text(feat[labels == i, 0].mean(), feat[labels == i, 1].mean(), str(i), color='black', fontsize=12)
        ax.plot([0,weight[i][0]],[0,weight[i][1]],linewidth=2,color=colors[i])
    ax.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    ax.text(0, 0, "epoch=%d" % epoch)
    canvas.draw()
    fig.savefig(imgDir + '/cos_epoch=%d.jpg' % epoch)
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    tt = transforms.ToTensor()
    timg = tt(img)
    timg.unsqueeze(0)
    writer.add_image('center_cos', timg, epoch)


def train_mnist(epoch, optimizer):
    feature_list = []
    label_list = []
    train_loss_nn = 0
    train_loss_center = 0
    train_acc = 0
    EbeddingNet.train()  # 切换到训练模式

    for images, label in train_loader:

        images = Variable(images).cuda()
        label = Variable(label).cuda()
        # 前向传播
        feature = EbeddingNet(images, epoch)
        out = head(feature)

        feature_list.append(feature)
        label_list.append(label)
        loss_nn = myCrossEntropyLoss(out, label)
        loss_center = lamda * mycenterloss(feature, label)
        loss = loss_nn + loss_center
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        for param in mycenterloss.parameters():  # 保证learning rate合适，不随lamda而变
            param.grad.data *= (1. / lamda)
        optimizer.step()
        # 记录误差
        train_loss_nn += loss_nn.item()  #############data[0]->item()
        train_loss_center += loss_center.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = float(num_correct) / images.shape[0]
        train_acc += acc

    feats = torch.cat(feature_list, 0)
    labels = torch.cat(label_list, 0)
    visualize(mycenterloss.center, feats, labels, epoch)
    visualize_cos(feats, labels, epoch)
    return train_loss_nn, train_loss_center, train_acc


def tes_mnist(epoch):
    # 在测试集上检验效果
    with torch.no_grad():
        eval_loss = 0
        eval_acc = 0
        EbeddingNet.eval()  # 将模型改为预测模式
        with torch.no_grad():
            for images, label in test_loader:
                images = images.cuda()
                label = label.cuda()
                feature = EbeddingNet(images, epoch)
                out = head(feature)
                loss = myCrossEntropyLoss(out, label)
                # 记录误差
                eval_loss += loss.item()
                # 记录准确率
                _, pred = out.max(1)
                num_correct = (pred == label).sum().item()
                acc = float(num_correct) / images.shape[0]
                eval_acc += acc
    return eval_acc, eval_loss

def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 5

def main():
    # 开始训练
    torch.manual_seed(1)
    optimizer = optim.Adam([
        {'params':EbeddingNet.parameters(), 'lr':1e-3},
        {'params': head.parameters(), 'lr': 1e-3},
        {'params':mycenterloss.parameters(), 'lr': 0.5},
    ])
    milestone = [20, 40, 60]
    for epoch in tqdm(range(80)):
        if epoch in milestone:
            schedule_lr(optimizer)
        train_loss_nn, train_loss_center, train_acc = train_mnist(epoch,optimizer)
        eval_acc, eval_loss = tes_mnist(epoch)

        writer.add_scalar('Train Loss_nn', train_loss_nn / len(train_loader), epoch)
        writer.add_scalar('Train Loss_center', train_loss_center / len(train_loader), epoch)
        writer.add_scalar('Train Acc', train_acc / len(train_loader), epoch)
        writer.add_scalar('eval Loss', eval_loss / len(test_loader), epoch)
        writer.add_scalar('eval acc', eval_acc / len(test_loader), epoch)
        print(
            'epoch: {}, Train Loss_nn: {:.6f}, Train Loss_center: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
                .format(epoch, train_loss_nn / len(train_loader), train_loss_center / len(train_loader),
                        train_acc / len(train_loader),
                        eval_loss / len(test_loader), eval_acc / len(test_loader)))
    if not os.path.exists('./save'):
        os.mkdir('./save')
    torch.save(EbeddingNet, './save/center.pkl')


if __name__ == '__main__':
    main()
