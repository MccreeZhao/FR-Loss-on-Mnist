from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torchvision import datasets, transforms
import numpy as np
from datetime import datetime

def visualize(feat, labels, epoch,writer,name):
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
        ax.text(feat[labels == i, 0].mean(), feat[labels == i, 1].mean(), str(i), color='black', fontsize=12)
    ax.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    ax.text(0, 0, "epoch=%d" % epoch)
    canvas.draw()

    # if (os.path.exists(imgDir)):
    #     pass
    # else:
    #     os.makedirs(imgDir)
    # fig.savefig(imgDir + '/epoch=%d.jpg' % epoch)
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    tt = transforms.ToTensor()
    timg = tt(img)
    timg.unsqueeze(0)
    writer.add_image(name, timg, epoch)

def visualize_center(centers,feat, labels, epoch,writer,name):
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
        ax.text(centers[i, 0], centers[i, 1], 'c' + str(i), color='black', fontsize=12)
        #ax.text(feat[labels == i, 0].mean(), feat[labels == i, 1].mean(), str(i), color='black', fontsize=12)
    ax.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    ax.text(0, 0, "epoch=%d" % epoch)
    canvas.draw()

    # if (os.path.exists(imgDir)):
    #     pass
    # else:
    #     os.makedirs(imgDir)
    # fig.savefig(imgDir + '/epoch=%d.jpg' % epoch)
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    tt = transforms.ToTensor()
    timg = tt(img)
    timg.unsqueeze(0)
    writer.add_image(name, timg, epoch)


def visualize_cos(feat, labels, epoch,writer,head,name):
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
    # fig.savefig(imgDir + '/cos_epoch=%d.jpg' % epoch)
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    tt = transforms.ToTensor()
    timg = tt(img)
    timg.unsqueeze(0)
    writer.add_image(name+'_cos', timg, epoch)

def get_time():
    return (str(datetime.now())[:-10]).replace(' ','-').replace(':','-')
