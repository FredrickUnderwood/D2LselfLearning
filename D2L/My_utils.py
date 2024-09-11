import numpy as np
import torch
from torchvision import transforms
import torchvision
from torchvision.transforms import v2
from torch.utils import data
from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt
from IPython import display
import time
from torch import nn
from torch.nn import functional as F
import os
import collections
import random
import re
import math


# 生成人工数据集
def generate_synthetic_data(w, b, num_examples):
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))


# 线性回归
def LinearRegression(X, w, b):
    return torch.matmul(X, w) + b


# 随机梯度下降优化
def sgd(params, learning_rate, batch_size):
    with torch.no_grad():
        for param in params:
            param -= learning_rate * param.grad / batch_size
            param.grad.zero_()


# 平方损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 使用svg格式来展示图片
def use_svg_display():
    return backend_inline.set_matplotlib_formats("svg")


# 从FashionMNIST数据集中读取数据
def load_data_from_fashion_MNIST(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))  # 用于调整图片大小
    trans = transforms.Compose(trans)  # compose用于将多个transform组合操作
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)
    workers_num = 4
    return data.DataLoader(train_set, batch_size, shuffle=True, num_workers=workers_num), \
        data.DataLoader(test_set, batch_size, shuffle=False, num_workers=workers_num)


# 画图，包括图本身和图的label
def show_image(imgs, row_num, col_num, titles=None, scale=1.5):
    figsize = (col_num * scale, row_num * scale)
    _, axes = plt.subplots(row_num, col_num, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):  # i是图片的索引
        if torch.is_tensor(img):
            ax.imshow(img.detach().numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)  # 将坐标轴设为不可见
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


# 累加器类型
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, item):
        return self.data[item]


# 用于设置坐标轴
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


# 展示正确率、错误率随训练轮数改变的曲线 这是一个动画类型！！！
class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
                 xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]  # 转换为列表用以统一数据结构
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim,
                                            xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):  # x,y包含多个数据点
        if not hasattr(y, '__len__'):  # 判断y是否是一个可迭代类型（list等）
            y = [y]
        n = len(y)
        if not hasattr(x, '__len__'):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):  # 将要绘制的点的集合传入类中
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()  # 清除该图上之前的内容，每次传进来的(x,y)是全体的，每一轮绘制从头绘制到最后一个点
        for x, y, fmts in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmts)  # 画图过程
        self.config_axes()  # 配置axe的相关参数
        display.display(self.fig)  # 显示绘制的图片
        display.clear_output(wait=True)  # 清除notebook中的输出内容


# 用于计数预测对的数量（多分类）
def count_accurate(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    correct_count = 0
    for i in range(len(y_hat)):
        if y_hat[i].type(y.dtype) == y[i]:
            correct_count += 1
    return float(correct_count)


# 计算模型的准确率
def calc_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()  # 进入评估模式
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(count_accurate(net(X), y), y.numel())
    return metric[0] / metric[1]


# Softmax回归的训练方法
def train_epoch_ch3(train_iter, net, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 用pytorch内置的优化器来优化
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])  # 第0维的数量就是batch_size
    metric.add(l.sum(), count_accurate(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


# Softmax回归的训练方法
def train_ch3(train_epochs, test_iter, train_iter, net, loss, updater):
    animator = Animator(xlabel='epoch', xlim=[1, train_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(train_epochs):
        train_metrics = train_epoch_ch3(train_iter, net, loss, updater)
        test_acc = calc_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics


# 生成pytorch数据迭代器
def load_array(data_arrays, batch_size, is_Train=True):
    data_set = data.TensorDataset(*data_arrays)  # 如果传入的data_arrays是feature和label的集合，*可以进行解包操作；TensorDataset则类似一个zip
    return data.DataLoader(data_set, batch_size, shuffle=is_Train)


# 用于计算模型损失率的函数
def evaluate_loss(net, data_iter, loss):
    metric = Accumulator(2)
    for X, y in data_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        metric.add(l.sum(), y.numel())
    return metric[0] / metric[1]


# 绘制非动画的静态曲线的函数
def plot(X, Y=None, xlabel=None, ylabel=None, xlim=None, ylim=None, legend=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    if legend is None:
        legend = []
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize
    axes = axes if axes else plt.gca()

    # 判断一个变量是否只有一维
    def has_one_axis(x):
        return hasattr(x, 'ndim') and x.ndim == 1 or isinstance(x, list) and not hasattr(x[0], '__len__')

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


# 查询GPU是否存在
def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    else:
        return torch.device('cpu')


# 查询机器上所有可用的GPU
def try_all_gpu():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else torch.device('cpu')


# 二维互相关运算 == 卷积运算
def corr2D(X, kernel):
    h, w = kernel.shape
    Y = torch.zeros(size=(X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * kernel).sum()
    return Y


# k折交叉验证
def get_K_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx, :]
        if j == i:  # 选取作为验证集
            X_valid, y_valid = X_part, y_part
        elif X_train is None:  # 训练集的初始化
            X_train, y_train = X_part, y_part
        else:  # 训练集的添加
            X_train = torch.cat((X_train, X_part), 0)
            y_train = torch.cat((y_train, y_part), 0)
    return X_train, y_train, X_valid, y_valid  # 返回训练集和验证集


# 记录多任务训练时间
class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return (sum(self.times)) / (len(self.times))

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


# GPU学习函数（基于对之前该库中函数的一些修改）


# GPU计算准确率
def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(count_accurate(net(X), y), y.numel())
    return metric[0] / metric[1]


# GPU训练函数
def train_gpu(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weight(m):
        if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)

    net.apply(init_weight)
    print('training on:', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', legend=['train_loss', 'train_acc', 'test_acc'], xlim=[1, num_epochs])
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], count_accurate(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter, device)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


# 生成VGG块的函数
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))  # 加入若干个不改变原图像大小的卷积层
        layers.append(torch.nn.ReLU())
        in_channels = out_channels
    layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
    return torch.nn.Sequential(*layers)


# 生成NiN块的函数
def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),  # 输入通道、输出通道、卷积核大小、步长、补全
        torch.nn.ReLU(), torch.nn.Conv2d(out_channels, out_channels, kernel_size=1),
        torch.nn.ReLU(), torch.nn.Conv2d(out_channels, out_channels, kernel_size=1),
        torch.nn.ReLU()
    )


# GoogLeNet Inception构建类
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):  # c1-c4是四条线路的输出通道的参数
        super(Inception, self).__init__(**kwargs)
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)  # 线路1，1*1的卷积核
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)  # 线路2，1*1的卷积核连接3*3的卷积核，不改变图片尺寸
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)  # 线路3，1*1的卷积核连接5*5的卷积核，也不改变图片尺寸
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 线路4，3*3的最大池化层拼接1*1的卷积核
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, X):
        p1 = F.relu(self.p1_1(X))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(X))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(X))))
        p4 = F.relu(self.p4_2(self.p4_1(X)))
        return torch.cat((p1, p2, p3, p4), dim=1)  # 在列上做扩展


# 实现批量归一化的函数
def batch_norm(x, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():  # 如果不在训练模式，而是预测模式
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(x.shape) in (2, 4)  # 对应全连接层和二维卷积层，tensor的维度是2或者4
        if len(x.shape) == 2:  # 全连接层的情况
            mean = x.mean(dim=0)
            var = ((x - mean) ** 2).mean(dim=0)
        else:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)  # 对每个通道求均值，keepdim后得到的是一个1*n*1*1的tensor
            var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + eps)
        # 更新移动平均值，相关内容：https://zhuanlan.zhihu.com/p/151786842
        moving_mean = momentum * moving_mean + (1 - momentum) * mean
        moving_var = momentum * moving_var + (1 - momentum) * var
    y = gamma * x_hat + beta
    return y, moving_mean.data, moving_var.data


# 实现批量归一化层
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        """
        :param num_features: 表示全连接层的输出变量个数，或者卷积层的输出通道数
        :param num_dims: 表示维度，对应全连接层和二维卷积层，tensor的维度是2或者4
        """
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))  # nn.Parameter()可参与求参数
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, self.moving_mean, self.moving_var,
                                                          eps=1e-5, momentum=0.9)
        return Y


# ResNet实现
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1,
                               stride=strides)  # VGG中不改变图片大小的卷积层
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        """
        convolution_1 -> batch_norm_1 -> ReLU_Activation -> convolution_2 -> batch_norm_2 -> Residual_layer -> ReLU_Activation
        :param X:
        :return:
        """
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


# 生成残差块的函数
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))  # 图片大小减半、通道数翻倍
        else:
            blk.append(Residual(num_channels, num_channels))  # 通道数和图片大小均不变
    return blk


# 生成ResNet18的函数
def resnet18(num_classes, in_channels=1):
    """
    :param num_classes: 图片分类的分类种类数
    :param in_channels:
    :return: ResNet网络
    """
    net = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(64), nn.ReLU())

    net.add_module('ResNet_Block1', nn.Sequential(*resnet_block(64, 64, 2, first_block=True)))
    net.add_module('ResNet_Block2', nn.Sequential(*resnet_block(64, 128, 2)))
    net.add_module('ResNet_Block3', nn.Sequential(*resnet_block(128, 256, 2)))
    net.add_module('ResNet_Block4', nn.Sequential(*resnet_block(256, 512, 2)))
    net.add_module('global_avg_pool', nn.AdaptiveAvgPool2d((1, 1)))
    net.add_module('fully_connected', nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))
    return net


# 设置子图的格式
def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


# 从CIFAR10数据集中读取数据的函数
def load_data_from_CIFAR10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root='./data', train=is_train, transform=augs, download=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=4)
    return dataloader


# 多GPU mini-batch训练函数
def train_batch_gpus(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()  # 表示模型进入训练模式
    trainer.zero_grad()  # 梯度清零
    pred = net(X)  # 计算y_hat
    l = loss(pred, y)  # 计算损失
    l.sum().backward()  # 对损失求梯度
    trainer.step()  # 用梯度更新模型参数
    train_loss = l.sum()
    train_acc_sum = count_accurate(pred, y)
    return train_loss, train_acc_sum


# 多GPU训练函数
def train_gpus(net, train_iter, test_iter, loss, trainer, num_epochs, devices=try_all_gpu()):
    timer, num_batches = Timer(), len(train_iter)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 5.0],
                        legend=['train_loss', 'train_acc', 'test_acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    best_test_acc = 0
    for epoch in range(num_epochs):
        metric = Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_gpus(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            """
            如果是分类问题，一个样本只有一个标签，这种情况下labels.shape[0]和labels.numel()没有区别
            但是如果是多多标签分类问题，即一个样本有多个标签，那numel就会多于shape[0]
            """
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[3], None))
                """
                计算平均损失率：总的损失/样本个数
                计算训练正确率：正确的预测标签个数/总的预测标签个数
                """
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        if test_acc > best_test_acc:
            torch.save(net.state_dict(), './pre_res_model.ckpt')
            best_test_acc = test_acc
    print(f'loss {metric[0] / metric[2]:.3f}, train_acc {metric[1] / metric[3]:.3f}, test_acc {test_acc:.3f}')
    print((f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on ', f'{str(devices)}'))


# 使用预训练参数进行训练的函数
def train_fine_tuning(net, train_iter, test_iter, trainer, num_epochs, learning_rate, model_path='./pre_res_model.ckpt',
                      param_group=True,
                      devices=try_all_gpu()):
    loss = nn.CrossEntropyLoss(reduction='none')
    if param_group:
        param_1x = [param for name, param in net.named_parameters() if
                    name not in ['fc.weight', 'fc.bias']]  # net.named_parameters()返回每一层的变量名和变量的值
        trainer = trainer([{'params': param_1x},
                           {'params': net.fc.parameters(),
                            'lr': learning_rate * 10}],
                          lr=learning_rate, weight_decay=0.001)
    else:
        trainer = trainer(net.parameters(), lr=learning_rate, weight_decay=0.001)
    train_gpus(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


# 物体检测与目标检测


# 框坐标从（左上x，左上y，右下x，右下y）转为（中心x，中心y，宽度，高度）
# 由于y轴从上到下增大，因此左上角的值对应的y值更小
def box_corner_to_center(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((center_x, center_y, w, h), axis=-1)
    return boxes


# 框坐标从（中心x，中心y，宽度，高度）转为（左上x，左上y，右下x，右下y）
def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    center_x, center_y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = center_x - 0.5 * w
    y1 = center_y - 0.5 * h
    x2 = center_x + 0.5 * w
    y2 = center_y + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes


# 绘制框的函数
def bbox_to_rect(bbox, color):
    # 左上x，左上y，宽，高
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2)  # plt.Rectangle 的xy对应的是矩形的左上角坐标


# 在原始图像上绘制框的函数
def show_bbox(axes, bboxes, labels=None, colors=None):
    """
    :param axes: 多个子图的合集
    :param bboxes: 边框坐标的合集
    :param labels:
    :param color:
    :return:
    """

    def make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (tuple, list)):
            obj = [obj]
        return obj

    labels = make_list(labels)
    colors = make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.cpu().detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center', ha='center',
                      fontsize=9, color=text_color, bbox=dict(facecolor=color, lw=0))


# 生成锚框的函数，在特征图上生成，因此输入的data是特征图
def anchor_boxes(data, sizes, ratios):
    img_height, img_width = data.shape[-2:]  # 4, 4
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 锚框的中心保持在像素的中心，因此设置一个偏移量0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / img_height
    steps_w = 1.0 / img_width

    # 生成中心坐标
    center_h = (torch.arange(img_height, device=device) + offset_h) * steps_h  # 0.125 0.375 0.625 0.875
    center_w = (torch.arange(img_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid([center_h, center_w], indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)  # 方便后续匹配x和y

    # 生成boxes_per_pixel个高和宽
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[
                                         1:]))) * img_height / img_width  # 理解不了看这个https://fkjkkll.github.io/2021/11/23/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8BSSD/#more
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))

    # 获得半高或者半高
    anchor_manipulations = torch.stack((-w, -h, w, h)). \
                               T.repeat(img_height * img_width, 1) / 2  # repeat(行上扩展的个数,列上扩展的个数)
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1). \
        repeat_interleave(boxes_per_pixel, dim=0)
    outputs = out_grid + anchor_manipulations
    return outputs.unsqueeze(0)


# 计算两个锚框或边界框列表中成对的交并比
def box_iou(boxes1, boxes2):
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    upper_left_coordinate = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    """
    假设 `boxes1` 的形状为 (N, 4)，表示 N 个边界框，每个框有 4 个坐标；`boxes2` 的形状为 (M, 4)，表示 M 个边界框。
    为了比较 `boxes1` 中的每个框与 `boxes2` 中的每个框，我们需要一个形状为 (N, M, 4) 的输出张量。
    通过增加一个维度，例如执行 `boxes1[:, None, :]`，我们将 `boxes1` 的形状从 (N, 4) 更改为 (N, 1, 4)。
    由于 `boxes2` 的形状为 (M, 4)，当我们应用广播规则时，`boxes1` 和 `boxes2` 都将被视为形状 (N, M, 4)。
    这样，我们可以在这两组框之间进行元素级的操作，例如寻找交集的左上角和右下角坐标。
    """
    lower_right_coordinate = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (lower_right_coordinate - upper_left_coordinate).clamp(min=0)
    """
    如果有交集，lower_right_coordinate的横纵坐标一定大于upper_left_coordinate的横纵坐标
    clamp()函数将两坐标相减后小于0（没交集）的部分置为0
    """
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = area1[:, None] + area2 - inter_areas
    return inter_areas / union_areas


# 在数据集中将ground-truth bbox的标记框分配给锚框
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    jaccard = box_iou(anchors, ground_truth)
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    max_ious, indices = torch.max(jaccard, dim=1)  # 第1维是ground-truth的维度，所以选出每个ground-truth对应的iou最大的anchor
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)  # nonzero返回tensor中非零的坐标
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)  # 清空一个列要N行个-1
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # 返回这个tensor的最大值是第几个
        box_idx = (max_idx % num_gt_boxes).long()  # 纵坐标
        anc_idx = (max_idx / num_gt_boxes).long()  # 横坐标
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


# 计算锚框和ground-truth的偏移量
def offset_boxes(anchors, assigned_bboxes, eps=1e-6):
    c_anc = box_corner_to_center(anchors)
    c_assigned_bboxes = box_corner_to_center(assigned_bboxes)
    offset_xy = 10 * (c_assigned_bboxes[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bboxes[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset


# 给锚框标记label，生成具体的偏移量、mask和label
def multi_box_labels(anchors, labels):
    # labels是一个类似迭代器的，由多个batch，每个batch里多个label组成
    # labels的第三维第一个值代表的是该bbox对应的类型，后四个值代表的是该bbox的坐标
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)  # 如果第0维的size是1，将第0维去掉
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]  # 取出第i个batch的labels
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)  # label第2维的第0个参数是分类的类别，即object的索引值
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)  # 第一维数量不变，第二维数量变为四倍
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bboxes = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)

        indices_true = torch.nonzero(anchors_bbox_map >= 0)  # 返回值大于等于0的index
        bboxes_idx = anchors_bbox_map[indices_true]  # 给分配了的锚框，分配的ground-truth标记框
        class_labels[indices_true] = label[bboxes_idx, 0].long() + 1  # 取出对应的object的索引值
        assigned_bboxes[indices_true] = label[bboxes_idx, 1:]  # 取出对应的ground-truth框的坐标
        offset = offset_boxes(anchors, assigned_bboxes) * bbox_mask  # 为0的算作背景，offset就为0

        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)


# 通过锚框和锚框的偏移量来预测边界框
def offset_inverse(anchors, offset_pred):
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_pred[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_pred[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat([pred_bbox_xy, pred_bbox_wh], axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox


# 对边界框置信度进行排序，以及非极大抑制的实现
def nms(boxes, scores, iou_threshold):
    # boxes中的内容已经是分配过且不为背景的边框了
    # scores每个列对应一个锚框，每一行对应的是某个label的预测
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel == 1:
            break
        iou = box_iou(boxes[i, :].reshape(-1, 4), boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        indices = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[indices + 1]  # B[1:]开始计数，因此index要+1
    return torch.tensor(keep, device=boxes.device)


# 使用非极大抑制来预测边界框
def multi_box_predictions(class_probs, offset_preds, anchors, nms_threshold=0.5, pos_threshold=0.009999999):
    device, batch_size = class_probs.device, class_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = class_probs.shape[1], class_probs.shape[2]
    """
    class_prob这个tensor每一行代表一个类别，每一列是每个anchor对应是该类的概率 
    """
    out = []
    for i in range(batch_size):
        class_prob, offset_pred = class_probs[i], offset_preds[i].reshape(-1, 4)  # 取出第i个batch
        conf, class_id = torch.max(class_prob[1:], 0)
        """
        [1:]用来屏蔽背景类，conf接收的是每个anchor预测率最高的概率，class_id接收的是对应的行号，也就是对应的类别id
        """
        predicted_bbox = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bbox, conf, nms_threshold)  # 通过nms筛选出需要保留的anchors

        # 对id进行排序，将不保留的anchors的idx排到后面
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))

        class_id[non_keep] = -1  # 将不保留的anchors替换成背景类
        class_id = class_id[all_id_sorted]  # 对class_id进行重新排序
        conf, predicted_bbox = conf[all_id_sorted], predicted_bbox[all_id_sorted]  # 重新排序
        below_min_idx = (conf < pos_threshold)  # 对非背景类进行检查，如果低于阈值也标为背景
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bbox), dim=1)
        out.append(pred_info)
    return out


# 语义分割


# voc对应RGB值
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

# voc类别
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


# 读取voc图像并标注
def read_voc_images(voc_dir, is_train=True):
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(voc_dir, 'SegmentationClass', f'{fname}.png'), mode))
    return features, labels


# 构建RGB到voc类别索引的映射
def voc_colormap2label():
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label


# 将RGB值映射到voc类别索引
def voc_label_indices(colormap, colormap2label):
    # colormap的值就是一张图上每个pixel对应的RGB值
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]


# 随机裁剪特征和标签图片
def voc_and_crop(feature, label, height, width):
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
    feature = v2.functional.crop(feature, *rect)
    label = v2.functional.crop(label, *rect)
    return feature, label


# voc数据集
class VOCSegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train)
        self.features = [self.normalize_image(feature) for feature in self.filter(features
                                                                                  )]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()

        print(f'read {str(len(self.features))} examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (img.shape[1] >= self.crop_size[0] and img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_and_crop(self.features[idx], self.labels[idx], *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)


# 语言模型


# 将一行一行的文字转成一个一个单词
def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token:', token)


# 统计每个token出现的频率
def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


# 文本语料库
class Vocab:
    def __init__(self, tokens=None, min_freq=0, reversed_tokens=None):
        if tokens is None:
            tokens = []
        if reversed_tokens is None:
            reversed_tokens = []
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reversed_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        # 返回的是该vocab词表的词汇有多少种，即后续的vocab_size
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    @property
    def unk(self):  # 返回 unknown token 的idx
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


# 使用随机抽样生成文本的mini-batch
def seq_data_iter_random(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0, num_steps - 1)]  # 一个随机的起始点来生成subseqs
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))  # initial_indices存了起点的pos
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        x = [data(j) for j in initial_indices_per_batch]
        y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(x), torch.tensor(y)


# 使用顺序分区生成文本的mini-batch
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    xs = torch.tensor(corpus[offset: offset + num_tokens])
    ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    xs, ys = xs.reshape(batch_size, -1), ys.reshape(batch_size, -1)
    num_batches = xs.shape[1] // num_steps
    for i in range(0, num_batches * batch_size, batch_size):
        x = xs[:, i: i + num_steps]
        y = ys[:, i: i + num_steps]
        yield x, y


# 读取time-machine
def read_time_machine():
    with open('./data/time_machine.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


# load time-machine数据
def load_corpus_time_machine(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


# 构建文本数据集
class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


# 生成文本数据迭代器
def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


# RNN初始化参数
def get_params(vocab_size, num_hiddens, device):
    # 输入和输出采用相同的词表，vocab_size是使用的词表的大小
    # 所以这实际上是一个多分类的问题
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    w_xh = normal((num_inputs, num_hiddens))
    w_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    w_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [w_xh, w_hh, b_h, w_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


# RNN初始化函数
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


# RNN前向传播函数
def rnn(inputs, state, params):
    w_xh, w_hh, b_h, w_hq, b_q = params
    h, = state
    outputs = []
    for x in inputs:
        h = torch.tanh(torch.mm(x, w_xh) + torch.mm(h, w_hh)+ b_h)  # h: batch_size * num_hiddens
        y = torch.mm(h, w_hq) + b_q
        outputs.append(y)
    # outputs是一个存tensor的list
    return torch.cat(outputs, dim=0), (h,)


# RNN模型
class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)  # 用于初始化模型参数
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, x, state):  # x: batch_size * num_steps
        x = F.one_hot(x.T, self.vocab_size).type(torch.float32)  # (35, 32, 28)
        return self.forward_fn(x, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


# 序列模型预测
def predict_seq(prefix, num_preds, net, vocab, device):
    # 生成prefix之后的字符
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]  # text -> num
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # 将prefix读入outputs中
    for y in prefix[1: ]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


# 梯度剪裁
def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# 序列模型单epoch训练
def train_epoch_seq(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, Timer()
    metric = Accumulator(2)  # 训练损失之和，词语元数量
    for x,y in train_iter:
        if state is None or use_random_iter:  # use_random_iter说明seq不连续
            state = net.begin_state(batch_size=x.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y = y.T.reshape(-1)  # y.T: num_steps * batch_size
        x, y = x.to(device), y.to(device)
        y_hat, state = net(x, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


# 序列模型训练
def train_seq(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr=lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_seq(prefix, 50, net, vocab, device)
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_seq(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'perplexity: {ppl:.1f}, speed: {speed:.1f} words/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))