import numpy as np
import torch
from torchvision import transforms
import torchvision
from torch.utils import data
from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt
from IPython import display
import time
import numpy


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
    _, axes = plt.subplot(row_num, col_num, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):  # i是图片的索引
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
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
        self.axes[0].cla()  # 清楚该图上之前的内容，每次传进来的(x,y)是全体的，每一轮绘制从头绘制到最后一个点
        for x, y, fmts in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmts)  # 画图过程
        self.config_axes()  # 配置axe的相关参数
        display.display(self.fig)  # 显示绘制的图片
        display.clear_output(wait=True)  # 清楚notebook中的输出内容


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
    set_axes(axes, xlabel, y, xlim, ylim, xscale, yscale, legend)


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
