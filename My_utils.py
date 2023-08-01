import torch
from torchvision import transforms
import torchvision
from torch.utils import data
from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt
from IPython import display


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
