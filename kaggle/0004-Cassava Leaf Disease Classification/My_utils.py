import numpy as np
import torch
from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt
from IPython import display
import time
from torch import nn



# 使用svg格式来展示图片
def use_svg_display():
    return backend_inline.set_matplotlib_formats("svg")


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


# 查询机器上所有可用的GPU
def try_all_gpu():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else torch.device('cpu')


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
