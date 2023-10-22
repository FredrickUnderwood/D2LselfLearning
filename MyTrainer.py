import torch
from torch import nn
import tqdm
import os


class MyTrainer:
    @staticmethod
    def accurate_count(y_hat, y_true):
        y_hat = y_hat.argmax(axis=1)
        y_true = y_true.argmax(axis=1)
        correct_count = 0
        for i in range(len(y_hat)):
            if y_hat[i].type(y_true.dtype) == y_true[i]:
                correct_count += 1
        return float(correct_count)

    def __init__(self, optimizer, model, criterion, train_dataloader, valid_dataloader,
                 learning_rate, num_epochs, devices, out_classes, param_group=True):
        self.optimizer_class = optimizer
        self.model = model
        self.criterion = criterion
        self.devices = devices
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.param_group = param_group
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.out_classes = out_classes

    def calc_valid_acc(self):
        self.model.eval()
        device = next(iter(self.model.parameters())).device
        test_num = 0
        test_acc_num = 0
        for x, y_true in self.valid_dataloader:
            if isinstance(x, list):
                x = [x_1.to(device) for x_1 in x]
            else:
                x = x.to(device)
            y_true_tensor = torch.zeros(size=(len(y_true), self.out_classes))
            for i in range(len(y_true)):
                label = y_true[i]
                y_true_tensor[i, label] = 1
            y_true = y_true_tensor
            y_true = y_true.to(device)
            test_num += y_true.shape[0]
            test_acc_num += MyTrainer.accurate_count(self.model(x), y_true)
        return test_acc_num / test_num

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        train_num = 0
        train_acc_num = 0
        batch_num = len(self.train_dataloader)
        param_1x = [param for name, param in self.model.named_parameters() if
                    name not in ['module.classifier.weight', 'module.classifier.bias']]
        optimizer = self.optimizer_class([{'params': param_1x},
                                          {'params': self.model.module.classifier.parameters(),
                                           'lr': self.learning_rate(epoch) * 10}],
                                         lr=self.learning_rate(epoch), weight_decay=0.001)
        print(f'epoch{epoch + 1} begins:')
        tk0 = tqdm(enumerate(self.train_dataloader), total=batch_num)
        for batch_idx, (x, y_true) in tk0:
            y_true_tensor = torch.zeros(size=(len(y_true), self.out_classes))
            for i in range(len(y_true)):
                label = y_true[i]
                y_true_tensor[i, label] = 1
            y_true = y_true_tensor
            x, y_true = x.to(self.devices[0]), y_true.to(self.devices[0])
            optimizer.zero_grad()
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y_true)
            loss.sum().backward()
            optimizer.step()
            total_loss += loss.sum()
            train_num += y_true.shape[0]
            train_acc_num += MyTrainer.accurate_count(y_hat, y_true)

        return total_loss / train_num, train_acc_num / train_num

    def train(self):
        best_valid_acc = 0
        self.model = nn.DataParallel(self.model, device_ids=self.devices).to(self.devices[0])
        for epoch in range(self.num_epochs):
            train_loss, train_acc = MyTrainer.train_epoch(self, epoch)
            valid_acc = MyTrainer.calc_valid_acc(self)
            if valid_acc > best_valid_acc:
                torch.save(self.model.state_dict(), os.path.join('best_model.pth'))
            print(f'epoch{epoch + 1}:train_loss:{train_loss}, train_acc:{train_acc}, valid_acc:{valid_acc}')
