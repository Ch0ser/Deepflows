from tensor import *
from utils.data.dataloader import dataloader
from utils.visual import Animator
from utils import evaluate


class train_test(object):

    def __init__(self, net, loss, optimizer):
        self.net = net
        self.loss = loss
        self.optimizer = optimizer

    # 定义训练方法
    def train(self, train_loader, epochs):
        pass

    def _train(self, train_features, train_labels, batch_size, epochs, shuffle=True, log_rmse=False):

        data_iter = dataloader(batch_size, shuffle)

        animator = Animator(xlabel='epoch', ylabel='loss', legend=['train_loss'], xlim=[1, epochs])

        for epoch in range(epochs):
            self.net.train()
            for X, y in data_iter(train_features, train_labels):
                output = self.net.forward(X)
                l = self.loss(output, y)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

            self.net.eval()
            output = self.net.forward(train_features)

            if log_rmse:
                clipped_preds = np.clip(output.data, 1, float('inf'))
                clipped_preds = Tensor(clipped_preds, float)
                L = self.loss(log(clipped_preds), log(train_labels)).data

            else:
                L = self.loss(output, train_labels).data

            print(f'epoch {epoch + 1}, loss {L:f}')
            animator.add(epoch + 1, L.item())

    def test(self, test_loader):
        pass

    # 定义测试方法
    def _test(self, batch_size, test_features, test_labels, shuffle=True):

        data_iter = dataloader(batch_size, shuffle)
        batch_num = len(test_features) // batch_size + 1
        i = 0
        animator = Animator(xlabel='batch', ylabel='accuracy', xlim=[1, batch_num],
                            ylim=[0, 0.5], legend=['test_accuracy'])
        metric = evaluate.Accumulator(2)
        for X, y in data_iter(test_features, test_labels):
            pre = self.net.forward(X)
            metric.add(evaluate.accuracy(pre.data, y.data), len(y))
            test_acc = metric[0] / metric[1]

            animator.add(i + 1, test_acc)
            i += 1
            print(f'test_Acc {test_acc:f}')

    # 定义预测函数
    def predict(self, features):
        predictions = self.net.forward(features)
        return predictions