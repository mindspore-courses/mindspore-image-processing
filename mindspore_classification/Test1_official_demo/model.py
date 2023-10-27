'''lenet模型'''
import mindspore.nn as nn


class LeNet(nn.Cell):
    '''LeNet'''

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, has_bias=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, has_bias=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Dense(32*5*5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)
        self.re = nn.ReLU()

    def construct(self, x):
        '''LeNet construct'''
        x = self.re(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = self.re(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = self.re(self.fc1(x))      # output(120)
        x = self.re(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x
