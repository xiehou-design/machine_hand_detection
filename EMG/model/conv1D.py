import torch
from torch import nn
from torch.nn import *


class Model(nn.Module):
    def __init__(self, classes=16):
        super(Model, self).__init__()
        self.classes = classes
        self.conv_1 = Conv2d(in_channels=1, out_channels=32, kernel_size=(19, 1), stride=(1, 1), padding=(9, 0))
        self.max_pool_1 = MaxPool2d(kernel_size=(10, 1), stride=(10, 1))

        self.conv_2 = Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
        self.max_pool_2 = MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv_3 = Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.max_pool_3 = MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.relu_activate = ReLU()

        self.flatten = Flatten()

        self.dropout = Dropout(p=0.5)
        self.linear_1 = Linear(in_features=640, out_features=128)
        self.linear_2 = Linear(in_features=128, out_features=self.classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_activate(x)
        x = self.max_pool_1(x)
        x = self.conv_2(x)
        x = self.relu_activate(x)
        x = self.max_pool_2(x)
        x = self.conv_3(x)
        x = self.relu_activate(x)
        x = self.max_pool_3(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear_1(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


if __name__ == '__main__':
    model = Model(classes=16)
    x = torch.ones((2, 6, 200, 1))
    print(model)
    y = model(x)
    print(y)
