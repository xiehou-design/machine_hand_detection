from torch import nn
import torch
from torch.nn import *
from torchvision.models import resnet50


class HandModel(nn.Module):
    def __init__(self, classes=5):
        super(HandModel, self).__init__()
        self.conv_1 = Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2))
        self.bn_1 = BatchNorm2d(num_features=32)
        self.relu_activate = ReLU()
        self.conv_2 = Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2))
        self.bn_2 = BatchNorm2d(num_features=32)
        self.maxpool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.flatten = Flatten()
        self.dropout = Dropout()
        self.linear_1 = Linear(in_features=4608, out_features=128)
        self.bn_3 = BatchNorm1d(num_features=128)
        self.linear_2 = Linear(in_features=128, out_features=classes)
        self.softmax_activate = Softmax(dim=-1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_activate(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu_activate(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.bn_3(x)
        x = self.relu_activate(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.softmax_activate(x)
        return x


class ResModel(Module):
    def __init__(self, classes=5):
        super(ResModel, self).__init__()
        backbone = resnet50(pretrained=False)
        fc_in_channel = backbone.fc.in_features
        backbone.fc = Linear(fc_in_channel, classes)
        self.model = backbone

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    # handModel = HandModel(classes=5)
    # x = torch.ones((2, 3, 100, 100))
    # print(handModel(x))
    resModel = ResModel(classes=6)
    new_state = {}
    weight_state = torch.load('./pretrained/resnet50-19c8e357.pth')
    for key, value in weight_state.items():
        if 'fc' not in key:
            new_state['model.' + key] = value
        else:
            new_state['model.' + key] = resModel.state_dict()['model.' + key]
    resModel.load_state_dict(new_state)
    x = torch.ones((2, 3, 100, 100))
    print(resModel(x))
