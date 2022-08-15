import torch
from torch.nn import *


class HandModel(Module):
    def __init__(self, classes=16):
        '''输入的形状为（200，6，1）（w,h,c)'''
        super(HandModel, self).__init__()
        self.classes = classes
        # self.conv_1=Conv2d(in_channels=1,out_channels=)
