import torch
from torch import nn

from ..layers.convolution import Conv2dBn, Conv2dBnAct

class Block53(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Block53, self).__init__()
        self.get_channel = out_channels
        self.conv1 = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=1, stride=1)
        self.conv2 = Conv2dBnAct(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.residual = Conv2dBn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        if input.size() != output.size():
            input = self.residual(input)
        output += input
        return output
    
    def get_channels(self):
        return self.get_channel


class Neck_Modules(nn.Module):
    def __init__(self, in_channels, pre_channels):
        super(Neck_Modules, self).__init__()
        self.get_channel = in_channels // 2
        self.conv1_1 = Conv2dBnAct(in_channels=pre_channels, out_channels=in_channels // 2, kernel_size=1, stride=1)
        self.conv1_2 = Conv2dBnAct(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=3, stride=1)
        self.conv2_1 = Conv2dBnAct(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1)
        self.conv2_2 = Conv2dBnAct(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=3, stride=1)
        self.conv = Conv2dBnAct(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1)

    def forward(self, input):
        output = self.conv1_1(input)
        output = self.conv1_2(output)
        output = self.conv2_1(output)
        output = self.conv2_2(output)
        output = self.conv(output)
        return output

    def get_channels(self):
        return self.get_channel


class Head_Modules(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Head_Modules, self).__init__()
        self.conv = Conv2dBnAct(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=3, stride=1)
        self.head = nn.Conv2d(in_channels=in_channels * 2, out_channels=3 * (5 + out_channels), kernel_size=1, stride=1)

    def forward(self, input):
        output = self.conv(input)
        output = self.head(output)
        return output


class Upsample_Modules(nn.Module):
    def __init__(self, in_channels):
        super(Upsample_Modules, self).__init__()
        self.conv = Conv2dBnAct(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, input):
        output = self.conv(input)
        output = self.upsample(output)
        return output