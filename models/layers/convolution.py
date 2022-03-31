import torch
from torch import nn

def getPadding(kernel_size):
    return (int((kernel_size - 1) / 2), (int((kernel_size - 1)/ 2)))

class Conv2dBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding_mode='zeros', act=None):
        super(Conv2dBnAct, self).__init__()
        self.padding = getPadding(kernel_size)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                padding=self.padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        if act is None:
            act = nn.LeakyReLU()
        self.act = act

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class Conv2dBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding_mode='zeros', act=None):
        super(Conv2dBn, self).__init__()
        self.padding = getPadding(kernel_size)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                padding=self.padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return output