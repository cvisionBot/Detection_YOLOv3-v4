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