import torch
from torch import nn

from ..layers.convolution import Conv2dBnAct


class SPP_Module(nn.Module):
    def __init__(self, in_channels):
        super(SPP_Module, self).__init__()

        self.conv1 = Conv2dBnAct(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, dilation=1, groups=1,
                                padding_mode='zeros', act='Mish')
        self.conv2 = Conv2dBnAct(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1, dilation=1, groups=1,
                                padding_mode='zeros', act='Mish')
        self.conv3 = Conv2dBnAct(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                                padding_mode='zeros', act='Mish')
        self.conv4 = Conv2dBnAct(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1, dilation=1, groups=1,
                                padding_mode='zeros', act='Mish')
        
        self.max_pool1 = nn.MaxPool2d(kernel_size=(13, 13), stride=1, padding=6)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(9, 9), stride=1, padding=4)
        self.max_pool3 = nn.MaxPool2d(kernel_size=(5, 5), stride=1, padding=2)

        self.conv5 = Conv2dBnAct(in_channels=(in_channels // 2 * 4), out_channels=in_channels // 2, kernel_size=1, stride=1, dilation=1, groups=1,
                                padding_mode='zeros', act='Mish')
        self.conv6 = Conv2dBnAct(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                                padding_mode='zeros', act='Mish')
        self.conv7 = Conv2dBnAct(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1, dilation=1, groups=1,
                                padding_mode='zeros', act='Mish')
    
    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)

        max_pool1 = self.max_pool1(output)
        max_pool2 = self.max_pool2(output)
        max_pool3 = self.max_pool3(output)

        output = torch.cat([max_pool1, max_pool2, max_pool3, output], axis=1)

        output = self.conv5(output)
        output = self.conv6(output)
        output = self.conv7(output)
        return output
