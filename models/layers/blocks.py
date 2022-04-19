import torch
from torch import nn

from ..layers.convolution import Conv2dBn, Conv2dBnAct


class CSPstemBlock(nn.Module):
    def __init__(self, in_channels, out_channels, iter_cnt, activation=None):
        super(CSPstemBlock, self).__init__()
        self.iter_cnt = iter_cnt
        self.block_list = nn.ModuleList([])
        self.conv1 = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=3, stride=1, dilation=1, groups=1,
                                    padding_mode='zeros', act=activation)
        self.conv2 = Conv2dBnAct(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=3, stride=2, dilation=1, groups=1,
                                    padding_mode='zeros', act=activation)
        
        self.conv3 = Conv2dBnAct(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, dilation=1, groups=1,
                                    padding_mode='zeros', act=activation)
        self.in_channels=out_channels
        for i in range(self.iter_cnt):
            self.block_list.append(Block53(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3, stride=1, activation='Mish'))
        
        self.conv4 = Conv2dBnAct(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, dilation=1, groups=1, 
                                    padding_mode='zeros', act=activation)
        self.identity = Conv2dBnAct(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, dilation=1, groups=1,
                                    padding_mode='zeros', act=activation)
                                    
    def forward(self, input):
        output = self.conv1(input)
        split_output = self.conv2(output)
        output = self.conv3(split_output)

        for b in self.block_list:
            output = b(output)
        output = self.conv4(output)
        split_output = self.identity(split_output)
        output = torch.cat([output, split_output], axis=1)
        return output


class StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None):
        super(StemBlock, self).__init__()
        self.conv1 = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=3, stride=1, dilation=1, groups=1,
                                    padding_mode='zeros', act=activation)
        self.conv2 = Conv2dBnAct(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=3, stride=2, dilation=1, groups=1,
                                    padding_mode='zeros', act=activation)

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        return output


class Block53(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=None):
        super(Block53, self).__init__()
        self.get_channel = out_channels
        self.conv1 = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=1, stride=1, dilation=1, groups=1,
                                    padding_mode='zeros', act=activation)
        self.conv2 = Conv2dBnAct(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=1, groups=1,
                                    padding_mode='zeros', act=activation)

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = output +  input
        return output
    
    def get_channels(self):
        return self.get_channel


class CSPBlock(nn.Module):
    def __init__(self, out_channels, iter_cnt, activation=None):
        super(CSPBlock, self).__init__()
        self.iter_cnt = iter_cnt
        self.block_list = nn.ModuleList([])
        self.conv2 = Conv2dBnAct(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=3, stride=2, dilation=1, groups=1,
                                    padding_mode='zeros', act=activation)
        
        self.conv3 = Conv2dBnAct(in_channels=out_channels, out_channels=out_channels // 2, kernel_size=1, stride=1, dilation=1, groups=1,
                                    padding_mode='zeros', act=activation)
        self.in_channels=out_channels // 2
        for i in range(self.iter_cnt):
            self.block_list.append(Block53(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, activation='Mish'))
        self.conv4 = Conv2dBnAct(in_channels=out_channels // 2, out_channels=out_channels // 2, kernel_size=1, stride=1, dilation=1, groups=1, 
                                    padding_mode='zeros', act=activation)
        self.identity = Conv2dBnAct(in_channels=out_channels, out_channels=out_channels // 2, kernel_size=1, stride=1, dilation=1, groups=1,
                                    padding_mode='zeros', act=activation)
                                    
    def forward(self, input):
        split_output = self.conv2(input)
        output = self.conv3(split_output)


        for b in self.block_list:
            output = b(output)

        output = self.conv4(output)
        split_output = self.identity(split_output)
        output = torch.cat([output, split_output], axis=1)
        return output


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

class Neck_Modules_v4(nn.Module):
    def __init__(self, in_channels, pre_channels):
        super(Neck_Modules_v4, self).__init__()
        self.get_channel = in_channels // 2
        self.conv1_1 = Conv2dBnAct(in_channels=pre_channels, out_channels=in_channels, kernel_size=1, stride=1)
        self.conv1_2 = Conv2dBnAct(in_channels=in_channels, out_channels=pre_channels, kernel_size=3, stride=1)
        self.conv2_1 = Conv2dBnAct(in_channels=pre_channels, out_channels=in_channels, kernel_size=1, stride=1)
        self.conv2_2 = Conv2dBnAct(in_channels=in_channels, out_channels=pre_channels, kernel_size=3, stride=1)
        self.conv = Conv2dBnAct(in_channels=pre_channels, out_channels=in_channels, kernel_size=1, stride=1)

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


class Route_Module(nn.Module):
    def __init__(self, in_channels, type=True):
        super(Route_Module, self).__init__()
        self.type = type
        if self.type:
            self.conv = Conv2dBnAct(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1, dilation=1, groups=1,
                                    padding_mode='zeros', act='Mish')
        else:
            self.conv = Conv2dBnAct(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=3, stride=2, dilation=1, groups=1,
                                    padding_mode='zeros', act='Mish')
        
    def forward(self, input):
        output = self.conv(input)
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

