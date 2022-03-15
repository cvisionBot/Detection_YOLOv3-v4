import torch
from torch import nn

from ..layers.convolution import Conv2dBn, Conv2dBnAct
from ..initialize import weight_initialize


class StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StemBlock, self).__init__()
        self.conv1 = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=3)
        self.conv2 = Conv2dBnAct(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=3, stride=2)

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        return output


class _DarkNet53(nn.Module):
    def __init__(self, in_channels, classes):
        super(_DarkNet53, self).__init__()
        self.in_channels = 64
        self.stem = StemBlock(in_channels=in_channels, out_channels=self.in_channels)

    def forward(self, input):
        stem = self.stem(input)
        print('stem shape : ', stem.shape)


def DarkNet(in_channels, classes=1000, varient=53):
    if varient == 53:
        model = _DarkNet53(in_channels=in_channels, classes=classes)
    else:
        assert "not yet!!"
    weight_initialize(model)
    return model


if __name__ == '__main__':
    model = DarkNet(in_channels=3, classes=1000)
    model(torch.rand(1, 3, 256, 256))