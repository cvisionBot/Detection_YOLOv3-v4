import torch
from torch import nn

from ..layers.convolution import Conv2dBn, Conv2dBnAct
from ..layers.blocks import Block53
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
        self.stages = [128, 256, 512, 1024]
        self.stem = StemBlock(in_channels=in_channels, out_channels=self.in_channels)
        # configs : out_channels, kernel_size, stride, iter_cnt
        block1 = [64, 3, 1, 1]
        block2 = [128, 3, 1, 2]
        block3 = [256, 3, 1, 8]
        block4 = [512, 3, 1, 8]
        block5 = [1024, 3, 1, 4]

        self.block1 = self.make_block(block1)
        self.s1_max_pool = Conv2dBnAct(in_channels=block1[0], out_channels=block2[0], kernel_size=3, stride=2)
        self.block2 = self.make_block(block2)
        self.s2_max_pool = Conv2dBnAct(in_channels=block2[0], out_channels=block3[0], kernel_size=3, stride=2)
        self.block3 = self.make_block(block3)
        self.s3_max_pool = Conv2dBnAct(in_channels=block3[0], out_channels=block4[0], kernel_size=3, stride=2)
        self.block4 = self.make_block(block4)
        self.s4_max_pool = Conv2dBnAct(in_channels=block4[0], out_channels=block5[0], kernel_size=3, stride=2)
        self.block5 = self.make_block(block5)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1024, classes, 1)
        )
        # self.yolo_classifier = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(),
        #     nn.Linear(in_features=1024, out_features=classes),
        #     nn.Softmax()
        # )

    def forward(self, input):
        stem = self.stem(input) 
        s1 = self.block1(stem) 
        s1_max = self.s1_max_pool(s1) 
        s2 = self.block2(s1_max) 
        s2_max = self.s2_max_pool(s2) 
        s3 = self.block3(s2_max) 
        s3_max = self.s3_max(s3) 
        s4 = self.block4(s3_max) 
        s4_max = self.s4_max(s4) 
        s5 = self.block5(s4_max) 
        pred = self.classifier(s5)
        b, c, h, w = pred.size()
        pred = pred.view(b, c)
        return {'pred':pred}

    def make_block(self, cfg):
        layers = []
        input_ch = cfg[0]
        for i in range(cfg[-1]):
            layer = Block53(in_channels=input_ch, out_channels=cfg[0], kernel_size=cfg[1], stride=cfg[2])
            layers.append(layer)
            input_ch = layer.get_channels()
        return nn.Sequential(*layers)


def DarkNet(in_channels, classes=1000, varient=53):
    if varient == 53:
        model = _DarkNet53(in_channels=in_channels, classes=classes)
    else:
        assert "not yet!!"
    weight_initialize(model)
    return model


if __name__ == '__main__':
    model = DarkNet(in_channels=3, classes=1000)
    model(torch.rand(1, 3, 416, 416))