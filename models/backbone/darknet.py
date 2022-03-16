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

        self.block1 = self.make_block(block1, 128)
        self.block2 = self.make_block(block2, 256)
        self.block3 = self.make_block(block3, 512)
        self.block4 = self.make_block(block4, 1024)
        self.block5 = self.make_block(block5, 1024)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1024, classes, 1)
        )
        self.yolo_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=classes),
            nn.Softmax()
        )

    def forward(self, input):
        stem = self.stem(input) # [1, 64, 208, 208]
        s1 = self.block1(stem) # [1, 128, 104, 104]
        s2 = self.block2(s1) # [1, 256, 52, 52]
        s3 = self.block3(s2) # [1, 512, 26, 26]
        s4 = self.block4(s3) # [1, 1024, 13, 13]
        s5 = self.block5(s4) # [1, 1024, 13, 13]
        pred = self.classifier(s5)
        b, c, h, w = pred.size()
        pred = pred.view(b, c)
        return {'pred':pred}


    def make_block(self, cfg, out_channels):
        layers = []
        input_ch = cfg[0]
        if cfg[-1] == 4:
            for i in range(cfg[-1]):
                layer = Block53(in_channels=input_ch, out_channels=cfg[0], kernel_size=cfg[1], stride=cfg[2])
                layers.append(layer)
                input_ch = layer.get_channels()
        else:
            for i in range(cfg[-1]):
                layer = Block53(in_channels=input_ch, out_channels=cfg[0], kernel_size=cfg[1], stride=cfg[2])
                layers.append(layer)
                input_ch = layer.get_channels()
            layers.append(Conv2dBnAct(in_channels=cfg[0], out_channels=out_channels, kernel_size=3, stride=2))
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