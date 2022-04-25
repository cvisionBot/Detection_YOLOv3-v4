import math
import torch

from torch import nn
from models.layers.blocks import Neck_Modules, Head_Modules, Upsample_Modules
from models.initialize import weight_initialize


class YOLOv3(nn.Module):
    def __init__(self, Backbone, num_classes, in_channels=3, varient=53):
        super(YOLOv3, self).__init__()

        self.backbone = Backbone(in_channels=in_channels, classes=num_classes, varient=varient)
        self.stage_channels = self.backbone.stages

        self.neck1 = Neck_Modules(in_channels=self.stage_channels[3], pre_channels=self.stage_channels[3])
        self.neck2 = Neck_Modules(in_channels=self.stage_channels[2] ,pre_channels=self.stage_channels[2] + self.stage_channels[1])
        self.neck3 = Neck_Modules(in_channels=self.stage_channels[1], pre_channels=self.stage_channels[1] + self.stage_channels[0])

        self.head1 = Head_Modules(in_channels=self.stage_channels[2], out_channels=num_classes)
        self.head2 = Head_Modules(in_channels=self.stage_channels[1], out_channels=num_classes)
        self.head3 = Head_Modules(in_channels=self.stage_channels[0], out_channels=num_classes)

        self.rout1 = Upsample_Modules(in_channels=self.stage_channels[2])
        self.rout2 = Upsample_Modules(in_channels=self.stage_channels[1])

        weight_initialize(self)

    def forward(self, input):
        stem = self.backbone.stem(input)
        s1 = self.backbone.block1(stem)
        s1_max = self.backbone.s1_max_pool(s1)
        s2 = self.backbone.block2(s1_max)
        s2_max = self.backbone.s2_max_pool(s2)
        s3 = self.backbone.block3(s2_max)
        s3_max = self.backbone.s3_max_pool(s3)
        s4 = self.backbone.block4(s3_max)
        s4_max = self.backbone.s4_max_pool(s4)
        s5 = self.backbone.block5(s4_max)

        neck1 = self.neck1(s5)
        branch1 = self.head1(neck1) # [1, 3 * (5 + classes), 13, 13]

        rout1 = self.rout1(neck1)
        concat1 = torch.cat([rout1, s4], dim=1)
        
        neck2 = self.neck2(concat1)
        branch2 = self.head2(neck2) # [1, 3 * (5 + classes), 26, 26]

        rout2 = self.rout2(neck2)
        concat2 = torch.cat([rout2, s3], dim=1)

        neck3 = self.neck3(concat2)
        branch3 = self.head3(neck3) # [1, 3 * (5 + classes), 52, 52]

        return branch1, branch2, branch3


if __name__ == '__main__':
    from models.backbone.darknet import DarkNet
    model = YOLOv3(Backbone=DarkNet, num_classes=20, in_channels=3, varient=53)
    # model visualize to onnx
    dumTensor = torch.rand(1, 3, 416, 416)
    # print(model(dumTensor))
    # torch.onnx.export(model, dumTensor, 'model_visualize.onnx', export_params=True, opset_version=9, do_constant_folding=True,
    #                     input_names=['input'], output_names=['branch1', 'branch2', 'branch3'])
