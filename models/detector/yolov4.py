from logging.config import valid_ident
import math
import torch

from torch import nn
from models.layers.attention import SPP_Module
from models.layers.blocks import Head_Modules, Route_Module, Neck_Modules_v4, Upsample_Modules
from models.initialize import weight_initialize


class YOLOv4(nn.Module):
    def __init__(self, Backbone, num_classes, in_channels=3, varient=None):
        super(YOLOv4, self).__init__()

        self.backbone = Backbone(in_channels=in_channels, classes=num_classes, varient=varient)
        self.stage_channels = self.backbone.stages
        #self.stages = [128, 256, 512, 1024]
        self.spp_module = SPP_Module(in_channels=self.stage_channels[-1])

        self.s4_route = Route_Module(in_channels=self.stage_channels[2])
        self.s3_route = Route_Module(in_channels=self.stage_channels[1])

        self.neck1 = Neck_Modules_v4(in_channels=self.stage_channels[1], pre_channels=self.stage_channels[2])
        self.neck2 = Neck_Modules_v4(in_channels=self.stage_channels[0], pre_channels=self.stage_channels[1])
        self.neck3 = Neck_Modules_v4(in_channels=self.stage_channels[1], pre_channels=self.stage_channels[2])
        self.neck4 = Neck_Modules_v4(in_channels=self.stage_channels[2], pre_channels=self.stage_channels[-1])

        self.up_module1 = Upsample_Modules(in_channels=self.stage_channels[2])
        self.up_module2 = Upsample_Modules(in_channels=self.stage_channels[1])
        
        self.branch_route1 = Route_Module(in_channels=self.stage_channels[0], type=False)
        self.branch_route2 = Route_Module(in_channels=self.stage_channels[1], type=False)

        self.head1 = Head_Modules(in_channels=self.stage_channels[0], out_channels=num_classes)
        self.head2 = Head_Modules(in_channels=self.stage_channels[1], out_channels=num_classes)
        self.head3 = Head_Modules(in_channels=self.stage_channels[2], out_channels=num_classes)
        weight_initialize(self)

    def forward(self, input):
        stem = self.backbone.stem(input)
        s1_head = self.backbone.block1_head(stem)
        s1 = self.backbone.block1(s1_head)
        s2_head = self.backbone.block2_head(s1)
        s2 = self.backbone.block2(s2_head)
        s3_head = self.backbone.block3_head(s2)
        s3 = self.backbone.block3(s3_head)
        s4_head = self.backbone.block4_head(s3)
        s4 = self.backbone.block4(s4_head)

        # s3_head
        spp_out = self.spp_module(s4)
        up_module1 = self.up_module1(spp_out)
        
        # PAN
        s4_route = self.s4_route(s4_head)
        s4_header = torch.cat([s4_route, up_module1], axis=1)
        neck1 = self.neck1(s4_header)
        up_module2 = self.up_module2(neck1)
        s3_route = self.s3_route(s3_head)
        s3_header = torch.cat([s3_route, up_module2], axis=1)
        neck2 = self.neck2(s3_header)
        branch1 = self.head1(neck2) # [1, 3 * (5 + classes), 64, 64]
        # print('branch1 : ', branch1.shape)

        branch1_route = self.branch_route1(neck2)
        branch1_cat = torch.cat([branch1_route, neck1], axis=1)
        neck3 = self.neck3(branch1_cat)
        branch2 = self.head2(neck3) # [1, 3 * (5 + classes), 32, 32]
        # print('branch2 : ', branch2.shape)

        branch2_route = self.branch_route2(neck3)
        branch2_cat = torch.cat([branch2_route, spp_out], axis=1)
        neck4 = self.neck4(branch2_cat)
        branch3 = self.head3(neck4) # [1, 3 * (5 + classes), 16, 16]
        # print('branch3 : ', branch3.shape)

        return branch3, branch2, branch1




if __name__ == '__main__':
    from models.backbone.darknet import DarkNet
    model = YOLOv4(Backbone=DarkNet, num_classes=20, in_channels=3)
    dumTensor = torch.rand(1, 3, 512, 512)
    # print(model(dumTensor))
    # torch.onnx.export(model, dumTensor, 'model_visualize.onnx', export_params=True, opset_version=9, do_constant_folding=True,
    #                     input_names=['input'], output_names=['branch3', 'branch2', 'branch1'])