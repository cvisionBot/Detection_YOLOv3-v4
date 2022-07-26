import torch

from torch import nn
from models.layers.blocks import Pretrain_Neck_Modules, Head_Modules, Upsample_Modules

class Pretrain_YOLOv3(nn.Module):
    def __init__(self, Backbone, num_classes,pretrain_flag, varient='MobileNetv2'):
        super(Pretrain_YOLOv3, self).__init__()

        self.backbone = Backbone(classes=num_classes, pretrain_flag=pretrain_flag, varient=varient)
        self.stage_channels = self.backbone.stages

        self.neck1 = Pretrain_Neck_Modules(in_channels=self.stage_channels[3], pre_channels=self.stage_channels[3], out_channels=self.stage_channels[2])
        self.neck2 = Pretrain_Neck_Modules(in_channels=self.stage_channels[2] ,pre_channels=self.stage_channels[2] + (self.stage_channels[2] // 2), out_channels=self.stage_channels[1])
        self.neck3 = Pretrain_Neck_Modules(in_channels=self.stage_channels[1], pre_channels=self.stage_channels[1] + (self.stage_channels[1]//2), out_channels=self.stage_channels[0]) 

        self.head1 = Head_Modules(in_channels=self.stage_channels[2], out_channels=num_classes)
        self.head2 = Head_Modules(in_channels=self.stage_channels[1], out_channels=num_classes)
        self.head3 = Head_Modules(in_channels=self.stage_channels[0], out_channels=num_classes)

        self.rout1 = Upsample_Modules(in_channels=self.stage_channels[2])
        self.rout2 = Upsample_Modules(in_channels=self.stage_channels[1])

    def forward(self, input):
        s3 = self.backbone.stage3(input)
        s4 = self.backbone.stage4(s3)
        s5 = self.backbone.stage5(s4)

        neck1 = self.neck1(s5)
        branch1 = self.head1(neck1) # [1, 3 * (5 + classes), 13, 13]
        print('branch1 shape : ', branch1.shape)

        rout1 = self.rout1(neck1)
        concat1 = torch.cat([rout1, s4], dim=1)
        
        neck2 = self.neck2(concat1)
        branch2 = self.head2(neck2) # [1, 3 * (5 + classes), 26, 26]
        print('branch2 shape : ', branch2.shape)

        rout2 = self.rout2(neck2)
        concat2 = torch.cat([rout2, s3], dim=1)

        neck3 = self.neck3(concat2)
        branch3 = self.head3(neck3) # [1, 3 * (5 + classes), 52, 52]
        print('branch3 shape : ', branch3.shape)
        return branch1, branch2, branch3


if __name__ == '__main__':
    from models.backbone.pretrain_model import PreTrainModel
    model = Pretrain_YOLOv3(Backbone=PreTrainModel, num_classes=20, pretrain_flag=True, varient='MobileNetv2')
    model(torch.rand(1, 3, 224, 224))
    # model visualize to onnx
    dumTensor = torch.rand(1, 3, 224, 224)
    torch.onnx.export(model, dumTensor, 'model_visualize.onnx', export_params=True, opset_version=9,
                        input_names=['input'], output_names=['branch1', 'branch2', 'branch3'])
    