import torch
from torch import nn
from torchvision.models import mobilenet_v2

def show_features(model : nn.Module):
    for name, children in model.named_children():
        print("name : {name}\n, children : {children}".format(name=name, children=children))

class Pretrain_MobileNetv2(nn.Module):
    def __init__(self, pretrained, classes):
        super(Pretrain_MobileNetv2, self).__init__()
        self.backbone = None
        self.stages = [24, 32, 96, 1280]
        if pretrained:
            self.backbone = mobilenet_v2(pretrained=True)
            # for p in self.backbone.parameters():
            #   p.requires_grad = False
        else:
            self.backbone = mobilenet_v2(pretrained=False)
        self.features = self.backbone.features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )

        self.stage3 = nn.Sequential(
            self.features[:5],
            self.features[5].conv[:4],
        )
        self.stage4 = nn.Sequential(
            self.features[5:12],
            self.features[12].conv[:4],
        )
        self.stage5 = nn.Sequential(
            self.features[12:19],
            # self.features[18].co,
        )

    def forward(self, input):
        s3 = self.stage3(input)
        print('s3 shape : ', s3.shape)
        s4 = self.stage4(s3)
        print('s4.shape : ', s4.shape)
        s5 = self.stage5(s4)
        print('s5.shape : ', s5.shape)
        pred = self.classifier(s5)
        b, c, h, w = pred.size()
        pred = pred.view(b, c)
        return {'pred':pred}


def PreTrainModel(classes=1000, pretrain_flag=False, varient=None):
    if varient == 'MobileNetv2':
        model = Pretrain_MobileNetv2(pretrained=pretrain_flag, classes=classes)
    else:
        raise NotImplementedError
    return model


if __name__ == '__main__':
    model = PreTrainModel(varient='MobileNetv2')
    # show_features(model)
    model(torch.rand(1, 3, 224, 224))