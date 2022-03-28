from torch import optim

from models.backbone.darknet import DarkNet
from models.detector.yolov3 import YOLOv3


def get_model(model_name):
    model_dict = {'DarkNet53':DarkNet}
    return model_dict.get(model_name)


def get_optimizer(optimizer_name, params, **kwargs):
    optim_dict = {'sgd': optim.SGD, 'adam': optim.Adam}
    optimizer = optim_dict.get(optimizer_name)
    return optimizer(params, **kwargs)