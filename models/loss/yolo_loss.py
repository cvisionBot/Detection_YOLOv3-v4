import math
import torch

from torch import nn
# from models.detector.anchors import anchors

class YOLO_Loss(nn.Module):
    def __init__(self, cfg):
        super(YOLO_Loss, self).__init__()