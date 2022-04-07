import os
import cv2
import random
import argparse

import torch
from torchsummary import summary
from models.detector.yolov3 import YOLOv3
from module.detector import Detector
from utils.module_select import get_model
from utils.utility import preprocess_input
from utils.yaml_helper import get_train_configs


def parse_names(names_file):
    names_file = os.getcwd()+names_file
    with open(names_file, 'r') as f:
        return f.read().splitlines()


def gen_random_colors(names):
    colors = [(random.randint(0, 255),
               random.randint(0, 255),
               random.randint(0, 255)) for i in range(len(names))]
    return colors


def main(cfg, image_name, save):
    names = parse_names(cfg['names'])
    colors = gen_random_colors(names)

    # Preprocess Image
    image = cv2.imread(image_name)
    image = cv2.resize(image, (320, 320))
    image_inp = preprocess_input(image)
    image_inp = image_inp.unsqueeze(0)
    if torch.cuda.is_available:
        image_inp=image_inp.cuda()

    # Load trained model
    backbone = get_model(cfg['backbone'])
    model = YOLOv3(Backbone=backbone, num_classes=cfg['classes'], in_channels=cfg['in_channels'], varient=cfg['varient'])
    if torch.cuda.is_available:
        model = model.to('cuda')

    model_module = Detector.load_from_checkpoint(
        '/home/insig/Detection_YOLOv3/saved/DarkNet53_YOLOv3_Pascal/version_3/checkpoints/last.ckpt', model=model)
    model_module.eval()

    