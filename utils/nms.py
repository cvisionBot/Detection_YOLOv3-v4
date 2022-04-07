import math
import torch
import torch.nn.functional as F
import numpy as np

def non_max_suppression(prediction, num_classes, conf_threshold=0.5, nms_thres=0.4):
    '''
    prediction (cx, cy, width, height)
    return (x1, y1, x2, y2, conf, class_score, class_pred)
    '''
    pred = prediction.new(prediction.shape)
    pred[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    pred[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    pred[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    pred[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = pred[:, :, :4]

    for image_i, image_pred in enumerate(prediction):
        conf_mask = (image_pred[:, 4] >= conf_threshold).squeeze()
        image_pred = image_pred[conf_mask]

        if not image_pred.size(0):
            continue

        
