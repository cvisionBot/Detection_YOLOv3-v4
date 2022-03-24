import math
import torch

from torch import nn

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    box2 = box2.T

    if x1y1x2y2:
        b1_x1, b1_y1, b2_x2, b2_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else: # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[0] + box2[3] / 2
    
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
        (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + 
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if DIoU:
                return iou - rho2 / c2 # Distance IoU
            elif CIoU:
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 + eps) - iou + v
                return iou - (rho2 / c2 + v * alpha) # Complete IoU
        else:
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area # G IoU
    else:
        return iou



class YOLO_Loss(nn.Module):
    def __init__(self, cfg, anchors):
        super(YOLO_Loss, self).__init__()
        self.cfg = cfg
        self.branch_anchors = anchors
        self.num_anchors = len(self.branch_anchors) # 3
        self.bbox_attrs = 5 + self.cfg['classes']
        self.img_w = self.cfg['input_size']
        self.img_h = self.cfg['input_size']

        self.ignore_threshold = 0.5
        self.lambda_xy = 0.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, input):
        pred, target = input
        batch_size, _, layer_h, layer_w = pred.size()
        stride_h = self.img_h / layer_h
        stride_w = self.img_w / layer_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.branch_anchors]

        # [b, 3, (5 + 20), layer_h, layer_w]
        prediction = pred.view(batch_size, self.num_anchors, self.bbox_attrs, layer_h, layer_w).permute(0, 1, 3, 4, 2).contiguous()
        

        


