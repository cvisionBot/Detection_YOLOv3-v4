import math
import torch
import numpy as np
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
        self.num_classes = self.cfg['classes']
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

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        ground_truth = target['annot']

        mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.encode_target(ground_truth, scaled_anchors, layer_w, layer_h, self.ignore_threshold)

        # loss
        loss_x = self.bce_loss(x * mask, tx * mask)
        loss_y = self.bce_loss(y * mask, ty * mask)
        loss_w = self.mse_loss(w * mask, tw * mask)
        loss_h = self.mse_loss(h * mask, th * mask)
        loss_conf = self.bce_loss(conf * mask, mask) + \
            0.5 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0)
        loss_cls = self.bce_loss(pred_cls[mask==1], tcls[mask==1])

        loss = (loss_x * self.lambda_xy) + (loss_y * self.lambda_xy) + \
            (loss_w * self.lambda_wh) + (loss_h * self.lambda_wh) + \
                (loss_conf * self.lambda_conf) + (loss_cls * self.lambda_cls)
        return loss




    def encode_target(self, target, anchors, layer_w, layer_h, ignore_threshold):
        batch_size = target.size()[0]

        mask = torch.zeros(batch_size, self.num_anchors, layer_h, layer_w)
        noobj_mask = torch.ones(batch_size, self.num_anchors, layer_h, layer_w)
        tx = torch.zeros(batch_size, self.num_anchors, layer_h, layer_w)
        ty = torch.zeros(batch_size, self.num_anchors, layer_h, layer_w)
        tw = torch.zeros(batch_size, self.num_anchors, layer_h, layer_w)
        th = torch.zeros(batch_size, self.num_anchors, layer_h, layer_w)
        tconf = torch.zeros(batch_size, self.num_anchors, layer_h, layer_w)
        tcls = torch.zeros(batch_size, self.num_anchors, layer_h, layer_w, self.num_classes)

        for b in range(batch_size):
            for t in range(target.size()[1]):
                if target[b, t].sum() == 0:
                    continue
                
                gx = target[b, t, 1] * layer_w
                gy = target[b, t, 2] * layer_h
                gw = target[b, t, 3] * layer_w
                gh = target[b, t, 4] * layer_h

                gi = int(gx)
                gj = int(gy)

                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)), np.zrray(anchors)), 1))

                calc_iou = bbox_iou(gt_box,anchor_shapes, x1y1x2y2=False)
                noobj_mask[batch_size, calc_iou > ignore_threshold, gj, gi] = 0
                best_n = np.argmax(calc_iou)

                mask[batch_size, best_n, gj, gi] = 1
                tx[batch_size, best_n, gj, gi] = gx - gi
                ty[batch_size, best_n, gj, gi] = gy - gj
                tw[batch_size, best_n, gj, gi] = math.log(gw/anchors[best_n][0] + 1e-16)
                th[batch_size, best_n, gj, gi] = math.log(gh/anchors[best_n][1] + 1e-16)
                tconf[batch_size, best_n, gj, gi] = 1
                tcls[batch_size, best_n, gj, gi, int(target[b, t, 0])] = 1
        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls    

    
        


