from collections import Counter
import math
from tkinter.messagebox import NO

import torch
import numpy as np

from utils.nms import non_max_suppression
from models.loss.yolo_loss import bbox_iou


class MeanAveragePrecision:
    def __init__(self, num_classes, anchors, input_size):
        self._all_true_boxes_variable = 0
        self._all_pred_boxes_variable = 0
        self._img_idx = 0
        self._num_classes = num_classes
        self._input_size = input_size
        self._anchors = anchors

    def reset_states(self):
        self._all_true_boxes_variable = 0
        self._all_pred_boxes_variable = 0
        self._img_idx = 0

    def update_state(self, y_true, y_pred):
        pred_decoded = []
        for i, pred in enumerate(y_pred):
            pred_decoded.append(self.decode_boxes(pred, self._anchors[i], self._num_classes, [self._input_size, self._input_size]))
        pred_output = torch.cat(pred_decoded, 1)
        pred_detections = non_max_suppression(pred_output, num_classes=self._num_classes, nms_thres=0.4) # [[num_boxes, 7], [num_boxes, 7], None, ...]

        true_decoded = []
        for i in torch.arange(len(self._anchors)): 
            encoded = self.encode_target(y_true, self._anchors[i], self._input_size, self._num_classes, i)
            true_decoded.append(self.decode_boxes(encoded, self._anchors[i], self._num_classes, [self._input_size, self._input_size], True))
        true_output = torch.cat(true_decoded, 1)
        true_detections = non_max_suppression(true_output, num_classes=self._num_classes, nms_thres=0.4) # [[num_boxes, 7], [num_boxes, 7], None, ...]

        for idx in torch.arange(y_true.size(0)):
            pred_nms = pred_detections[idx]
            if pred_nms is None:
                if torch.cuda.is_available:
                    pred_nms = torch.zeros((0, 7)).cuda()
                else:
                    pred_nms = torch.zeros((0, 7))
            pred_img_idx = torch.zeros([pred_nms.size(0), 1], dtype=torch.float32) + self._img_idx
            if pred_nms.is_cuda:
                pred_img_idx = pred_img_idx.cuda()
            pred_concat = torch.cat([pred_img_idx, pred_nms], dim=1)

            true_nms = true_detections[idx]
            if true_nms is None:
                if torch.cuda.is_available:
                    true_nms = torch.zeros((0, 7)).cuda()
                else:
                    true_nms = torch.zeros((0, 7))
            true_img_idx = torch.zeros([true_nms.size(0), 1], dtype=torch.float32) + self._img_idx
            if true_nms.is_cuda:
                true_img_idx = true_img_idx.cuda()
            true_concat = torch.cat([true_img_idx, true_nms], dim=1)

            if self._img_idx == 0.:
                self._all_true_boxes_variable = true_concat
                self._all_pred_boxes_variable = pred_concat
            else:
                self._all_true_boxes_variable = torch.concat([self._all_true_boxes_variable, true_concat], axis=0)
                self._all_pred_boxes_variable = torch.concat([self._all_pred_boxes_variable, pred_concat], axis=0)

            self._img_idx += 1

    def result(self):
        return self.mean_average_precision(self._all_true_boxes_variable, self._all_pred_boxes_variable, self._num_classes)

    def encode_target(self, target, anchors, input_size, num_classes, branch_idx):
        """Encoding Target(Label)

        Arguments:
            target (Tensor): Tensor of Label, specified shape as [batch_size, max_num_annots, 5(cx, cy, w, h, class_idx)]
            anchors (List): Anchors, specified shape as [num_anchors, 2(w, h)]
            input_size (int): input size of Image
            num_classes (int): number of classes
            branch_idx (int): branch index 

        Returns:
            List: encoded target, specified as [batch_size, num_anchors*(5(x, y, w, h)+num_classes), input_size/(32, 16, 8), input_size/(32, 16, 8)]
        """
        
        batch_size = target.size(0)
        num_anchors = len(anchors)
        scaled_anchors = [(a_w / (32 / (2 ** branch_idx)), a_h / (32 / (2 ** branch_idx))) for a_w, a_h in anchors]
        layer_h = int(input_size / (32 / (2 ** branch_idx)))
        layer_w = int(input_size / (32 / (2 ** branch_idx)))
    
        tx = torch.zeros(batch_size, num_anchors, layer_h, layer_w)
        ty = torch.zeros(batch_size, num_anchors, layer_h, layer_w)
        tw = torch.zeros(batch_size, num_anchors, layer_h, layer_w)
        th = torch.zeros(batch_size, num_anchors, layer_h, layer_w)
        tconf = torch.zeros(batch_size, num_anchors, layer_h, layer_w)
        tcls = torch.zeros(batch_size, num_anchors, layer_h, layer_w, num_classes)

        for b in range(batch_size):
            for t in range(target.shape[1]):
                if target[b, t].sum() <= 0:
                    continue
                gx = target[b, t, 0] * layer_w
                gy = target[b, t, 1] * layer_h
                gw = (target[b, t, 2] * layer_w).cpu()
                gh = (target[b, t, 3] * layer_h).cpu()
                gi = int(gx)
                gj = int(gy)

                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((num_anchors, 2)), np.array(scaled_anchors)), 1))

                calc_iou = bbox_iou(gt_box,anchor_shapes, x1y1x2y2=False)
                best_n = np.argmax(calc_iou)
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj
                tw[b, best_n, gj, gi] = math.log(gw/scaled_anchors[best_n][0] + 1e-16)
                th[b, best_n, gj, gi] = math.log(gh/scaled_anchors[best_n][1] + 1e-16)
                tconf[b, best_n, gj, gi] = 1
                tcls[b, best_n, gj, gi, int(target[b, t, 4])] = 1

        tx = tx.unsqueeze(-1)
        ty = ty.unsqueeze(-1)
        tw = tw.unsqueeze(-1)
        th = th.unsqueeze(-1)
        tconf = tconf.unsqueeze(-1)
        
        output = torch.cat([tx, ty, tw, th, tconf, tcls], -1) # [batch_size, num_anchors, layer_h, layer_w, 5(tx, ty, tw, th, tconf) + num_classes]
        output = output.permute(0, 1, 4, 2, 3).contiguous().view(batch_size, -1, layer_h, layer_w) # [batch_size, num_anchors*(5(tx, ty, tw, th, tconf) + num_classes), layer_h, layer_w]
        
        if torch.cuda.is_available():
            output = output.cuda()
        
        return output
    
    def decode_boxes(self, input, anchors, num_classes, image_size, gt=False):
        num_anchors = len(anchors)
        bbox_attrs = 5+num_classes
        bs = input.size(0)
        in_w = input.size(2)
        in_h = input.size(3)
        stride_h = image_size[1] / in_h
        stride_w = image_size[0] / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in anchors]
        prediction = input.view(bs, num_anchors, bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        
        # Get outputs
        if gt:
            x = prediction[..., 0]          # Center x
            y = prediction[..., 1]          # Center y
            w = prediction[..., 2]          # Width
            h = prediction[..., 3]          # Height
            conf = prediction[..., 4]       # Conf
            pred_cls = prediction[..., 5:]  # Cls pred.
            
        else:
            x = torch.sigmoid(prediction[..., 0])          # Center x
            y = torch.sigmoid(prediction[..., 1])          # Center y
            w = prediction[..., 2]                         # Width
            h = prediction[..., 3]                         # Height
            conf = torch.sigmoid(prediction[..., 4])       # Conf
            pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
        
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        # Calculate offsets for each grid
        grid_x = torch.linspace(0, in_w-1, in_w).repeat(in_w, 1).repeat(
            bs * num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h-1, in_h).repeat(in_h, 1).t().repeat(
            bs * num_anchors, 1, 1).view(y.shape).type(FloatTensor)
        # Calculate anchor w, h
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        # Results
        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale,
                            conf.view(bs, -1, 1), pred_cls.view(bs, -1, num_classes)), -1)
        return output.data

    def mean_average_precision(self, true_boxes, pred_boxes, num_classes, iou_threshold=0.5):
        """Calculates mean average precision

        Arguments:
            true_boxes (Tensor): Tensor of all boxes with all images (None, 8), specified as [img_idx, x1, y1, x2, y2, object_conf, class_score, class_pred]
            pred_boxes (Tensor): Similar as true_bboxes
            num_classes (int): number of classes
            iou_threshold (float): threshold where predicted boxes is correct

        Returns:
            Float: mAP value across all classes given a specific IoU threshold
        """
        # list storing all AP for respective classes
        average_precisions = []

        # used for numerical stability later on
        epsilon = 1e-6

        for c in torch.arange(num_classes, dtype=torch.float32):
            # print('\nCalculating AP: ', int(c), ' / ', num_classes)

            # detections, ground_truths variables in specific class
            detections = pred_boxes[torch.where(pred_boxes[..., -1] == c)[0]]
            ground_truths = true_boxes[torch.where(true_boxes[..., -1] == c)[0]]

            # If none exists for this class then we can safely skip
            total_true_boxes = len(ground_truths)
            if total_true_boxes == 0:
                average_precisions.append(torch.tensor(0))
                continue

            # print(c, ' class ground truths size: ', ground_truths.size()[0])
            # print(c, ' class detections size: ', detections.size()[0])

            # find the amount of bboxes for each training example
            # Counter here finds how many ground truth bboxes we get
            # for each training example, so let's say img 0 has 3,
            # img 1 has 5 then we will obtain a dictionary with:
            # amount_bboxes = {0:3, 1:5}
            amount_bboxes = Counter([int(gt[0]) for gt in ground_truths])

            # We then go through each key, val in this dictionary
            # and convert to the following (w.r.t same example):
            # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val)
            
            # sort by confidence score
            detections = detections[torch.sort(detections[..., 5], descending=True)[1]]
            true_positive = torch.zeros((len(detections)))
            false_positive = torch.zeros((len(detections)))

            for detection_idx, detection in enumerate(detections):
                # Only take out the ground_truths that have the same
                # training idx as detection
                ground_truth_img = ground_truths[torch.where(ground_truths[..., 0] == detection[0])[0]]

                num_gts = len(ground_truth_img)
                best_iou = 0

                for idx, gt in enumerate(ground_truth_img):
                    iou = bbox_iou(detection[1:5].unsqueeze(0), gt[1:5].unsqueeze(0), x1y1x2y2=True)
                    # print(f'\niou: {iou}\n')
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx
                        # print(f'\nbest_iou: {best_iou}\n')

                if best_iou > iou_threshold:
                    # only detect ground truth detection once
                    if amount_bboxes[int(detection[0])][best_gt_idx] == 0:
                        # true positive and add this bounding box to seen
                        true_positive[detection_idx] = 1
                        amount_bboxes[int(detection[0])][best_gt_idx] = 1
                    else:
                        false_positive[detection_idx] = 1

                # if IOU is lower then the detection is a false positive
                else:
                    false_positive[detection_idx] = 1

            tf_cumsum = torch.cumsum(true_positive, dim=0)
            fp_cumsum = torch.cumsum(false_positive, dim=0)
            recalls = tf_cumsum / (total_true_boxes + epsilon)
            precisions = torch.divide(tf_cumsum, (tf_cumsum + fp_cumsum + epsilon))
            precisions = torch.cat((torch.tensor([1]), precisions))
            recalls = torch.cat((torch.tensor([0]), recalls))
            # torch.trapz for numerical integration
            average_precisions.append(torch.trapz(precisions, recalls))

        return torch.mean(torch.stack(average_precisions))
