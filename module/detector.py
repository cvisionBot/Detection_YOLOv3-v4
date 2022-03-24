import pytorch_lightning as pl

from torch import nn
from utils.module_select import get_optimizer
from models.loss.yolo_loss import YOLO_Loss
from models.detector.anchors import pascal_voc, coco

class Detector(pl.LightningModule):
    def __init__(self, model, cfg, epoch_length=None):
        super(Detector, self).__init__()
        self.save_hyperparameters(ignore='model')
        self.model = model
        self.branch1_anchors = pascal_voc['anchors'][0]
        self.branch2_anchors = pascal_voc['anchors'][1]
        self.branch3_anchors = pascal_voc['anchors'][2]
        self.loss_fn = YOLO_Loss(cfg)


    def forward(self, input):
        branch1, branch2, branch3 = self.model(input)
        return branch1, branch2, branch3


    def training_step(self, batch, batch_idx):
        loss = self.common_opt_training_step(batch)
        return loss


    def common_opt_training_step(self, batch):
        branch1, branch2, branch3 = self.model(batch['img'])
        branch1_loss= self.loss_fn
        loss = self.loss_fn([branch1, branch2, branch3, batch])
        return loss


    def configure_optimizer(self):
        cfg = self.hparams.cfg
        optim = get_optimizer(cfg['optimizer'],
            params=self.model.parameters(),
            **cfg['optimizer_options']
        )
        return optim

    
    
