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
        self.branch1_loss = YOLO_Loss(cfg, self.branch1_anchors)
        self.branch2_loss = YOLO_Loss(cfg, self.branch2_anchors)
        self.branch3_loss = YOLO_Loss(cfg, self.branch3_anchors)


    def forward(self, input):
        branch1, branch2, branch3 = self.model(input)
        return branch1, branch2, branch3


    def training_step(self, batch, batch_idx):
        loss = self.opt_training_step(batch)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        val_branch1, val_branch2, val_branch3 = self.model(batch)
        val_branch1_loss = self.branch1_loss([val_branch1, batch])
        val_branch2_loss = self.branch2_loss([val_branch2, batch])
        val_branch3_loss = self.branch3_loss([val_branch3, batch])
        loss = val_branch1_loss + val_branch2_loss + val_branch3_loss
        self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        return loss


    def opt_training_step(self, batch):
        branch1, branch2, branch3 = self.model(batch['img'])
        branch1_loss = self.branch1_loss([branch1, batch])
        branch2_loss = self.branch2_loss([branch2, batch])
        branch3_loss = self.branch3_loss([branch3, batch])
        loss = branch1_loss + branch2_loss + branch3_loss
        return loss


    def configure_optimizer(self):
        cfg = self.hparams.cfg
        optim = get_optimizer(cfg['optimizer'],
            params=self.model.parameters(),
            **cfg['optimizer_options']
        )
        return optim

    
    
