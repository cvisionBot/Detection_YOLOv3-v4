import pytorch_lightning as pl

from torch import nn
from utils.module_select import get_optimizer
from models.loss.yolo_loss import YOLO_Loss

class Detector(pl.LightningModule):
    def __init__(self, model, cfg, epoch_length=None):
        super(Detector, self).__init__()
        self.save_hyperparameters(ignore='model')
        self.model = model
        self.loss_fn = YOLO_Loss(cfg)


    def forward(self, input):
        output1, output2, output3 = self.model(input)
        return output1, output2, output3


    def training_step(self, batch, batch_idx):
        loss = self.common_opt_training_step(batch)
        return loss


    def common_opt_training_step(self, batch):
        output1, output2, output3 = self.model(batch['img'])
        loss = self.loss_fn([output1, output2, output3])
        return loss


    def configure_optimizer(self):
        cfg = self.hparams.cfg
        optim = get_optimizer(cfg['optimizer'],
            params=self.model.parameters(),
            **cfg['optimizer_options']
        )
        return optim

    
    
