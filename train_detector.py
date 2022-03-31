import argparse
import platform
import albumentations
import albumentations.pytorch
import pytorch_lightning as pl
from torch import Tensor

from utils.module_select import get_model
from utils.utility import make_model_name
from utils.yaml_helper import get_train_configs

from dataset.detection import yolo_format
from module.detector import Detector
from models.detector.yolov3 import YOLOv3

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging, QuantizationAwareTraining


def train(cfg, ckpt=None):
    input_size = cfg['input_size']
    train_transforms = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ColorJitter(),
        albumentations.RandomResizedCrop(input_size, input_size, (0.8, 1)),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ], bbox_params=albumentations.BboxParams(format='yolo', min_visibility=0.1))

    valid_transform = albumentations.Compose([
        albumentations.Resize(input_size, input_size, always_apply=True),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ], bbox_params=albumentations.BboxParams(format='yolo', min_visibility=0.1))

    data_module = yolo_format.YoloFormat(
        train_path=cfg['train_path'], val_path=cfg['val_path'],
        workers=cfg['workers'], batch_size=cfg['batch_size'],
        train_transforms=train_transforms, val_transforms=valid_transform
    )

    backbone = get_model(cfg['backbone'])
    model = YOLOv3(Backbone=backbone, num_classes=cfg['classes'], in_channels=cfg['in_channels'], varient=cfg['varient'])
    model_module = Detector(model, cfg, epoch_length=data_module.train_dataloader().__len__())

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(monitor='val_loss', save_last=True, every_n_epochs=cfg['save_freq']),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        logger=TensorBoardLogger(cfg['save_dir'], make_model_name(cfg)),
        gpus=cfg['gpus'],
        accelerator='ddp' if platform.system() != 'Windows' else None,
        plugins=DDPPlugin() if platform.system() != 'Windows' else None,
        callbacks=callbacks,
        resume_from_checkpoint=ckpt,
        **cfg['trainer_options'],
    )
    trainer.fit(model_module, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='Train config file')
    parser.add_argument('--ckpt', required=False, type=str, help='Train checkpoint')
    args = parser.parse_args()
    cfg = get_train_configs(args.cfg)
    train(cfg, args.ckpt)
