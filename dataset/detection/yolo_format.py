import cv2
import glob
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from dataset.detection.utils import collater

class YoloDataset(Dataset):
    def __init__(self, transforms, path=None):
        super(YoloDataset, self).__init__()
        self.transforms = transforms
        self.image = glob.glob(path + '/*.jpg')

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_file = self.image[index]
        image = cv2.imread(image_file)
        boxes = self.load_annotation(image_file, image.shape)
        transformed = self.transforms(image=image, bboxes=boxes)
        return transformed 

    def load_annotation(self, img_file, img_shape):
        img_h, img_w, _ = img_shape
        annotation_file = img_file.replace('.jpg', '.txt')
        boxes = np.zeros((0, 5))
        with open(annotation_file, 'r') as f:
            annotations = f.read().splitlines()
            for annot in annotations:
                cid, cx, cy, w, h = map(float, annot.split(' '))
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                w *= img_w
                h *= img_h
                annotation = np.array([[x1, y1, w, h, cid]])
                boxes = np.append(boxes, annotation, axis=0)
        return boxes


class YoloFormat(pl.LightningDataModule):
    def __init__(self, train_path, val_path, workers, train_transforms,
                 val_transforms,
                 batch_size=None):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.batch_size = batch_size
        self.workers = workers

    def train_dataloader(self):
        return DataLoader(YoloDataset(
            transforms=self.train_transforms,
            path=self.train_path),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0,
            collate_fn=collater)

    def val_dataloader(self):
        return DataLoader(YoloDataset(
            transforms=self.val_transforms,
            path=self.val_path),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0,
            collate_fn=collater)


if __name__ == '__main__':
    """
    Data loader 테스트 코드
    python -m dataset.detection.yolo_format
    """
    import albumentations
    import albumentations.pytorch
    from dataset.detection.utils import visualize

    train_transforms = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ColorJitter(),
        albumentations.RandomResizedCrop(416, 416, (0.8, 1)),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ], bbox_params=albumentations.BboxParams(format='coco', min_visibility=0.1))

    loader = DataLoader(YoloDataset(
        transforms=train_transforms, path='/mnt/det_test'),
        batch_size=1, shuffle=True, collate_fn=collater)

    for batch, sample in enumerate(loader):
        imgs = sample['img']
        annots = sample['annot']
        visualize(imgs, annots)