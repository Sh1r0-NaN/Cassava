import os
import sys
import cv2
import numpy as np
import pandas as pd

import argparse
import timm
import torch
import torch.nn as nn
import albumentations as A
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from albumentations.core.composition import Compose, OneOf
from albumentations.augmentations.transforms import CLAHE, GaussNoise, ISONoise
from albumentations.pytorch import ToTensorV2

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import StratifiedKFold

from adabelief_pytorch import AdaBelief
from loss import bi_tempered_logistic_loss

package_path = '../input/image-fmix/FMix-master'
sys.path.append(package_path)

from fmix import sample_mask

# OUTPUT_DIR = './'
# if not os.path.exists(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)

TRAIN_PATH = "../input/cassava-leaf-disease-classification/train_images"

DEBUG = False
USE_2019 = True


class CFG:
    seed = 1105
    model_name = 'resnet50'
    pretrained = True
    img_size = 512
    num_classes = 5
    lr = 5e-4
    min_lr = 1e-6
    t_max = 30
    num_epochs = 30
    batch_size = 32
    accum = 1
    precision = 16
    n_fold = 5
    smoothing = 0.1
    t1 = 0.8
    t2 = 1.2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data, targets


def fmix(data, targets, alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)
    # mask =torch.tensor(mask, device=device).float()
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    if CFG.precision == 16:
        x1 = torch.from_numpy(mask).to(CFG.device, dtype=torch.half) * data
        x2 = torch.from_numpy(1 - mask).to(CFG.device, dtype=torch.half) * shuffled_data
    else:
        x1 = torch.from_numpy(mask).to(CFG.device) * data
        x2 = torch.from_numpy(1 - mask).to(CFG.device) * shuffled_data
    targets = (targets, shuffled_targets, lam)

    return (x1 + x2), targets


class TrainDataset(Dataset):
    def __init__(self, df, data_path, phase, transform=None, soft_transform=None, hard_transform=None):
        self.df = df
        self.data_path = data_path
        self.image_path = df['image_id'].values
        self.labels = df['label'].values
        self.phase = phase
        self.transform = transform
        self.soft_transform = soft_transform
        self.hard_transform = hard_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.image_path[idx]
        image_path = self.image_path[idx]
        image_path = f'{self.data_path}/{image_path}'
        image_ori = cv2.imread(image_path)
        image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
        if self.phase == 'train':
            augmented = self.soft_transform(image=image_ori)
            image = augmented['image']
            if self.hard_transform is not None:
                augmented = self.hard_transform(image=image)
                image_hard = augmented['image']

            augmented = self.transform(image=image)
            image = augmented['image']
            augmented = self.transform(image=image_hard)
            image_hard = augmented['image']
        else:
            augmented = self.transform(image=image_ori)
            image = augmented['image']
            image_hard = image

        target = torch.tensor(self.labels[idx])
        sample = {'image_path': image_path, 'image': image, 'image_hard': image_hard, 'target': np.array(target).astype('int64')}
        return sample


def get_transforms(*, phase):
    if phase == 'soft':
        return Compose([
            A.RandomResizedCrop(height=CFG.img_size, width=CFG.img_size),
            A.Flip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
                A.CLAHE(p=0.5),
            ], p=0.5),
        ])
    elif phase == 'hard':
        return Compose([
            A.OneOf([
                A.Blur(p=0.1),
                A.GaussianBlur(p=0.1),
                A.MotionBlur(p=0.1),
            ], p=0.1),
            A.OneOf([
                A.GaussNoise(p=0.1),
                A.ISONoise(p=0.1),
                A.GridDropout(ratio=0.5, p=0.2),
                A.CoarseDropout(max_holes=16, min_holes=8, max_height=16, max_width=16, min_height=8, min_width=8, p=0.2)
            ], p=0.2),
        ])
    elif phase == 'train':
        return Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    elif phase == 'valid':
        return Compose([
            A.Resize(height=CFG.img_size, width=CFG.img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


class CustomResNet(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.get_classifier().in_features
        self.model.fc = nn.Linear(in_features, CFG.num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class CustomDenseNet(nn.Module):
    def __init__(self, model_name='', pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.get_classifier().in_features
        self.model.classifier = nn.Linear(in_features, CFG.num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class CustomEffiNet(nn.Module):
    def __init__(self, model_name='', pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.get_classifier().in_features
        self.model.classifier = nn.Linear(in_features, CFG.num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class LitCassava(pl.LightningModule):
    def __init__(self, model):
        super(LitCassava, self).__init__()
        self.model = model
        self.metric = pl.metrics.Accuracy()
#         self.criterion = nn.CrossEntropyLoss()
        self.lr = CFG.lr

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=CFG.t_max, eta_min=CFG.min_lr)

        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}

    def training_step(self, batch, batch_idx):
        image = batch['image']
        image_hard = batch['image_hard']
        target = batch['target']
#         logits = self.model(image)
#         loss = self.criterion(logits, target)
#         loss = bi_tempered_logistic_loss(logits, target, t1=CFG.t1, t2=CFG.t2, label_smoothing=CFG.smoothing)
#         score = self.metric(logits.argmax(1), target)
#         logs = {'train_loss': loss, 'train_accuracy': score, 'lr': self.optimizer.param_groups[0]['lr']}
#         self.log_dict(
#             logs,
#             on_step=False, on_epoch=True, prog_bar=True, logger=True
#         )

        if (self.current_epoch + 10) < CFG.num_epochs:
            mix_decision = np.random.rand()
            if mix_decision < 0.25:
                image, target = cutmix(image, target, 1.)
            elif mix_decision >= 0.25 and mix_decision < 0.5:
                image, target = fmix(image, target, alpha=1., decay_power=5., shape=(CFG.img_size,CFG.img_size))

            logits = self.model(image)

            if mix_decision < 0.5:
                loss = bi_tempered_logistic_loss(logits, target[0], t1=CFG.t1, t2=CFG.t2, label_smoothing=CFG.smoothing) * target[2] \
                       + bi_tempered_logistic_loss(logits, target[1], t1=CFG.t1, t2=CFG.t2, label_smoothing=CFG.smoothing) * (1. - target[2])
                score = self.metric(logits.argmax(1), target[0])
            else:
                loss = bi_tempered_logistic_loss(logits, target, t1=CFG.t1, t2=CFG.t2, label_smoothing=CFG.smoothing)
                score = self.metric(logits.argmax(1), target)
        else:
            logits = self.model(image_hard)
            loss = bi_tempered_logistic_loss(logits, target, t1=CFG.t1, t2=CFG.t2, label_smoothing=CFG.smoothing)
            score = self.metric(logits.argmax(1), target)

        logs = {'train_loss': loss, 'train_accuracy': score, 'lr': self.optimizer.param_groups[0]['lr']}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        target = batch['target']
        logits = self.model(image)
#         loss = self.criterion(logits, target)
        loss = bi_tempered_logistic_loss(logits, target, t1=CFG.t1, t2=CFG.t2, label_smoothing=CFG.smoothing)
        score = self.metric(logits.argmax(1), target)
        logs = {'valid_loss': loss, 'valid_accuracy': score}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss


def split_data(df_all):
    folds = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)

    df_all['fold'] = 0
    for i, (train_indexes, valid_indexes) in enumerate(folds.split(df_all, df_all['label'])):
        df_all.loc[valid_indexes, 'fold'] = i
    return df_all


def create_dataloader(df_all, df_2019, data_path, fold):
    df_train = df_all[df_all['fold'] != fold]
    df_valid = df_all[df_all['fold'] == fold]
    # marge data in 2019
    if USE_2019:
        df_train = pd.concat([df_train, df_2019])

    soft_transform = get_transforms(phase='soft')
    hard_transform = get_transforms(phase='hard')

    train_dataset = TrainDataset(df_train, data_path=data_path, phase='train', transform=get_transforms(phase='train'), soft_transform=soft_transform, hard_transform=hard_transform)
    valid_dataset = TrainDataset(df_valid, data_path=data_path, phase='valid', transform=get_transforms(phase='valid'))

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4)
    return train_loader, valid_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()

    df_2019 = pd.read_csv("../input/cassava-leaf-disease-merged/merged.csv")
    df_2019 = df_2019[df_2019['source'] == 2019]
    df_2019 = df_2019.drop('source', axis=1)
    print('df_2019 : ', df_2019.shape)

    df_all = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
    if DEBUG:
        df_all = df_all.iloc[:200]
        CFG.num_epochs = 10
    print('df_2020 : ', df_all.shape)

    seed_everything(CFG.seed)
    df_all = split_data(df_all)
    print('=' * 10, f"Start Fold:{args.fold}", '=' * 10)
    train_loader, valid_loader = create_dataloader(df_all, df_2019, "../input/cassava-leaf-disease-merged/train", fold=args.fold)

    model = CustomResNet(model_name=CFG.model_name, pretrained=CFG.pretrained)
    # model = CustomDenseNet(model_name=CFG.model_name, pretrained=CFG.pretrained)
    # print(model)

    lit_model = LitCassava(model)
    logger = CSVLogger(save_dir='logs/', name=CFG.model_name)
    logger.log_hyperparams(args)
    logger.log_hyperparams(CFG.__dict__)
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss',
                                          save_top_k=1,
                                          save_last=True,
                                          save_weights_only=True,
                                          filepath=logger.log_dir+'/checkpoint/{epoch}_{valid_loss:.4f}_{valid_accuracy:.4f}',
                                          verbose=False,
                                          mode='min')

    trainer = Trainer(
        max_epochs=CFG.num_epochs,
        gpus=1,
        accumulate_grad_batches=CFG.accum,
        precision=CFG.precision,
        # callbacks=[EarlyStopping(monitor='valid_loss', patience=10, mode='min')],
        checkpoint_callback=checkpoint_callback,
        logger=logger,
        weights_summary='top',
    )

    trainer.fit(lit_model, train_dataloader=train_loader, val_dataloaders=valid_loader)

    metrics = pd.read_csv(f'{trainer.logger.log_dir}/metrics.csv')

    train_acc = metrics['train_accuracy'].dropna().reset_index(drop=True)
    valid_acc = metrics['valid_accuracy'].dropna().reset_index(drop=True)

    fig = plt.figure(figsize=(7, 6))
    plt.grid(True)
    plt.plot(train_acc, color="r", marker="o", label='train/acc')
    plt.plot(valid_acc, color="b", marker="x", label='valid/acc')
    plt.ylabel('Accuracy', fontsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.legend(loc='lower right', fontsize=18)
    plt.savefig(f'{trainer.logger.log_dir}/accuracy.png')

    train_loss = metrics['train_loss'].dropna().reset_index(drop=True)
    valid_loss = metrics['valid_loss'].dropna().reset_index(drop=True)

    fig = plt.figure(figsize=(7, 6))
    plt.grid(True)
    plt.plot(train_loss, color="r", marker="o", label='train/loss')
    plt.plot(valid_loss, color="b", marker="x", label='valid/loss')
    plt.ylabel('Loss', fontsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.legend(loc='upper right', fontsize=18)
    plt.savefig(f'{trainer.logger.log_dir}/loss.png')\

    lr = metrics['lr'].dropna().reset_index(drop=True)

    fig = plt.figure(figsize=(7, 6))
    plt.grid(True)
    plt.plot(lr, color="g", marker="o", label='learning rate')
    plt.ylabel('LR', fontsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.legend(loc='upper right', fontsize=18)
    plt.savefig(f'{trainer.logger.log_dir}/lr.png')


if __name__ == '__main__':
    main()
