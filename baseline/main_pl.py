import os
import multiprocessing as mp
from pathlib import Path
from PIL import Image
import argparse
import random

import torch
from torchvision import models
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from model import SpecifiedResNet

from dataset import PoseCategoryDataset, train_transforms, test_transforms, collate


# backbone model, a resnet


# constants

BATCH_SIZE = 24
EPOCHS     = 200
LR         = 3e-5
NUM_GPUS   = 1
NUM_WORKERS = mp.cpu_count()

# pytorch lightning module

class SupervisedLearner(pl.LightningModule):
    def __init__(self, target, **kwargs):
        super().__init__()
        out_bins = None
        if target == 'azimuth' or target == 'inplane_rotation':
            out_bins = 12
        elif target == 'distance':
            out_bins = 20
        elif target == 'elevation':
            out_bins = 20
        self.learner = SpecifiedResNet(out_bins=out_bins)
        self.save_hyperparameters(ignore=['net'])

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        self.log(name='loss', value=loss, batch_size=BATCH_SIZE)
        return {'loss': loss}

    def validation_step(self, images, _):
        acc = self.learner.classify(images)
        self.log(name='acc', value=acc, batch_size=8)
        return {'acc pi/6': acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)



# main

if __name__ == '__main__':
    # arguments

    parser = argparse.ArgumentParser(description='byfl-lightning')

    parser.add_argument('--train_root', type=str, required=True,\
                        help='path to your folder of training images')
    parser.add_argument('--val_root', type=str, required=True,\
                        help='path to your folder of validation images')
    args = parser.parse_args()

    # train_ds = PoseCategoryDataset(root=args.train_root, labels_name='train', category='aeroplane', target='azimuth')
    # val_ds = PoseCategoryDataset(root=args.val_root, labels_name='iid', category='aeroplane', target='azimuth')
    train_ds = PoseCategoryDataset(root=args.train_root, labels_name='train', category='aeroplane', target='azimuth', transforms=train_transforms)
    val_ds = PoseCategoryDataset(root=args.val_root, labels_name='iid', category='aeroplane', target='azimuth', transforms=test_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        persistent_workers=True, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=8, num_workers=NUM_WORKERS,
        persistent_workers=True, shuffle=False, collate_fn=collate)

    # Checkpoint every epoch
    checkpoint_callback = pl.callbacks.ModelCheckpoint(every_n_epochs=1)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=NUM_GPUS,
        max_epochs=EPOCHS,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )
    
    model = SupervisedLearner(target='azimuth')
    
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)