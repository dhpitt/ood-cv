import os
import multiprocessing
from pathlib import Path
from PIL import Image
import argparse
import random

import torch
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from model import SpecifiedResNet

from dataset import PoseCategoryDataset, train_transforms, test_transforms


# backbone model, a resnet


# constants

BATCH_SIZE = 24
EPOCHS     = 1000
LR         = 3e-5
NUM_GPUS   = 1
#NUM_WORKERS = multiprocessing.cpu_count()
NUM_WORKERS = 4 * NUM_GPUS # according to PyTorch forum

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
        self.log(name='loss', value=loss, sync_dist=True)
        return {'loss': loss}

    def validation_step(self, images, _):
        acc = self.validate(images)
        return {'acc pi/6': acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)



# main

if __name__ == '__main__':
    # arguments

    parser = argparse.ArgumentParser(description='byfl-lightning')

    parser.add_argument('--train_root', type=str, required=True,\
                        help='path to your folder of training images')
    parser.add_argument('--val', type=str, required=True,\
                        help='path to your folder of validation images')
    args = parser.parse_args()

    train_ds = PoseCategoryDataset(root=args.train_root, labels_name='train', category='aeroplane', target='azimuth', transforms=train_transforms)
    val_ds = PoseCategoryDataset(root=args.test_root, labels_name='iid_test', category='aeroplane', target='azimuth', transforms=train_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        persistent_workers=True, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        persistent_workers=True, shuffle=False)

    # Checkpoint every epoch
    checkpoint_callback = pl.callbacks.ModelCheckpoint(every_n_epochs=1)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=NUM_GPUS,
        max_epochs=EPOCHS,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        callbacks=[checkpoint_callback],
    )
    
    model = SupervisedLearner(target='azimuth')
    
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)