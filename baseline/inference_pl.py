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

BATCH_SIZE = 1
LR         = 3e-6
NUM_GPUS   = 1
NUM_WORKERS = mp.cpu_count()


# pytorch lightning module

class SupervisedLearner(pl.LightningModule):
    def __init__(self, target, **kwargs):
        super().__init__()
        out_bins = None
        if target == 'azimuth' or target == 'theta':
            out_bins = 12
        elif target == 'distance':
            out_bins = 50
        elif target == 'elevation':
            out_bins = 50
        self.learner = SpecifiedResNet(out_bins=out_bins)
        self.save_hyperparameters(ignore=['net'])

    def forward(self, images):
        return self.learner(images)

    
    
    def test_step(self, batch, _):
        acc = self.learner.classify(images)
        self.log(name='val_acc', value=acc, batch_size=8)
        return {'acc pi/6': acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

# main

if __name__ == '__main__':
    # arguments

    parser = argparse.ArgumentParser(description='6dpose-lightning')

    parser.add_argument('--train_root', type=str, required=True,\
                        help='path to your folder of training images')
    parser.add_argument('--val_root', type=str, required=True,\
                        help='path to your folder of validation images')
    parser.add_argument('--ckpt', type=str, required=False,\
        help='path to a saved model checkpoint')
    args = parser.parse_args()
    print(args.train_root)

    categories = ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train']
    #targets = ['azimuth', 'theta', 'elevation', 'distance']
    targets = ['azimuth', 'theta']
    nuisance_types = ['occlusion', 'context','texture','shape','pose','weather']

    for target in targets:
        for category in categories:
            
            val_ds = PoseCategoryDataset(root=args.val_root, labels_name='iid', category=category, target=target, transforms=test_transforms)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                persistent_workers=True, shuffle=True, collate_fn=collate)
            val_loader = DataLoader(val_ds, batch_size=8, num_workers=NUM_WORKERS,
                persistent_workers=True, shuffle=False, collate_fn=collate)

            valid_checkpoints = os.listdir('./checkpoints/{}_{}/'.format(category, target))
            # Checkpoint every epoch
            checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='./checkpoints/{}_{}'.format(category, target), save_top_k=2, monitor='val_acc', mode='max')

            trainer = pl.Trainer(
                accelerator='gpu',
                devices=NUM_GPUS,
                max_epochs=EPOCHS,
                accumulate_grad_batches=1,
                sync_batchnorm=True,
                callbacks=[checkpoint_callback],
                log_every_n_steps=10
            )
            
            if args.ckpt is not None:
                model = SupervisedLearner(target='azimuth').load_from_checkpoint(args.ckpt.format(category))
                trainer.test(model, dataloaders=val_loader)
            else:
                model = SupervisedLearner(target=target)
                trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)