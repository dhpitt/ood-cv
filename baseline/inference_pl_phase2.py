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
import pandas as pd

from model import SpecifiedResNet

from dataset import Phase2PoseDataset, test_transforms, collate


# backbone model, a resnet


# constants

BATCH_SIZE = 1
LR         = 3e-6
NUM_GPUS   = 1
EPOCHS     = 10
NUM_WORKERS = mp.cpu_count()


# pytorch lightning module

class SupervisedLearner(pl.LightningModule):
    def __init__(self, target, **kwargs):
        super().__init__()
        out_bins = None
        if target == 'azimuth' or target == 'theta':
            out_bins = 12
        elif target == 'distance':
            out_bins = 10
        elif target == 'elevation':
            out_bins = 10
        self.learner = SpecifiedResNet(out_bins=out_bins)
        self.save_hyperparameters(ignore=['net'])
        self.df = pd.DataFrame({'imgs': [], 'data':[]})

    def forward(self, images):
        return self.learner(images)

    def test_step(self, images, _):
        y_hat, imname = self.learner.unlabeled_inference_p2(images)
        #input = pd.Series({'imgs':names, 'labels': clslabel, 'data':y_hat})
        self.df.loc[len(self.df.index)] = [str(imname), y_hat]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

# main

if __name__ == '__main__':
    # arguments

    parser = argparse.ArgumentParser(description='6dpose-lightning')

    parser.add_argument('--test_root', type=str, required=True,\
                        help='path to your folder of testing images')
    parser.add_argument('--ckpt', type=str, required=False,\
        help='path to a saved model checkpoint')
    args = parser.parse_args()

    categories = ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train']
    targets = ['azimuth', 'theta', 'elevation', 'distance']

    save_labels_root = './pose_res/'

    results = None
    for category in categories:
        running_df = None
        for target in targets:
            
            test_ds = Phase2PoseDataset(root=args.test_root, category=category, transforms=test_transforms)

            test_loader = DataLoader(test_ds, batch_size=1, num_workers=NUM_WORKERS,
                persistent_workers=True, shuffle=False, collate_fn=collate)

            if args.ckpt is not None:
                checkpoint_path = '{}/{}_{}/'.format(args.ckpt,category, target)
            else:
                checkpoint_path = './checkpoints/{}_{}/'.format(category, target)
            valid_checkpoints = os.listdir(checkpoint_path)
            ckpt = checkpoint_path + valid_checkpoints[0]

            trainer = pl.Trainer(
                accelerator='gpu',
                devices=NUM_GPUS,
                max_epochs=EPOCHS,
                accumulate_grad_batches=1,
                sync_batchnorm=True,
                log_every_n_steps=1
            )
        
            model = SupervisedLearner(target=target).load_from_checkpoint(ckpt)
            trainer.test(model, dataloaders=test_loader)
            preds = trainer.model.df
                
            if target == 'azimuth' or target == 'theta':
                preds['data'] *= 30.
            elif target == 'distance':
                preds['data'] *= 10.
            elif target == 'elevation':
                preds['data'] *= 9.
            
            preds = preds.rename(columns={'data':target})

            if running_df is None:
                running_df = preds
            else:
                running_df = running_df.merge(right=preds, on='imgs', how='left').drop_duplicates()
            
            print(running_df)
        
        if results is None:
            results = running_df
        else:
            results = pd.concat([results, running_df])     
        print(results)
    results.to_csv(save_labels_root + 'preds.csv')
            

