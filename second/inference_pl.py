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

from dataset import PoseCategoryDataset, UnlabeledPoseDataset, test_transforms, collate_test


# backbone model, a resnet


# constants

BATCH_SIZE = 1
LR         = 3e-6
NUM_GPUS   = 1
EPOCHS     = 1
NUM_WORKERS = mp.cpu_count()


# pytorch lightning module

class SupervisedLearner(pl.LightningModule):
    def __init__(self, target, **kwargs):
        super().__init__()
        out_bins = None
        if target == 'azimuth' or target == 'theta':
            out_bins = 12
        elif target == 'distance':
            out_bins = 5
        elif target == 'elevation':
            out_bins = 6
        self.learner = SpecifiedResNet(out_bins=out_bins)
        self.save_hyperparameters(ignore=['net'])
        self.df = pd.DataFrame({'imgs': [], 'labels':[], 'data':[]})

    def forward(self, images):
        return self.learner(images)

    def test_step(self, images, _):
        y_hat, names, clslabel = self.learner.unlabeled_inference(images)
        #input = pd.Series({'imgs':names, 'labels': clslabel, 'data':y_hat})
        self.df.loc[len(self.df.index)] = [str(names), str(clslabel), y_hat]
       # self.df = pd.concat([self.df, input], ignore_index=True, axis=0)

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

    parser.add_argument('--iid', required=False, action='store_true', default=False)
    args = parser.parse_args()

    categories = ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train']
    targets = ['azimuth', 'theta', 'elevation', 'distance']
    nuisances = ['iid', 'occlusion', 'context','texture','shape','pose','weather']
    #nuisances = ['iid']

    #categories = ['boat']
    #targets = ['elevation']
    #uisances = ['occlusion']

    save_labels_root = './pose_res/'

    for nuisance in nuisances:
        results = None
        for category in categories:
            running_df = None
            for target in targets:
                
                if nuisance == 'iid':   
                    test_ds = UnlabeledPoseDataset(root=args.test_root + 'iid_test/', labels_name=nuisance,\
                         category=category, transforms=test_transforms)
                else:
                    test_ds = UnlabeledPoseDataset(root=args.test_root + 'nuisances/' + nuisance + '/',\
                         labels_name=nuisance + '_bias', category=category, transforms=test_transforms)

                test_loader = DataLoader(test_ds, batch_size=1, num_workers=NUM_WORKERS,
                    persistent_workers=True, shuffle=False, collate_fn=collate_test)

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
                if target == 'distance':
                    preds['data'] *= 20.
                elif target == 'elevation':
                    preds['data'] *= 30.
                    preds['data'] -= 90.
                
                preds = preds.rename(columns={'data':target})

                if running_df is None:
                    running_df = preds
                else:
                    preds = preds.drop(columns=['labels'])
                    running_df = running_df.merge(right=preds, on='imgs', how='left').drop_duplicates()
                
                print(running_df)
            
            if results is None:
                results = running_df
            else:
                results = pd.concat([results, running_df])     
            print(results)
        results.to_csv(save_labels_root + '{}.csv'.format(nuisance))
            

