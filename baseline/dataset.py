import os
from typing import NamedTuple

import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as  tvt
from torchvision.io import read_image


class Viewpoint(NamedTuple):
    """
    Four output variables in OODCV pose est challenge
    """
    azi: int
    theta: int
    elev: int
    dist: int

def collate(data):
    # TODO: Implement your function
    # But I guess in your case it should be:
    return torch.stack([a[0] for a in data]), ([a[1] for a in data])

def collate_test(data):
    # TODO: Implement your function
    # But I guess in your case it should be:
    return torch.stack([a[0] for a in data]), ([a[1] for a in data]), ([a[2] for a in data])

class PoseCategoryDataset(Dataset):
    '''
    Basic pose estimation dataset. Bucketizes angles into buckets of size pi/6.
    '''
    def __init__(self, *, root, labels_name, category, target, transforms=None):
        self.root = root
        self.transforms = transforms
        raw_df = pd.read_csv(root + labels_name + '.csv')
        self.manifest = raw_df.loc[raw_df['cls_name'] == category].reset_index()
        self.target = target

    def __getitem__(self, idx):
        im_data = self.manifest.iloc[idx]
        #print(im_data['im_path'])
        image = read_image(self.root + im_data['im_path']).float()
        if image.shape[0] != 3:
            image = torch.cat([image, image, image], dim=0)
        #print(image)
        if self.transforms:
            image = self.transforms(image)
        if self.target == 'azimuth':
            azi = int(im_data['azimuth'] // 30)
            return image, azi
        elif self.target == 'theta':
            theta = int(im_data['inplane_rotation']%360 // 30)
            return image, theta
        elif self.target == 'elevation':
            elev = int(abs(im_data['elevation']) // 9)
            return image, elev
        elif self.target == 'distance':
            dist = int(im_data['distance'] // 10)
            return image, dist
        else:
            return None
    def __len__(self):
        return self.manifest.shape[0]


class UnlabeledPoseDataset(Dataset):
    '''
    Basic pose estimation dataset. Bucketizes angles into buckets of size pi/6.
    '''
    def __init__(self, *, root, labels_name, category, transforms=None):
        self.root = root
        self.transforms = transforms
        raw_df = pd.read_csv(root + labels_name + '.csv')
        self.manifest = raw_df.loc[raw_df['cls_name'] == category].reset_index()

    def __getitem__(self, idx):
        im_data = self.manifest.iloc[idx]
        #print(im_data['im_path'])
        image = read_image(self.root + im_data['im_path']).float()
        if image.shape[0] != 3:
            image = torch.cat([image, image, image], dim=0)
        #print(image)
        if self.transforms:
            image = self.transforms(image)
        return image, im_data['source'] + '_' + im_data['cls_name'] + '_' + im_data['im_name'] + '_' + str(im_data['object']), im_data['cls_name']

    def __len__(self):  
        return self.manifest.shape[0]

train_transforms = tvt.Compose([
    tvt.Resize([224, 224]),
    
    #tvt.ColorJitter(brightness=0.3, hue=0.1, saturation=0.2),
    tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

test_transforms = tvt.Compose([
    tvt.Resize([224, 224]),    
    tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    
if __name__ == "__main__":
    d1 = UnlabeledPoseDataset(root='/research/cwloka/projects/dpitt/ROBIN-dataset/ROBINv1.1/iid_test/', labels_name='iid',\
         category='aeroplane',  transforms=train_transforms)
    print(d1.manifest)
    print(d1[20])
    train_loader = DataLoader(d1, batch_size=1, num_workers=4, persistent_workers=True, shuffle=True, collate_fn=collate)
    for idx, b in enumerate(train_loader):
        print(b)