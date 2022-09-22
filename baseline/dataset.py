import os
from typing import NamedTuple

import torch
from torch.utils.data import Dataset, dataloader
import torchvision.transforms as  tvt
import pandas as pd
from PIL import Image as PILImage

class Viewpoint(NamedTuple):
    """
    Four output variables in OODCV pose est challenge
    """
    azi: int
    theta: int
    elev: int
    dist: int


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
            image = PILImage.open(self.root + im_data['im_path'])
            if self.transforms:
                image = self.transforms(image)
            
            if self.target == 'azimuth':
                azi = int(im_data['azimuth'] // 30)
                return (image, azi)
            elif self.target == 'theta':
                theta = int(im_data['inplane_rotation'] // 30)
                return (image, theta)
            elif self.target == 'elevation':
                elev = int(im_data['elevation'])
                return (image, elev)
            elif self.target == 'distance':
                dist = int(im_data['distance'])
                return (image, dist)
            else:
                return None

train_transforms = tvt.Compose([
    tvt.Resize(224),
    tvt.ColorJitter(brightness=0.3, hue=0.1, saturation=0.2),
    tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    tvt.ToTensor()
    ])

test_transforms = tvt.Compose([
    tvt.Resize(224),
    tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    tvt.ToTensor()
    ])

    
if __name__ == "__main__":
    d1 = PoseCategoryDataset(root='/research/cwloka/projects/dpitt/ROBIN-dataset/ROBINv1.1/train/', labels_name='train', category='aeroplane', target='azimuth')
    print(d1.manifest)
    print(d1[20])