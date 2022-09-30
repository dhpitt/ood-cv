import os
from typing import NamedTuple
import math

import numpy as np
import pandas as pd
from PIL import Image
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as  tvt
from torchvision.io import read_image
from torchvision.utils import save_image


class Viewpoint(NamedTuple):
    """
    Four output variables in OODCV pose est challenge
    """
    azi: int
    theta: int
    elev: int
    dist: int

class GaussianNoiseTransform(object):
    '''Custom transform to add Gaussian noise to an image'''
    def __init__(self, mu=0, sigmasq=0.05):
        self.mu = mu
        self.sigmasq = sigmasq

    def __repr__(self):
        return 'GaussianNoiseTransform object'

    def __call__(self, img):
        '''
        Applies gaussian noise to a multi-channel image
        img: torch.Tensor of shape c x h x w
        '''
        noise = torch.randn(size=img.size()) * math.sqrt(self.sigmasq)
        img += noise
        return img

def collate(data):
    return torch.stack([a[0] for a in data]), ([a[1] for a in data])

def collate_test(data):
    return torch.stack([a[0] for a in data]), ([a[1] for a in data]), ([a[2] for a in data])

def collate_with_bboxcrop(data):
    return torch.stack([a[0] for a in data]), torch.stack([a[1] for a in data]), ([a[2] for a in data])

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
        #image = read_image(self.root + im_data['im_path']).float()
        #print(self.root + im_data['im_path'])
        image = Image.open(self.root + im_data['im_path'])
        if len(image.getbands()) != 3:
            image = image.convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        if self.target == 'azimuth':
            azi = int(im_data['azimuth'] // 30)
            return image, azi
        elif self.target == 'theta':
            theta = int(im_data['inplane_rotation']%360 // 30)
            return image, theta
        elif self.target == 'elevation':
            elev = int((im_data['elevation'] + 90) // 30)
            return image, elev
        elif self.target == 'distance':
            dist = int(im_data['distance'] // 20)
            return image, dist
        else:
            return None
    def __len__(self):
        return self.manifest.shape[0]

class PoseBboxCategoryDataset(Dataset):
    '''
    Pose estimation dataset that has bounding box crops as an augmentation.
    '''
    def __init__(self, *, root, anno_root, labels_name, category, target, transforms=None):
        self.root = root
        self.anno_root = anno_root
        self.transforms = transforms
        raw_df = pd.read_csv(root + labels_name + '.csv')
        self.manifest = raw_df.loc[raw_df['cls_name'] == category].reset_index()
        self.target = target
        self.category = category

    def __getitem__(self, idx):
        im_data = self.manifest.iloc[idx]
        #print(im_data['im_path'])
        image = read_image(self.root + im_data['im_path']).float()
        
        # Load bounding box from annotations
        anno_path = self.anno_root + '{}_{}/{}.mat'.format(self.category, im_data['source'], im_data['im_name'])
        objects = sio.loadmat(anno_path)['record']['objects']
        bbox = objects[0, 0]['bbox'][0, 0][0] # taken from the NeMo repository
        x0, y0, x1, y1 = bbox

        # handle rgba
        image = Image.open(self.root + im_data['im_path'])
        if len(image.getbands()) != 3:
            image = image.convert('RGB')
        
        #cropped_img = image[:, int(y0):int(y1), int(x0):int(x1)] #second view of image: the at-issue content inside the box
        cropped_img = image.crop((x0, y0, x1, y1)) #second view of image: the at-issue content inside the box
        if 0 in cropped_img.size:
            cropped_img = image

        if self.transforms:
            image = self.transforms(image)
            cropped_img = self.transforms(cropped_img)
        if self.target == 'azimuth':
            data = int(im_data['azimuth'] // 30)
        elif self.target == 'theta':
            data = int(im_data['inplane_rotation']%360 // 30)
        elif self.target == 'elevation':
            data = int((im_data['elevation'] + 90) // 30)
        elif self.target == 'distance':
            data = int(abs(min(100,im_data['distance']) // 20)) # cap distance at 100 as per model assumption
        return image, cropped_img, data

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
        if image.shape[0] == 4:
            image = image[:3, :, :]
        elif image.shape[0] != 3:
            print('n_channels: {}'.format(image.shape[0]))
            image = torch.cat([image, image, image], dim=0)
        #print(image) 
        if self.transforms:
            image = self.transforms(image)
        return image, im_data['source'] + '_' + im_data['cls_name'] + '_' + im_data['im_name'] + '_' + str(im_data['object']), im_data['cls_name']

    def __len__(self):  
        return self.manifest.shape[0]

class Phase2PoseDataset(Dataset):
    '''
    Basic pose estimation dataset. Bucketizes angles into buckets of size pi/6.
    '''
    def __init__(self, *, root, category, transforms=None):
        self.root = root
        self.transforms = transforms
        all_imgs = os.listdir(root)
        self.images = [x for x in all_imgs if category in x]

    def __getitem__(self, idx):
        imname = self.images[idx]
        image = Image.open(self.root + imname)
        if len(image.getbands()) != 3:
            image = image.convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return image, imname 

    def __len__(self):  
        return len(self.images)

# Even more augmentations!
train_transforms = tvt.Compose([
    tvt.ToTensor(),
    tvt.Resize([224, 224]),
    tvt.GaussianBlur(kernel_size=(5, 5), sigma=(4,4)),
    tvt.ColorJitter(brightness=.2, hue=.4),
    tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ])
# train_transforms = tvt.Compose([
#     tvt.ToTensor(),
#     tvt.Resize([224, 224]),
#     tvt.GaussianBlur(kernel_size=(5, 5), sigma=(2.5,2.5)),
#     tvt.ColorJitter(brightness=.1, hue=.3),
#     tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

#     ])

test_transforms = tvt.Compose([
    tvt.ToTensor(),
    tvt.Resize([224, 224]),    
    tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    
if __name__ == "__main__":
    # d1 = PoseBboxCategoryDataset(root='/research/cwloka/projects/dpitt/ROBIN-dataset/ROBINv1.1/train/',\
    #     anno_root='/research/cwloka/projects/dpitt/ROBIN-dataset/ROBINv1.1/annotations/', labels_name='train',\
    #      category='aeroplane', target='azimuth', transforms=train_transforms)
    # print(d1.manifest)
    # print(d1[20])
    # 

    d1 = PoseCategoryDataset(root='/research/cwloka/projects/dpitt/ROBIN-dataset/ROBINv1.1/train/',\
         labels_name='train', category='aeroplane', target='azimuth', transforms=train_transforms)
    img, data = d1[1]
    train_loader = DataLoader(d1, batch_size=1, num_workers=4, persistent_workers=True, shuffle=True, collate_fn=collate)
    #xformed = train_transforms(img)
    #save_image(xformed,fp='/research/cwloka/projects/dpitt/oodcv/second/after.jpg')
    #save_image(img,fp='/research/cwloka/projects/dpitt/oodcv/second/after.jpg')
    for idx, b in enumerate(train_loader):
        print(b)