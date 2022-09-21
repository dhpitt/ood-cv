import os
import shutil

import argparse
global args
parser = argparse.ArgumentParser(description='Generate list of filenames for dir indexed by category in ROBIN')


parser.add_argument('--root_path', default='/research/cwloka/projects/dpitt/ROBIN-dataset/RobinNPZ/phase-1/iid_test', type=str, help='')
args = parser.parse_args()
cates = ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train']

for cate in cates:
    f = open(args.root_path + cate + '.txt', 'w+')
    for fname in os.listdir(args.root_path + 'images/'):
        if cate in fname:
            f.write(fname.split('.')[0] + '\n') # everything up to extension
    f.close()


