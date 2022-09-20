import os
import shutil

root = '/research/cwloka/projects/dpitt/ROBIN-dataset/RobinNPZ/'
train_dir = 'train/'
val_dir = 'phase-1/iid_test/'
test_dir = 'phase-1/nuisances/'
categories = ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train']
nuisances = os.listdir(root + test_dir)

### train directory
# for category in categories:
#     if os.path.exists(root + train_dir + 'processed/images/' + category):
#         shutil.rmtree(root + train_dir + 'processed/images/' + category)
#     if os.path.exists(root + train_dir + 'processed/annotations/' + category):
#         shutil.rmtree(root + train_dir + 'processed/annotations/' + category)
#     os.mkdir(root + train_dir + 'processed/images/' + category)

#     os.mkdir(root + train_dir + 'processed/annotations/' + category)

#     with open(root + train_dir + category + '.txt') as f:
#         for line in f:
#             im_name = line.strip()
#             # train_img_path = root+train_dir + 'processed/images/'+ im_name + '.JPEG'
#             # train_anno_path = root+train_dir + 'processed/annotations/'+ im_name + '.npz'

#             # organized_train_img_path = root+train_dir + 'processed/images/' + category + '/' + im_name + '.JPEG'
#             # organized_train_anno_path = root+train_dir + 'processed/annotations/' + category + '/' + im_name + '.npz'

#             # if os.path.isfile(train_img_path):
#             #     shutil.copyfile(train_img_path, organized_train_img_path)
#             #     shutil.copyfile(train_anno_path, organized_train_anno_path)

#             train_img_path = root+train_dir + 'processed/images/'+ im_name + '.JPEG'
#             train_anno_path = root+train_dir + 'processed/annotations/'+ im_name + '.npz'

#             organized_train_img_path = root+train_dir + 'processed/images/' + category + '/' + im_name + '.JPEG'
#             organized_train_anno_path = root+train_dir + 'processed/annotations/' + category + '/' + im_name + '.npz'

#             if os.path.isfile(train_img_path):
#                 shutil.copyfile(train_img_path, organized_train_img_path)
#                 shutil.copyfile(train_anno_path, organized_train_anno_path)
#             elif 

### validation directory
# for category in categories:
#     if os.path.exists(root + val_dir + 'images/' + category):
#         shutil.rmtree(root + val_dir + 'images/' + category)
#     if os.path.exists(root + val_dir + 'annotations/' + category):
#         shutil.rmtree(root + val_dir + 'annotations/' + category)
#     os.mkdir(root + val_dir + 'images/' + category)
#     os.mkdir(root + val_dir + 'annotations/' + category)

#     for file in os.listdir(root + val_dir + 'images/'):
#         if category in file and file[-1] == 'G':
#             shutil.copyfile(root + val_dir + 'images/' + file, root + val_dir + 'images/' + category + '/' + file)

#     for file in os.listdir(root + val_dir + 'annotations/'):
#         if category in file and file[-1] == 'z':
#             shutil.copyfile(root + val_dir + 'annotations/' + file, root + val_dir + 'annotations/' + category + '/' +  file)

for nuisance in nuisances:
    for category in categories:
        if os.path.exists(root + test_dir + nuisance + '/images/' + category):
            shutil.rmtree(root + test_dir + nuisance + '/images/' + category)
        if os.path.exists(root + test_dir + nuisance + '/annotations/' + category):
            shutil.rmtree(root + test_dir + nuisance + '/annotations/' + category)
        os.mkdir(root + test_dir + nuisance + '/images/' + category)
        os.mkdir(root + test_dir + nuisance + '/annotations/' + category)

        for file in os.listdir(root + test_dir + nuisance + '/images/'):
            if category in file and file[-1] == 'G':
                shutil.copyfile(root + test_dir + nuisance + '/images/' + file, root + test_dir + nuisance + '/images/' + category + '/' + file)

        for file in os.listdir(root + test_dir + nuisance + '/annotations/'):
            if category in file and file[-1] == 'z':
                shutil.copyfile(root + test_dir + nuisance + '/annotations/' + file, root + test_dir + nuisance + '/annotations/' + category + '/' +  file)