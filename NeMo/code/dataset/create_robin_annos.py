import numpy as np
import BboxTools as bbt
import scipy.io as sio
import os
from PIL import Image
import pickle
import cv2
import math
from kp_list import kp_list_dict, mesh_len
from kp_list import top_50_size_dict
import argparse
from tqdm import tqdm

global args

name_list += img_name + '.JPEG\n'

                anno_path = os.path.join(anno_dir, '{}.mat'.format(img_name))
                mat_contents = sio.loadmat(anno_path)
                record = mat_contents['record']

                mesh_idx = get_anno(record, 'cad_index')

                objects = record['objects']
                azimuth_coarse = objects[0, 0]['viewpoint'][0, 0]['azimuth_coarse'][0, 0][0, 0]
                elevation_coarse = objects[0, 0]['viewpoint'][0, 0]['elevation_coarse'][0, 0][0, 0]
                distance = objects[0, 0]['viewpoint'][0, 0]['distance'][0, 0][0, 0]
                bbox = objects[0, 0]['bbox'][0, 0][0]

                box = bbt.from_numpy(bbox, sorts=('x0', 'y0', 'x1', 'y1'))

                resize_rate = float(200 * get_anno(record, 'distance') / 1000)
                if resize_rate <= 0:
                    resize_rate = 224 / min(box.shape)

                box_ori = box.copy()

                box *= resize_rate

                img = np.array(Image.open(os.path.join(load_image_path, img_name + '.JPEG')))
                box_ori = box_ori.set_boundary(img.shape[0:2])

                img = cv2.resize(img, dsize=(int(img.shape[1] * resize_rate), int(img.shape[0] * resize_rate)))

                center = (get_anno(record, 'principal')[::-1] * resize_rate).astype(int)

                box1 = bbt.box_by_shape(out_shape, center)

                if out_shape[0] // 2 - center[0] > 0 or out_shape[1] // 2 - center[1] > 0 or out_shape[0] // 2 + center[0] - img.shape[0] > 0 or out_shape[1] // 2 + center[1] - img.shape[1] > 0:
                    if len(img.shape) == 2:
                        padding = ((max(out_shape[0] // 2 - center[0], 0), max(out_shape[0] // 2 + center[0] - img.shape[0], 0)),
                                   (max(out_shape[1] // 2 - center[1], 0), max(out_shape[1] // 2 + center[1] - img.shape[1], 0)))
                    else:
                        padding = ((max(out_shape[0] // 2 - center[0], 0), max(out_shape[0] // 2 + center[0] - img.shape[0], 0)),
                                   (max(out_shape[1] // 2 - center[1], 0), max(out_shape[1] // 2 + center[1] - img.shape[1], 0)),
                                   (0, 0))

                    img = np.pad(img, padding, mode='constant')
                    box = box.shift([padding[0][0], padding[1][0]])
                    box1 = box1.shift([padding[0][0], padding[1][0]])

                box_in_cropped = box.copy()
                box = box1.set_boundary(img.shape[0:2])
                box_in_cropped = box.box_in_box(box_in_cropped)

                img_cropped = box.apply(img)

                proj_foo = bbt.projection_function_by_boxes(box_ori, box_in_cropped, compose=False)

                cropped_kp_list = []
                states_list = []
                for kp in kp_list:
                    states = objects[0, 0]['anchors'][0, 0][kp][0, 0]['status'][0, 0][0, 0]
                    if states == 1:
                        kp_x, kp_y = objects[0, 0]['anchors'][0, 0][kp][0, 0]['location'][0, 0][0]
                        if len(args.data_pendix) > 0 and kp_x < occ_mask.shape[1] and kp_y < occ_mask.shape[0] and occ_mask[int(kp_y), int(kp_x)]:
                            states = 0
                        cropped_kp_x = proj_foo[1](kp_x)
                        cropped_kp_y = proj_foo[0](kp_y)
                    else:
                        cropped_kp_x = cropped_kp_y = 0
                    states_list.append(states)
                    cropped_kp_list.append([cropped_kp_y, cropped_kp_x])

                save_parameters = dict(name=img_name, box=box.numpy(), box_ori=box_ori.numpy(), box_obj=box_in_cropped.numpy(), cropped_kp_list=cropped_kp_list, visible=states_list, occ_mask=occ_mask)

                save_parameters = {**save_parameters, **{k: v for k, v in zip(mesh_para_names, get_anno(record, *mesh_para_names))}}

                np.savez(os.path.join(save_annotation_path, img_name), **save_parameters)