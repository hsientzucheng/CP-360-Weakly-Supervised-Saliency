import os
import sys
import datetime
import argparse
import math
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import collections
import ruamel_yaml as yaml

from class_activation_model import CAM
from utils.optical_flow import calcOpticalFlow
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from utils.equi_to_cube import Equi2Cube
from utils.cube_to_equi import Cube2Equi
from utils.utils import im_norm
from model.resnet_cubic import resnet50
sys.path.append('..')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='static',
                        help='The path of output dir')
    parser.add_argument('--mode', type=str, default='resnet50', help='model')
    parser.add_argument('-oi', '--output_img', action='store_true',
                        help='output images or not')
    parser.add_argument('-of', '--output_feature', action='store_true',
                        help='output features or not')
    parser.add_argument('-om', '--output_motion', action='store_true',
                        help='output optical flow or not')

    args, unparsed = parser.parse_known_args()

    # Configurations
    with open('../config.yaml') as f:
        config = yaml.safe_load(f)
    for key in config.keys():
        print("\t{} : {}".format(key, config[key]))
    cfg = collections.namedtuple('GenericDict', config.keys())(**config)
    mode = args.mode
    if 'resnet50' in mode:
        model = resnet50(pretrained=True)

    # Currently support ResNet-50 only
    # if 'vgg16' in mode:
    #     model = vgg16_bn(pretrained=True)

    out_path = os.path.join(cfg.output_path, args.out + '_' + mode)

    if cfg.use_gpu:
        model = model.cuda()

    def get_vid_list(in_dir):
        out_list = []
        for item in os.listdir(in_dir):
            out_list.append(item)
        return out_list

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    vid_num = 0

    # Testing set
    if cfg.test_mode:
        vid_path = os.path.join(cfg.data_vid_path, 'test')
        out_list = get_vid_list(vid_path)
        in_file = open('../data/test_25.txt', "r")
        d_list = in_file.readlines()
        data_list = [x.split('\n')[0]+'.mp4' for x in d_list]

    # Training set
    if cfg.train_mode:
        vid_path = os.path.join(cfg.data_vid_path, 'train')
        out_list = get_vid_list(vid_path)
        in_file = open('../data/train_60.txt', "r")
        d_list = in_file.readlines()
        data_list_train = [x.split('\n')[0]+'.mp4' for x in d_list]

    for vid_n in out_list:
        if cfg.test_mode:
            if vid_n not in data_list:
                continue
        if cfg.train_mode:
            if vid_n not in data_list_train:
                continue

        # Process an input video with cv2
        print("Now process {}!".format(vid_n))
        tStart = time.time()
        vid_num += 1
        vid_name = os.path.join(vid_path, vid_n)
        vid_out_dir = os.path.join(out_path, vid_n.split('.mp4')[0])
        cap = cv2.VideoCapture(vid_name)

        # Create files
        local_out_path_feat = os.path.join(vid_out_dir, 'cube_feat')
        local_out_path_motion = os.path.join(vid_out_dir, 'motion')
        local_out_path = os.path.join(vid_out_dir, 'img')

        if not os.path.exists(vid_out_dir):
            os.makedirs(vid_out_dir)
        if not os.path.exists(local_out_path):
            os.makedirs(local_out_path)
        if not os.path.exists(local_out_path_feat):
            os.makedirs(local_out_path_feat)
        if not os.path.exists(local_out_path_motion):
            os.makedirs(local_out_path_motion)

        cnt = 0
        success = True

        # Iterate to process each frame
        for cnt in tqdm.tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            try:
                success, frame = cap.read()
            except:
                break

            # For frame cnt & frame cnt+1
            if cnt == 0:
                equi_img = Image.fromarray(frame)
                equi_img = equi_img.convert('RGB')
                equi_img = equi_img.resize(
                    (cfg.equi_h, cfg.equi_w), resample=Image.LANCZOS)
                input_img = np.array(equi_img) / 255.0
                e2c = Equi2Cube(cfg.cube_dim, input_img)
                cur_frame = frame
                c2e_init_flag = False
                continue

            cnt += 1  # Count from 1
            equi_img = Image.fromarray(cur_frame)
            equi_img = equi_img.convert('RGB')
            equi_img = equi_img.resize(
                (cfg.equi_h, cfg.equi_w), resample=Image.LANCZOS)
            input_img = np.array(equi_img) / 255.0

            # Equirectangular to cube
            output_cubeset = e2c.to_cube(input_img)

            # Process batch
            init_batch = np.array([])
            for idx in range(6):
                cube_img = im_norm(output_cubeset[idx], [
                                   0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                if idx == 0:
                    init_batch = np.expand_dims(cube_img, axis=0)
                else:
                    init_batch = np.concatenate(
                        (init_batch, np.expand_dims(cube_img, axis=0)), axis=0)
            init_batch = init_batch.astype(np.float32)

            # Class activation
            if mode == 'resnet50':
                cube_1000scores, cube_feat, cube_sm = CAM(init_batch, equi_img, model,
                                                          'layer4', 'fc.weight', use_gpu=cfg.use_gpu)

            # Currently support ResNet-50 only
            # if mode=='vgg16':
            #     cube_1000scores, cube_feat, cube_sm = CAM(init_batch, equi_img, model,
            #                                               'camconv', 'classifier.weight')

            if not c2e_init_flag:
                c2e = Cube2Equi(cube_1000scores.shape[2])
                c2e_init_flag = True

            # Generate heatmap
            _equi = c2e.to_equi_nn(cube_1000scores)
            _sal = np.max(_equi, 0)
            _sal = _sal[:, :]**2
            heatmap_img = overlay(equi_img, _sal)
            if cfg.opt_flow:
                motion_sal, flow = calcOpticalFlow(cur_frame, frame)

            # Output equi image
            if args.output_img:
                heatmap_img.save(os.path.join(
                    vid_out_dir, '{0:06}.jpg'.format(cnt)))
                equi_img.save(os.path.join(
                    local_out_path, '{0:06}.jpg'.format(cnt)))
            if args.output_feature:
                np.save(os.path.join(local_out_path_feat,
                                     '{0:06}.npy'.format(cnt)), cube_1000scores)
            if args.output_motion:
                np.save(os.path.join(local_out_path_motion,
                                     '{0:06}.npy'.format(cnt)), flow)
            cur_frame = frame

        tEnd = time.time()
        print("It takes {0} sec for {1} frames".format(tEnd - tStart, cnt))


if __name__ == '__main__':
    main()
