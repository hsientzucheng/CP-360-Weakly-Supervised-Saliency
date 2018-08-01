#from __future__ import print_function
import os
import sys

sys.path.append('..')
import datetime
import argparse
import math
import time
import cv2
import skvideo.io
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from CAM1000 import CAM, overlay
from OPTFL import calcOpticalFlow
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from utils.Equi2Cube import Equi2Cube
from utils.Cube2Equi import Cube2Equi
from resnet_cubic import resnet50


USE_GPU = True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def im_norm(in_img, mean, std):
    out_img = in_img
    out_img[:, :, 0] = (in_img[:, :, 0] - mean[0]) / std[0]
    out_img[:, :, 1] = (in_img[:, :, 1] - mean[1]) / std[1]
    out_img[:, :, 2] = (in_img[:, :, 2] - mean[2]) / std[2]
    return out_img

def get_vid_list(in_dir):
    out_list = []
    for item in os.listdir(in_dir):
        out_list.append(item)

    return out_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None, help='The path of test video', required=True)
    parser.add_argument('--out', type=str, default='./FINAL_INTERCUBE_224', help='Test video path.')
    parser.add_argument('--mode', type=str, default='resnet50', help='model')
    args, unparsed = parser.parse_known_args()

    TEST_ONLY = True
    MODE = args.mode
    if 'resnet50' in MODE:
        model = resnet50(pretrained=True)
    if 'vgg16' in MODE:
        model = vgg16_bn(pretrained=True)
    
    args.out = args.out + '_' + MODE

    if USE_GPU:
        model = model.cuda()

    out_list = get_vid_list(args.dir)

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    vid_num = 0

    if TEST_ONLY:
        in_file = open('../utils/test_25.txt', "r")
        ddd_list = in_file.readlines()
        data_list = [x.split('\n')[0]+'.mp4' for x in ddd_list]

    for vid_n in out_list:
        
        if TEST_ONLY:
            if vid_n not in data_list:
                continue
        
        print("Now process {}!".format(vid_n))
        tStart = time.time()
        vid_num+=1
        vid_name = os.path.join(args.dir, vid_n)
        vid_out_dir = os.path.join(args.out, vid_n.split('.mp4')[0])
        cap = skvideo.io.vreader(vid_name)
        local_out_path_feat = os.path.join(vid_out_dir, 'cube_feat')
        local_out_path = os.path.join(vid_out_dir, 'img')


        if not os.path.exists(vid_out_dir):
            os.makedirs(vid_out_dir)
        else:
            print("{} exists.".format(local_out_path))
            continue
            if raw_input("Sure to overwrite {}? [Y/N]".format(local_out_path)) == 'N':
                continue
 
        if not os.path.exists(local_out_path):
            os.makedirs(local_out_path)
        if not os.path.exists(local_out_path_feat):
            os.makedirs(local_out_path_feat)

        cnt=0
        FIRST_FRAME=True
        while(True):
            try:
                frame = cap.next()
            except:
                break
            if FIRST_FRAME: # frame cnt & frame cnt+1
                equi_img = Image.fromarray(frame)
                equi_img = equi_img.convert('RGB')
                equi_img = equi_img.resize((1920, 960), resample=Image.LANCZOS)
                input_img = np.array(equi_img) / 255.0
                e2c = Equi2Cube(224, input_img)
                cur_frame = frame
                C2E_INIT = False
                FIRST_FRAME=False
                continue
            cnt+=1
            equi_img = Image.fromarray(cur_frame)
            equi_img = equi_img.convert('RGB')
            equi_img = equi_img.resize((1920, 960), resample=Image.LANCZOS)
            input_img = np.array(equi_img) / 255.0
            
            #################### equi to cube #################### 
            output_cubeset = e2c.to_cube(input_img)
            init_batch = np.array([])
            for idx in range(6):
                cube_img = im_norm(output_cubeset[idx], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                if idx==0:
                    init_batch = np.expand_dims(cube_img, axis=0)
                else:
                    init_batch = np.concatenate((init_batch, np.expand_dims(cube_img, axis=0)), axis=0)
            init_batch = init_batch.astype(np.float32)
            ######################################################
            
            if MODE=='resnet50':
                cube_1000scores, cube_feat, cube_sm = CAM(init_batch, equi_img, model, 'layer4', 'fc.weight', USE_GPU=USE_GPU)
            if MODE=='vgg16':
                cube_1000scores, cube_feat, cube_sm = CAM(init_batch, equi_img, model, 'camconv', 'classifier.weight', USE_GPU=USE_GPU)
            if not C2E_INIT:
                c2e = Cube2Equi(cube_1000scores.shape[2])
                C2E_INIT = True
            zp_equi = c2e.to_equi_cv2(cube_1000scores)
            zp_sal = np.max(zp_equi, 0)
            zp_sal = zp_sal[:, :]**2
            heatmap_img = overlay(equi_img, zp_sal)
            #motion_sal, flow = calcOpticalFlow(cur_frame, frame)
            heatmap_img.save(os.path.join(vid_out_dir, '{0:06}.jpg'.format(cnt)))
            np.save(os.path.join(local_out_path_feat, '{0:06}.npy'.format(cnt)), cube_1000scores)
            # Output equi image
            #equi_img.save(os.path.join(local_out_path,'{0:06}.jpg'.format(cnt)))
             
            cur_frame = frame

        tEnd = time.time()
        print("It takes {0} sec for {1} frames".format(tEnd - tStart, cnt))

if __name__ == '__main__':
    main()
