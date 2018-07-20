from __future__ import print_function

import os
import datetime
import argparse
import math

import cv2
import skvideo.io
import numpy as np
import matplotlib.pyplot as plt

from CAM import CAM
from CAM import overlay
from OPTFL import calcOpticalFlow

from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from CAM import overlay
from equi2cube import equi_to_cube
from resnet_cubic import resnet50
from vgg import vgg16_bn

USE_GPU = True

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def im_norm(in_img, mean, std):
    out_img = in_img
    out_img[:, :, 0] = (in_img[:, :, 0] - mean[0]) / std[0]
    out_img[:, :, 1] = (in_img[:, :, 1] - mean[1]) / std[1]
    out_img[:, :, 2] = (in_img[:, :, 2] - mean[2]) / std[2]
    return out_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid', type=str, default=None, help='Test video path.')
    args, unparsed = parser.parse_known_args()

    #model = resnet50(pretrained=True)
    model = vgg16_bn(pretrained=True)
    if USE_GPU:
        model = model.cuda()

    vid_name = args.vid
    #cap = skvideo.io.vreader(vid_name,inputdict={'-ss': '00:00:01'})
    cap = skvideo.io.vreader(vid_name)
    local_out_path_motion = './'+vid_name.split('/')[-1].split('.mp4')[0]+'_motion'
    local_out_path_data = './'+vid_name.split('/')[-1].split('.mp4')[0]+'_camimg'
    local_out_path = './'+vid_name.split('/')[-1].split('.mp4')[0]
    if not os.path.exists(local_out_path):
        os.makedirs(local_out_path)
    if not os.path.exists(local_out_path_data):
        os.makedirs(local_out_path_data)
    if not os.path.exists(local_out_path_motion):
        os.makedirs(local_out_path_motion)

    cnt=0
    FIRST_FRAME=True
    for frame in cap:

        if FIRST_FRAME: # frame cnt & frame cnt+1
            cur_frame = frame
            FIRST_FRAME=False
            continue

        cnt+=1

        out_im_path = os.path.join(local_out_path,'{:06}.jpg'.format(cnt))

        print("frame {0}, time {1}".format(cnt, datetime.datetime.now()))

        equi_img = Image.fromarray(frame)
        equi_img = equi_img.convert('RGB')
        equi_img = equi_img.resize((1920, 960), resample=Image.LANCZOS)
        input_img = np.array(equi_img) / 255.0

        #################### equi to cube #################### 
        output_cubeset = equi_to_cube(224, input_img)
        init_batch = np.array([])
        for idx in range(6):
            cube_img = im_norm(output_cubeset[idx], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            if idx==0:
                init_batch=np.expand_dims(cube_img, axis=0)
            else:
                init_batch=np.concatenate((init_batch,np.expand_dims(cube_img, axis=0)),axis=0)
        init_batch = init_batch.astype(np.float32)
        ######################################################
        
        # static_sal is a np.array range from 0 to 1
        #static_sal = CAM(init_batch, equi_img, model, 'layer4', 'fc.weight', USE_GPU=USE_GPU)
        static_sal = CAM(init_batch, equi_img, model, 'camconv', 'classifier.weight', USE_GPU=USE_GPU)
        # motion_sal is a np.array range from 0 to 1

        motion_sal, flow = calcOpticalFlow(cur_frame, frame)
        
        static_sal_resized = cv2.resize(static_sal,(motion_sal.shape[1],motion_sal.shape[0]),interpolation=cv2.INTER_CUBIC)

        static_sal_equi = cv2.resize(static_sal,(input_img.shape[1],input_img.shape[0]),interpolation=cv2.INTER_CUBIC)
        
        cam_and_img = np.concatenate((np.expand_dims(static_sal_equi,2),input_img),axis=2)

        static_img = overlay(equi_img, static_sal)
        motion_img = overlay(equi_img, motion_sal)

        #blended_sal = sigmoid(gaussian_filter(motion_sal, sigma=7)+static_sal_resized)
        blended_sal = sigmoid(motion_sal+static_sal_resized)
        
        colorize = plt.get_cmap('jet')
        blended_sal = blended_sal - np.min(blended_sal)
        blended_sal = blended_sal / np.max(blended_sal)

        heatmap = colorize(blended_sal, bytes=True)
        heatmap = Image.fromarray(heatmap[:, :, :3], mode='RGB')
        heatmap_img = overlay(equi_img, heatmap)
        heatmap_img.save(os.path.join(local_out_path,'{0:06}.jpg'.format(cnt)))
        np.save(os.path.join(local_out_path,'{0:06}.npy'.format(cnt)),blended_sal)
        np.save(os.path.join(local_out_path_motion,'{0:06}.npy'.format(cnt)),flow)
        np.save(os.path.join(local_out_path_data,'{0:06}.npy'.format(cnt)),cam_and_img)
        cur_frame = frame

if __name__ == '__main__':
    main()
