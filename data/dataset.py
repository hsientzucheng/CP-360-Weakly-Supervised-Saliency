import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

class Sal360Dataset:
    def __init__(self, video_dir, motion_dir, input_d_list, seq_len, transform=None):
        self.video_dir = video_dir
        self.motion_dir = motion_dir
        self.data_list = input_d_list
        self.seq_len = seq_len

        ffile = open(input_d_list, "r")
        ddd_list = ffile.readlines()
        self.data_list = [x.split('\n')[0] for x in ddd_list]

        self.data = []
        self.motion = []

        video_categories = os.listdir(video_dir)
        video_categories.sort()

        for video_category in video_categories:
            if video_category not in self.data_list:
                continue
            print("Got {}".format(video_category))
            feat_sequences = os.listdir(os.path.join(self.video_dir, video_category, 'cube_feat'))
            feat_sequences.sort()
            max_len = int(feat_sequences[-1].split('.')[0])
            for seq in feat_sequences:
                if ('.npy' in seq) and int(seq.split('.')[0])<(max_len-seq_len+1):
                    self.data.append(os.path.join(self.video_dir, video_category,'cube_feat', seq))
            motion_sequences = os.listdir(os.path.join(self.motion_dir, video_category, 'motion'))
            motion_sequences.sort()
            for seq in motion_sequences:
                if ('.npy' in seq) and int(seq.split('.')[0])<(max_len-seq_len+1):
                    self.motion.append(os.path.join(self.motion_dir, video_category, 'motion', seq))

        assert len(self.data)==len(self.motion)
        self.transform = transform

    def __getitem__(self, index):
        seq = []
        motion = []
        category = self.data[index].split('/')[-3]
        filename = self.data[index].split('/')[-1]

        for offset in range(self.seq_len):
            # Static model features
            category, mid_filename, filename = self.data[index].split('/')[-3:]
            targ_feat_path = os.path.join(self.data[index].split(mid_filename)[0], mid_filename,
                            '{:06}{}'.format(int(filename.split('.')[0]) + offset, filename[-4:]))
            if os.path.exists(targ_feat_path):
                cam = np.load(targ_feat_path)
                seq.append(torch.Tensor(cam))
            else:
                print("{} doesn't exist.".format(targ_feat_path))

            # Optical flow
            mcategory, mmid_filename, mfilename = self.motion[index].split('/')[-3:]
            targ_motion_path = os.path.join(self.motion[index].split(mmid_filename)[0], mmid_filename,
                            '{:06}{}'.format(int(mfilename.split('.')[0]) + offset, mfilename[-4:]))
            if os.path.exists(targ_motion_path):
                cam = np.load(targ_motion_path)
                motion.append(torch.Tensor(cam))
            else:
                print("{} doesn't exist.".format(targ_motion_path))
        return seq, motion, category, filename

    def __len__(self):
        return len(self.data)



