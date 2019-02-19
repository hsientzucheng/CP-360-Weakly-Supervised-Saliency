import os
import csv
import pdb
import time
import numpy as np
import matplotlib.pyplot as plt
import tables
import cv2
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from sph_utils import cube2equi, cube2equi_layer
from torch.utils.data import DataLoader
from torch.autograd import Variable
from clstm_cubic import ConvLSTMCell
from PIL import Image

USE_GPU = True

SEQ_LEN = 5
HIDDEN_SIZE = 1000
INPUT_SIZE = 1000


TEMPORAL_LOSS_LEN = 3
BATCH_SIZE = 1
EPOCH = 1
MSG_DISPLAY_FREQ = 20
SNAPSHOT_FREQ = 5000

def cam_visual(input_equi, cam):
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam_img = np.uint8(255 * cam)
    result = overlay(input_equi, cam_img)
    return result

class Sal360Dateset:
    def __init__(self, video_dir, label_dir, train_test_split ,transform=None):
        self.video_dir = video_dir
        self.label_dir = label_dir
        self.data_list = train_test_split
        fff = open(train_test_split, "r")
        ddd_list = fff.readlines()
        self.data_list = [x.split('\n')[0] for x in ddd_list]

        self.data = []
        self.motion = []

        video_categories = os.listdir(video_dir)
        video_categories.sort()

        for video_category in video_categories:
            if video_category not in self.data_list:
                continue
            feat_sequences = os.listdir(self.video_dir + '/' + video_category+'/cube_feat')
            feat_sequences.sort()
            for seq in feat_sequences:

                if ('.npy' in seq) and ('feat' in seq):
                    self.data.append(self.video_dir + '/' + video_category + '/cube_feat/' + seq)
                #if ('.npy' in seq) and ('motion' in seq):
                #    self.motion.append(self.video_dir + '/' + video_category + '/' + seq)
            motion_sequences = os.listdir(self.label_dir + '/' + video_category)
            motion_sequences.sort()
            for seq in motion_sequences:

                if ('.npy' in seq) and ('feat' in seq):
                    self.motion.append(self.label_dir + '/' + video_category + '/' + seq)

        self.transform = transform

    def __getitem__(self, index):
        seq = []
        motion = []
        category = self.data[index].split('/')[-3]
        filename = self.data[index].split('/')[-1]

        for offset in range(SEQ_LEN):
            category, _, filename = self.data[index].split('/')[-3:]
            if os.path.exists(self.data[index][:-15]+'feat_{:06}{}'.format(int(filename[-10:-4]) + offset, filename[-4:])):
                filename = self.data[index][:-15]+'feat_{:06}{}'.format(int(filename[-10:-4]) + offset, filename[-4:])
                cam = np.load(filename)
                seq.append(torch.Tensor(cam))
            else:
                filename = self.data[index][:-15]+'feat_{:06}{}'.format(int(filename[-10:-4]), filename[-4:])
                cam = np.load(filename)
                seq.append(torch.Tensor(cam))
            category, filename = self.motion[index].split('/')[-2:]
            if os.path.exists(self.motion[index][:-15]+'feat_{:06}{}'.format(int(filename[-10:-4]) + offset, filename[-4:])):
                filename = self.motion[index][:-15]+'feat_{:06}{}'.format(int(filename[-10:-4]) + offset, filename[-4:])
                cam = np.load(filename)
                motion.append(torch.Tensor(cam))
            else:
                filename = self.motion[index][:-15]+'feat_{:06}{}'.format(int(filename[-10:-4]), filename[-4:])
                cam = np.load(filename)
                motion.append(torch.Tensor(cam))
        return seq, motion, category, filename

    def __len__(self):
        return len(self.data)

def generate_meshgrid(flow):
        h = flow.size(1)
        w = flow.size(2)
        y = torch.arange(0, h).unsqueeze(1).repeat(1, w) / (h - 1) * 2 - 1
        x = torch.arange(0, w).unsqueeze(0).repeat(h, 1) / (w - 1) * 2 - 1
        mesh_grid = Variable(torch.stack([x,y], 0).unsqueeze(0).repeat(flow.size(0), 1, 1, 1).cuda(async=True))
        return mesh_grid

def train(train_loader, model, criterion, optimizer, epoch, out_model_path, SM_WEIGHT,TEMP_WEIGHT,MASK_WEIGHT, init_iter):

    loss_saver = []

    TARG_FLOW_H = 480 #480 960
    model.train()
    avg_pool = nn.AvgPool2d(14)
    batch_time = 0.0
    running_loss = 0.0
    for i, (seq, flow, cc, ff) in enumerate(train_loader):
        pdb.set_trace()
        i+=init_iter
        if i>len(train_loader)+1:
            break
        t = time.time()

        #======================= intra sequence normalize ==================
        seq_items = np.array([])
        for seq_item in seq:
            if seq_items.shape[0]==0:
                seq_items = seq_item.numpy()
            else:
                seq_items = np.concatenate((seq_items,seq_item.numpy()),0)
        min_seq = np.min(seq_items)
        max0min_seq = np.max(seq_items-min_seq)

        #===================================================================

        prev_hidden_buff = []

        #======================== input sequence ===========================
        for idx, cube_cam in enumerate(seq):
            cube_cam = torch.Tensor(cube_cam.numpy()-min_seq)
            cube_cam = torch.Tensor(cube_cam.numpy()/max0min_seq)
            if idx==0:
                hidden = Variable(cube_cam[0]).cuda(async=True)
                cell = Variable(cube_cam[0]).cuda(async=True)

            if USE_GPU:
                cube_cam = Variable(cube_cam).cuda(async=True)
            else:
                cube_cam = Variable(cube_cam)
            cube_cam = cube_cam[0]
            # CLSTM forward
            hidden, cell = model(cube_cam, [hidden, cell])

            # prepare cube to equi grids
            grid, face_map = cube2equi(cube_cam.size(2))
            face_map = face_map.astype(int)
            if USE_GPU:
                grid = Variable(torch.Tensor(grid)).cuda(async=True)
                face_map = Variable(torch.LongTensor(face_map)).cuda(async=True)
            else:
                grid = Variable(torch.Tensor(grid))
                face_map = Variable(torch.LongTensor(face_map))
            # equirectangular cam (from static)
            hidden_equi = nn_cube2equi_layer(hidden, grid, face_map)
            if idx>=(SEQ_LEN-TEMPORAL_LOSS_LEN-1):
                aaaa, bbbb = torch.max(hidden_equi,1)
                prev_hidden_buff.append(torch.unsqueeze(aaaa,0))

        out_cam = prev_hidden_buff[-1]

        fscale = TARG_FLOW_H/float(flow[idx].size(1))
        npflow = fscale * np.array([cv2.resize(flow[idx].numpy()[0], (TARG_FLOW_H*2,TARG_FLOW_H), interpolation=cv2.INTER_CUBIC) for idx in range(len(flow))])
        flow = [torch.unsqueeze(torch.Tensor(npflow[idx]),0) for idx in range(len(flow))]

        # ======================================= Loss calculation = ====================================================
        mesh_grid = generate_meshgrid(flow[-1]).permute(0,2,3,1)
        for fidx in range(TEMPORAL_LOSS_LEN):

            targ_idx = SEQ_LEN-fidx-2 # 3, 2, 1, 0
            tmp_flow = flow[targ_idx] # flow targ_idx to targ_idx+1

            # get flow magnitude and motion mask
            abs_flow = torch.sqrt(tmp_flow[0,:,:,0]**2+tmp_flow[0,:,:,1]**2) 
            motion_mask = Variable(torch.sqrt(tmp_flow[0,:,:,0]**2+tmp_flow[0,:,:,1]**2) < 0.1).cuda()
            # calculate flow warping grid
            curr_buff_step = targ_idx-SEQ_LEN+len(prev_hidden_buff)
            tmp_feat = nn.functional.upsample(prev_hidden_buff[curr_buff_step], size=(tmp_flow.size(1), tmp_flow.size(2)), mode='bilinear')
            tmp_feat_next = nn.functional.upsample(prev_hidden_buff[curr_buff_step+1], size=(tmp_flow.size(1), tmp_flow.size(2)), mode='bilinear')
            tmp_flow[:, :, :, 0] = (tmp_flow[:, :, :, 0]) / tmp_feat.size()[3] * 2
            tmp_flow[:, :, :, 1] = (tmp_flow[:, :, :, 1]) / tmp_feat.size()[2] * 2
            tmp_grid = Variable(tmp_flow).cuda()+ mesh_grid
            tmp_feat_val = tmp_feat
            tmp_feat_val_next = tmp_feat_next

            # calculate prediction warping with optical flow grid
            warp_prediction = nn.functional.grid_sample(tmp_feat_val, tmp_grid)

            warp_prediction = warp_prediction.detach()
            tmp_feat_val = tmp_feat_val.detach()
            tmp_feat_val_mask = tmp_feat_val_next.clone()

            # masking out saliency prediction by motion masking
            tmp_feat_val_mask[motion_mask]=0
            tmp_feat_val_mask = tmp_feat_val_mask.detach()

            # loss aggregation through time steps
            if fidx == 0:
                loss_sm = criterion(tmp_feat_val_next, warp_prediction)
                loss_temp = criterion(tmp_feat_val_next, tmp_feat_val)
                loss_mask = criterion(tmp_feat_val_next, tmp_feat_val_mask)
            else:
                loss_sm += criterion(tmp_feat_val_next, warp_prediction)
                loss_temp += criterion(tmp_feat_val_next, tmp_feat_val)
                loss_mask += criterion(tmp_feat_val_next, tmp_feat_val_mask)
        if i % MSG_DISPLAY_FREQ == (MSG_DISPLAY_FREQ-1):
            print("smooth loss: {0:.3f}, temporal loss: {1:.3f}, motion mask loss: {2:.3f}".format(SM_WEIGHT*loss_sm.data.cpu().numpy()[0],
                                                    TEMP_WEIGHT*loss_temp.data.cpu().numpy()[0],MASK_WEIGHT*loss_mask.data.cpu().numpy()[0]))

        loss = SM_WEIGHT*loss_sm + TEMP_WEIGHT*loss_temp + MASK_WEIGHT*loss_mask
        loss_saver.append([loss_sm.data.cpu().numpy()[0],loss_temp.data.cpu().numpy()[0],loss_mask.data.cpu().numpy()[0],loss.data.cpu().numpy()[0]])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time += time.time() - t
        running_loss += loss.data[0]

        if i % MSG_DISPLAY_FREQ == (MSG_DISPLAY_FREQ-1):

            print("Epoch: [{}][{}/{}]\t Loss (avg.): {}\t Batch Time (avg.): {:.3f}".format(epoch, i+1, len(train_loader), running_loss/MSG_DISPLAY_FREQ, batch_time/MSG_DISPLAY_FREQ))
            batch_time = 0.0
            running_loss = 0.0
        if i % SNAPSHOT_FREQ == (SNAPSHOT_FREQ-1):
            torch.save(model.state_dict(), out_model_path+'/CLSTM_{0:02}_{1:06}.pth'.format(epoch,i))

        loss.detach()
        loss_sm.detach()
        loss_mask.detach()
        loss_temp.detach()
        grid.detach()
        face_map.detach()
        motion_mask.detach()
        tmp_grid.detach()

    np.save(out_model_path+'/loss_{0:02}_{1:06}.npy'.format(epoch,i),np.array(loss_saver))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sml', type=float, default=0.1, help='smooth weight')
    parser.add_argument('--mml', type=float, default=0.0, help='motion mask weight')
    parser.add_argument('--tmpl', type=float, default=1.0, help='temporal weight')
    parser.add_argument('--lr', type=float, default=1e-6, help='lr')
    parser.add_argument('--input', type=str, help='input path e.g./home/jimcheng/Desktop/CAM_VGG_224_vgg16')
    parser.add_argument('--motion', type=str, help='motion path')

    args, unparsed = parser.parse_known_args()

    dataset_root = args.input
    motion_root = args.motion

    SM_WEIGHT = args.sml
    TEMP_WEIGHT = args.tmpl
    MASK_WEIGHT = args.mml

    out_model_path = "./"+dataset_root.split('/')[-1]+"./CLSTM_sm{0:04}t{1:04}m{2:04}".format(SM_WEIGHT,TEMP_WEIGHT,MASK_WEIGHT)
    if not os.path.exists(out_model_path):
        os.makedirs(out_model_path)

    feat_name = args.input.split('/')[-1]
    train_dataset = Sal360Dateset(dataset_root, motion_root, './train_60.txt')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    #test_dataset = Sal360Dateset(dataset_root, motion_root,'./test.txt')
    #test_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print("=> using CLSTM")
    model = ConvLSTMCell(INPUT_SIZE, HIDDEN_SIZE)
    init_model_num = 0

    if USE_GPU:
        model = model.cuda()

    model_num = []
    init_model_num=0
    if model_num != []:
        model.load_state_dict(torch.load(os.path.join(out_model_path,'CLSTM_{0:02}_{1:06}.pth'.format(0,max(model_num)))))
        init_model_num = max(model_num)


    if USE_GPU:
        criterion = nn.MSELoss(size_average=False).cuda()
    else:
        criterion =nn.MSELoss(size_average=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(EPOCH):
        train(train_loader, model, criterion, optimizer, epoch, out_model_path, SM_WEIGHT,TEMP_WEIGHT,MASK_WEIGHT, init_model_num)
        torch.save(model.state_dict(), out_model_path+'/'+feat_name+'_CLSTM_{:02}.pth'.format(epoch))
        #model.load_pretrained_model_seq(torch.load(feat_name+'_CLSTM_00.pth'))
        #test(test_loader, model)

if __name__ == '__main__':
    main()
