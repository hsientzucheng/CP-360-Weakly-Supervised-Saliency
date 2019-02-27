import os, sys
sys.path.append('..')
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import collections
import ruamel_yaml as yaml

from utils.cube_to_equi import Cube2Equi
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model.clstm import ConvLSTMCell
from PIL import Image
from utils.utils import cam_visual
from data.dataset import Sal360Dataset


def generate_meshgrid(flow):
        h = flow.size(1)
        w = flow.size(2)
        y = torch.arange(0, h).unsqueeze(1).repeat(1, w) / (h - 1) * 2 - 1
        x = torch.arange(0, w).unsqueeze(0).repeat(h, 1) / (w - 1) * 2 - 1
        mesh_grid = Variable(torch.stack([x,y], 0).unsqueeze(0).repeat(flow.size(0), 1, 1, 1).cuda(async=True))
        return mesh_grid.float()

def train(train_loader, model, criterion, optimizer, epoch,
                    out_model_path, init_iter, cfg, tmp_loss_len=3):
    """
        Train temporal model
        Args:
            train_loader: Sal360Dataset with loader
            model: clstm model
            criterion: loss functions
            optimizer: optimizer for training
            epoch: epoch number
            out_model_path: path to save checkpoints
            init_iter: start from previous inter (when load model)
            cfg: training configuration (from config.yaml and argprase)
            tmp_loss_len: calculate losses from how many frames
        Output:
            Save checkpoint 'CLSTM_[epoch]_[iteration].pth'
    """
    seq_len = cfg.seq_len
    flow_h = cfg.flow_h
    l_s = cfg.l_s
    l_t = cfg.l_t
    l_m = cfg.l_m
    assert cfg.use_gpu

    model.train()
    batch_time = 0.0
    running_loss = 0.0
    for i, (seq, flow, _, _) in enumerate(train_loader):

        ttime = time.time()
        i += init_iter

        # Real batch size (how many cube)
        num_input_cube = seq[0].shape[0]

        if i > len(train_loader) + 1:
            break

        c2e = Cube2Equi(seq[0].size(-1))

        # Intra sequence normalize
        seq_items = np.array([])

        for seq_item in seq:
            if seq_items.shape[0]==0:
                seq_items = seq_item.numpy()
            else:
                seq_items = np.concatenate((seq_items, seq_item.numpy()), 0)
        min_seq = np.min(seq_items)
        max0min_seq = np.max(seq_items - min_seq)

        prev_hidden_buff = []
        flow_buff = []
        # Input sequence
        for idx, cube_cam in enumerate(seq):

            cube_cam = torch.Tensor(cube_cam.numpy() - min_seq)
            cube_cam = torch.Tensor(cube_cam.numpy() / max0min_seq)
            cube_cam = cube_cam.view(cube_cam.shape[0]*cube_cam.shape[1],
                    cube_cam.shape[2], cube_cam.shape[3], cube_cam.shape[4])
            if idx==0:
                hidden = Variable(cube_cam).cuda(async=True)
                cell = Variable(cube_cam).cuda(async=True)

            var_cube_cam = Variable(cube_cam).cuda(async=True)

            # CLSTM forward
            hidden, cell = model(var_cube_cam, [hidden, cell])

            # Equirectangular cam (from static)
            for i_b in range(num_input_cube):
                if idx >= (cfg.seq_len - tmp_loss_len - 1):
                    hidden_equi = c2e.to_equi_nn(hidden[6*i_b:6*(i_b+1),:])
                    aaaa, bbbb = torch.max(hidden_equi, 1)
                    prev_hidden_buff.append(torch.unsqueeze(aaaa, 0))

                    # For flow resizing
                    fscale = flow_h / float(flow[idx].size(2))
                    np_flow = fscale * np.array(cv2.resize(flow[idx].numpy()[i_b],
                                    (flow_h*2, flow_h), interpolation=cv2.INTER_CUBIC))
                    flow_buff.append(torch.unsqueeze(torch.Tensor(np_flow), 0))

        # Out put last cam if needed
        # out_cam = prev_hidden_buff[-1]

        # Loss calculation
        mesh_grid = generate_meshgrid(flow_buff[-1]).permute(0, 2, 3, 1)
        for i_b in range(num_input_cube): # Iter for batch size
            for fidx in range(tmp_loss_len):

                targ_idx = i_b + fidx * num_input_cube
                tmp_flow = flow_buff[targ_idx] # Flow -> targ_idx to (targ_idx+1)

                # Get flow magnitude and motion mask
                abs_flow = torch.sqrt(tmp_flow[0,:,:,0]**2 + tmp_flow[0,:,:,1]**2)
                motion_mask = Variable(torch.sqrt(tmp_flow[0, :, :, 0]**2 + tmp_flow[0, :, :, 1]**2) < cfg.mm_th).cuda()

                # Calculate flow warping grid
                curr_buff_step = targ_idx
                tmp_feat = nn.functional.upsample(prev_hidden_buff[curr_buff_step],
                                size=(tmp_flow.size(1), tmp_flow.size(2)), mode='bilinear')
                tmp_feat_next = nn.functional.upsample(prev_hidden_buff[curr_buff_step + num_input_cube],
                                size=(tmp_flow.size(1), tmp_flow.size(2)), mode='bilinear')
                tmp_flow[:, :, :, 0] = (tmp_flow[:, :, :, 0]) / tmp_feat.size()[3] * 2
                tmp_flow[:, :, :, 1] = (tmp_flow[:, :, :, 1]) / tmp_feat.size()[2] * 2
                tmp_grid = Variable(tmp_flow).cuda() + mesh_grid
                tmp_feat_val = tmp_feat
                tmp_feat_val_next = tmp_feat_next

                # Calculate prediction warping with optical flow grid
                warp_prediction = nn.functional.grid_sample(tmp_feat_val, tmp_grid)

                warp_prediction = warp_prediction.detach()
                tmp_feat_val = tmp_feat_val.detach()
                tmp_feat_val_mask = tmp_feat_val_next.clone()

                # Mask out saliency prediction by motion masking
                tmp_feat_val_mask[:, :, motion_mask]=0
                tmp_feat_val_mask = tmp_feat_val_mask.detach()

                # Loss aggregation through time steps
                if targ_idx == 0:
                    loss_sm = criterion(tmp_feat_val_next, warp_prediction)
                    loss_temp = criterion(tmp_feat_val_next, tmp_feat_val)
                    loss_mask = criterion(tmp_feat_val_next, tmp_feat_val_mask)
                else:
                    loss_sm += criterion(tmp_feat_val_next, warp_prediction)
                    loss_temp += criterion(tmp_feat_val_next, tmp_feat_val)
                    loss_mask += criterion(tmp_feat_val_next, tmp_feat_val_mask)

        if i % cfg.summary_freq == (cfg.summary_freq - 1):
            print("Smooth loss: {0:.3f}, Tmp loss: {1:.3f}, MMask loss: {2:.3f}".format(cfg.l_s*loss_sm.data.item(),
                                                                                        cfg.l_t*loss_temp.data.item(),
                                                                                        cfg.l_m*loss_mask.data.item()))
        loss = cfg.l_s*loss_sm + cfg.l_t*loss_temp + cfg.l_m*loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time += time.time() - ttime
        running_loss += loss.data.item()

        if i % cfg.summary_freq == (cfg.summary_freq-1):
            print("Epoch: [{}][{}/{}]\t Loss (avg.): {:.3f}\t Batch Time (avg.):{:.3f}".format(epoch,
                        i+1, len(train_loader),
                        running_loss/cfg.summary_freq,
                        batch_time/cfg.summary_freq))
            batch_time = 0.0
            running_loss = 0.0

        if i % cfg.save_freq == (cfg.save_freq-1):
            print(os.path.join(out_model_path, 'CLSTM_{0:02}_{1:06}.pth'.format(epoch, i)))
            torch.save(model.state_dict(), os.path.join(out_model_path,
                        'CLSTM_{0:02}_{1:06}.pth'.format(epoch, i)))

        # Depatch variables
        loss.detach()
        loss_sm.detach()
        loss_mask.detach()
        loss_temp.detach()
        motion_mask.detach()
        tmp_grid.detach()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sml', type=float, help='smooth weight')
    parser.add_argument('--mml', type=float, help='motion mask weight')
    parser.add_argument('--tmpl', type=float, help='temporal weight')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--input', type=str, help='input path')
    parser.add_argument('--motion', type=str, help='motion path')

    args, unparsed = parser.parse_known_args()

    # Configurations
    with open('../config.yaml') as f:
        config = yaml.safe_load(f)
    for key in config.keys():
        print("\t{} : {}".format(key, config[key]))
        cfg = collections.namedtuple('GenericDict', config.keys())(**config)

    if args.sml is not None:
        cfg.l_s = args.sml
    if args.tmpl is not None:
        cfg.l_t = args.tmpl
    if args.mml is not None:
        cfg.l_m = args.mml
    if args.lr is not None:
        cfg.lr = args.lr

    dataset_root = args.input
    motion_root = args.input

    out_model_path = os.path.join(cfg.checkpoint_path,
                    "CLSTM_s_{0:04}_t_{1:04}_m_{2:04}".format(cfg.l_s, cfg.l_t, cfg.l_m))
    if not os.path.exists(out_model_path):
        os.makedirs(out_model_path)

    feat_name = args.input.split('/')[-1]
    train_dataset = Sal360Dataset(dataset_root, motion_root, '../data/train_60.txt', cfg.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                    shuffle=True, num_workers=cfg.processes)

    # test_dataset = Sal360Dateset(dataset_root, motion_root,'../data/test_25.txt')
    # test_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
    #               shuffle=False, num_workers=cfg.processes)

    model = ConvLSTMCell(cfg.input_size, cfg.hidden_size)

    if cfg.use_gpu:
        model = model.cuda()

    model_num = []
    init_model_num=0
    if model_num != []:
        model.load_state_dict(torch.load(os.path.join(out_model_path,
                        'CLSTM_{0:02}_{1:06}.pth'.format(0, max(model_num)))))
        init_model_num = max(model_num)

    if cfg.use_gpu:
        criterion = nn.MSELoss(size_average=False).cuda()
    else:
        criterion =nn.MSELoss(size_average=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        train(train_loader, model, criterion, optimizer, epoch, out_model_path, init_model_num, cfg)
        torch.save(model.state_dict(), os.path.join(out_model_path,
                        feat_name + '_CLSTM_{:02}.pth'.format(epoch)))

        # Load pretrained model if needed
        # model.load_pretrained_model_seq(torch.load(feat_name+'_CLSTM_00.pth'))

        # Current test function got no data loader (see test_temporal.py)
        # test(test_loader, model)

if __name__ == '__main__':
    main()
