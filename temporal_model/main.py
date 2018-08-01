from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
sys.path.append('..')
import argparse
import torch
import torchvision
import cv2
import tqdm

from torch.autograd import Variable
from utils.Cube2Equi import Cube2Equi
from clstm import ConvLSTMCell

from eval_saliency import AUC_Borji
from eval_saliency import AUC_Judd
from eval_saliency import CorrCoeff
from eval_saliency import similarity

HIDDEN_SIZE = 1000
INPUT_SIZE = 1000

def test(model, vid_name, seq, indir, output_dir, gt_dir, FRAMES_IN_SUBSEQ, c2e, save_result=True):

    # open video
    cap = cv2.VideoCapture(os.path.join('test', vid_name))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # col
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # row
    # check for the output folder

    if not os.path.exists(os.path.join(output_dir, vid_name)):
        os.mkdir(os.path.join(output_dir, vid_name))
    if not os.path.exists(os.path.join(output_dir, vid_name, 'overlay')):
        os.mkdir(os.path.join(output_dir, vid_name, 'overlay'))
    
    # create evaluation log
    AUC = np.array([])
    AUCB = np.array([])
    CC = np.array([])
    SIM = np.array([])

    # inference by subseq
    for idx, frame_filename in enumerate(tqdm.tqdm(seq)):
        # load frame into subseq
        subseq = []
        if idx>=(len(seq)-FRAMES_IN_SUBSEQ):
            continue
        for subframe_filename in seq[idx:(idx+FRAMES_IN_SUBSEQ)]:
            frame = np.load(os.path.join(indir, vid_name, 'cube_feat', subframe_filename))
            subseq.append(frame)
        # get normalization param
        max_in_subseq = np.max(subseq)
        min_in_subseq = np.min(subseq)
        # reset C0 anf H0 for each subsequence
        init_frame = np.load(os.path.join(indir, vid_name, 'cube_feat', frame_filename))
        init_frame = (init_frame - min_in_subseq) / (max_in_subseq - min_in_subseq)
        cell = Variable(torch.FloatTensor(init_frame)).cuda(async=True)
        hidden = Variable(torch.FloatTensor(init_frame)).cuda(async=True)
        # inference a subsequence
        for frame in subseq:
            frame = (frame - min_in_subseq) / (max_in_subseq - min_in_subseq)
            frame = Variable(torch.FloatTensor(frame)).cuda(async=True)
            # CLSTM forward
            hidden, cell = model(frame, [hidden, cell])
        # output for each subseq
        output = hidden

        equi_output = c2e.to_equi_nn(output)
        equi_output = torch.max(equi_output, 1)[0]
        equi_output = torch.squeeze(equi_output)
        equi_output = equi_output.data.cpu().numpy()
        if save_result == True:
            np.save(os.path.join(output_dir, vid_name, '{:05}.npy'.format(idx+4)), equi_output)
        

        # Evaluation section
        # load ground truth
        fixation_map = np.load(os.path.join(gt_dir, vid_name+'.mp4', '{:05}.npy'.format(idx+4)))
        auc = AUC_Judd(equi_output, fixation_map)
        AUC = np.append(AUC, auc)
        aucb = AUC_Borji(equi_output, fixation_map)
        AUCB = np.append(AUCB, aucb)
        cc = CorrCoeff(equi_output, fixation_map)
        CC = np.append(CC, cc)
        sim = similarity(equi_output, fixation_map)
        SIM = np.append(SIM, sim)

        if (idx + FRAMES_IN_SUBSEQ) == len(seq):
            break
    
    return AUC, CC, SIM, AUCB

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='./CP_VGG_sm0.05t0.7m0.02/_CLSTM_00.pth', required=True)
    parser.add_argument('--dir', type=str, help='input path e.g.../Label_Generation/REC_CP_vgg16', required=True)
    parser.add_argument('--outdir', type=str, default='output', help='output path')
    parser.add_argument('--seql', type=int, default = 5, help='sequence length')
    parser.add_argument('--gt', type=str, help='gt path e.g. /media/raul/6d82a58e-25ab-480a-90eb-1cb88e379370/jimcheng/output25', required=True)

    args, unparsed = parser.parse_known_args()

    # obtain all the video names in test set
    vid_names = open('../utils/test_25.txt', 'r').read().splitlines()

    # construct the cam frame list for each video
    cam_dict = {}
    for vid_name in vid_names:
        seq = []
        for filename in os.listdir(os.path.join(args.dir, vid_name, 'cube_feat')):
            if filename[-4:] == '.npy':
                seq.append(filename)
        seq.sort()
        cam_dict[vid_name] = seq

    # build the model
    model = ConvLSTMCell(INPUT_SIZE, HIDDEN_SIZE)
    model = model.cuda()
    model.load_state_dict(torch.load(args.model))
    model.eval()

    # create AUC log for whole dataset
    AUC = np.array([])
    AUCB = np.array([])
    CC = np.array([])
    SIM = np.array([])
    # inference
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    for idx, vid_name in enumerate(vid_names):
        print("Extracting video {}[{}/{}]".format(vid_name, idx+1, len(vid_names)))
        if idx==0:
            frame = np.load(os.path.join(args.dir, vid_name, 'cube_feat', cam_dict[vid_name][0]))
            c2e = Cube2Equi(int(frame.shape[2]))
        auc, cc, sim, aucb = test(model, vid_name, cam_dict[vid_name], args.dir, args.outdir, args.gt, args.seql, c2e)

        print("[{}]\tAUCB:{}".format(vid_name, np.mean(aucb)))
        print("[{}]\tAUC:{}".format(vid_name, np.mean(auc)))
        print("[{}]\tCC:{}".format(vid_name, np.mean(cc)))
        AUC = np.append(AUC, np.mean(auc))
        AUCB = np.append(AUCB, np.mean(aucb))
        CC = np.append(CC, np.mean(cc))
        SIM = np.append(SIM, np.mean(sim))

    frame_cnt = [len(cam_dict[vid_name]) for vid_name in vid_names]
    wAUC = np.sum([AUC[i]*frame_cnt[i]/np.sum(frame_cnt) for i in range(len(vid_names))])
    wAUCB = np.sum([AUCB[i]*frame_cnt[i]/np.sum(frame_cnt) for i in range(len(vid_names))])
    wCC = np.sum([CC[i]*frame_cnt[i]/np.sum(frame_cnt) for i in range(len(vid_names))])
    wSIM = np.sum([SIM[i]*frame_cnt[i]/np.sum(frame_cnt) for i in range(len(vid_names))])
    print('========== AUC: {}\tCC: {}\wAUCB: {}'.format(wAUC, wCC, wAUCB))
    with open("{}_result.txt".format(args.dir.split('/')[-1]), "w") as text_file:
        print("total result:"+str(wCC)+", "+str(wAUC)+", "+str(wAUCB), file=text_file)

if __name__ == '__main__':
    main()
