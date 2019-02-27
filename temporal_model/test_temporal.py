import os, sys
sys.path.append('..')
import numpy as np
import argparse
import torch
import torchvision
import cv2
import tqdm
import collections
import ruamel_yaml as yaml
from PIL import Image
from utils.utils import overlay
from torch.autograd import Variable
from utils.cube_to_equi import Cube2Equi
from model.clstm import ConvLSTMCell
from utils.eval_saliency import AUC_Borji, AUC_Judd, CorrCoeff, similarity


def test(model, vid_name, seq, indir, output_dir, gt_dir, num_subseq,
        c2e, save_img=False, save_result=True):
    """
        Test temporal model
        Args:
            model: clstm model
            vid_name: input video name
            seq: seq of frames (name)
            indir: input dir (static feature map path)
            output_dir: output dir
            gt_dir: ground truth label path
            num_subseq: clstm sequence length
            c2e: cube to equirectangular function
            save_img: output image or not
            save_result: saving result or not
        Output:
            Evaluation metrics: AUC, CC, SIM, AUCB
            Output image: frame overlay heatmap
    """
    # Open video
    cap = cv2.VideoCapture(os.path.join('test', vid_name))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # col
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # row

    # Check output folders
    if not os.path.exists(os.path.join(output_dir, vid_name)):
        os.mkdir(os.path.join(output_dir, vid_name))
    if not os.path.exists(os.path.join(output_dir, vid_name, 'overlay')):
        os.mkdir(os.path.join(output_dir, vid_name, 'overlay'))

    # Create evaluation log
    AUC = np.array([])
    AUCB = np.array([])
    CC = np.array([])
    SIM = np.array([])

    # Inference subseq
    for idx, frame_filename in enumerate(tqdm.tqdm(seq)):

        # Load frame into subseq
        subseq = []
        if idx >= (len(seq)-num_subseq):
            continue
        for subframe_filename in seq[idx:(idx+num_subseq)]:
            frame = np.load(os.path.join(indir, vid_name, 'cube_feat', subframe_filename))
            subseq.append(frame)
        max_in_subseq = np.max(subseq)
        min_in_subseq = np.min(subseq)

        # Reset C0 anf H0 for each subsequence
        init_frame = np.load(os.path.join(indir, vid_name, 'cube_feat', frame_filename))
        init_frame = (init_frame - min_in_subseq) / (max_in_subseq - min_in_subseq)
        cell = Variable(torch.FloatTensor(init_frame)).cuda(async=True)
        hidden = Variable(torch.FloatTensor(init_frame)).cuda(async=True)

        # Inference a subsequence with CLSTM
        for frame in subseq:
            frame = (frame - min_in_subseq) / (max_in_subseq - min_in_subseq)
            frame = Variable(torch.FloatTensor(frame)).cuda(async=True)
            hidden, cell = model(frame, [hidden, cell])
        output = hidden

        equi_output = c2e.to_equi_nn(output)
        equi_output = torch.max(equi_output, 1)[0]
        equi_output = torch.squeeze(equi_output)
        equi_output = equi_output.data.cpu().numpy()
        if save_result == True:
            np.save(os.path.join(output_dir, vid_name,
                                '{:05}.npy'.format(idx+num_subseq-1)), equi_output)

        # Visualization
        if save_img == True:
            equi_image = Image.open(os.path.join(indir, vid_name, 'img',
                '{:06}.jpg'.format(idx+num_subseq-1)))
            equi_output = equi_output[:, :]**2
            heatmap_img = overlay(equi_image, equi_output)
            heatmap_img.save(os.path.join(output_dir, vid_name, 'overlay',
                                          '{0:06}.jpg'.format(idx+num_subseq-1)))

        # Evaluation
        # Load ground truth
        fixation_map = np.load(os.path.join(gt_dir, vid_name+'.mp4',
                                            '{:05}.npy'.format(idx+num_subseq-1)))
        auc = AUC_Judd(equi_output, fixation_map)
        AUC = np.append(AUC, auc)
        aucb = AUC_Borji(equi_output, fixation_map)
        AUCB = np.append(AUCB, aucb)
        cc = CorrCoeff(equi_output, fixation_map)
        CC = np.append(CC, cc)
        sim = similarity(equi_output, fixation_map)
        SIM = np.append(SIM, sim)

        if (idx + num_subseq) == len(seq):
            break

    return AUC, CC, SIM, AUCB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='your_model.pth', required=True)
    parser.add_argument('--dir', type=str, help='input path', required=True)
    parser.add_argument('--overlay', action='store_true',
                        help='output image or not')
    args, unparsed = parser.parse_known_args()

    # Configirations
    with open('../config.yaml') as f:
        config = yaml.safe_load(f)
    for key in config.keys():
        print("\t{} : {}".format(key, config[key]))
    cfg = collections.namedtuple('GenericDict', config.keys())(**config)

    # Obtain all the video names in test set
    vid_names = open('../data/test_25.txt', 'r').read().splitlines()

    # Construct the cam frame list for each video
    cam_dict = {}
    for vid_name in vid_names:
        seq = []
        for filename in os.listdir(os.path.join(args.dir, vid_name, 'cube_feat')):
            if filename[-4:] == '.npy':
                seq.append(filename)
        seq.sort()
        cam_dict[vid_name] = seq

    # Build the model
    model = ConvLSTMCell(cfg.input_size, cfg.hidden_size)
    model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(cfg.checkpoint_path, args.model)))
    model.eval()

    # Create AUC log for whole dataset
    AUC = np.array([])
    AUCB = np.array([])
    CC = np.array([])
    SIM = np.array([])

    # Inference
    outdir = os.path.join(cfg.output_path, 'temporal')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for idx, vid_name in enumerate(vid_names):
        print("Extracting video {}[{}/{}]".format(vid_name, idx+1, len(vid_names)))
        if idx==0:
            frame = np.load(os.path.join(args.dir, vid_name, 'cube_feat', cam_dict[vid_name][0]))
            c2e = Cube2Equi(int(frame.shape[2]))

        # Test temporal model
        auc, cc, sim, aucb = test(model, vid_name, cam_dict[vid_name],
                args.dir, outdir, cfg.label_path, cfg.seq_len, c2e, args.overlay)

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
