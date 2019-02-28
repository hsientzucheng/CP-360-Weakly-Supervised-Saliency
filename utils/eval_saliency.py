from utils.cube_to_equi import Cube2Equi
from PIL import Image
from __future__ import print_function
from __future__ import division

import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
sys.path.append('..')


def AUC_Borji(saliency_map, fixation_map, Nsplits=100, stepSize=0.1, to_plot=False):
    # saliency_map is the saliency map
    # fixation_map is the human fixation map (binary matrix)
    # jitter = True will add tiny non-zero random constant to all map
    # locations to ensure ROC can be calculated robustly
    # if to_plot = True, displays ROC curve

    score = float('nan')

    if ~np.any(fixation_map):
        print('no fixation_map')
        exit()

    saliency_map = cv2.resize(saliency_map, (240, 120), cv2.INTER_LANCZOS4)
    fixation_map = cv2.resize(fixation_map, (240, 120), cv2.INTER_LANCZOS4)
    assert saliency_map.shape == fixation_map.shape

    # if jitter:
    #    # jitter the saliency map slightly to disrupt ties of same numbers
    #    sshape = saliency_map.shape
    #    saliency_map = saliency_map+np.random.randn(sshape[0], sshape[1])/1e7

    # normalize saliency map
    saliency_map[saliency_map > np.mean(
        saliency_map)+2*np.std(saliency_map)] = 1.0
    saliency_map = (saliency_map-np.min(saliency_map)) / \
        (np.max(saliency_map) - np.min(saliency_map))

    if np.sum(np.isnan(saliency_map)) == np.size(saliency_map):
        print('NaN saliency_map')
        exit()

    S = saliency_map.flatten()
    F = fixation_map.flatten()

    Sth = S[F > np.mean(F)+2*np.std(F)]  # sal map values at fixation locations
    Nfixations = np.size(Sth)
    Npixels = np.size(S)

    rr = np.random.randint(0, high=Npixels, size=(Nfixations, Nsplits))
    randfix = S[rr]

    auc = []

    for ss in range(Nsplits):
        curfix = randfix[:, ss]
        # print(str(Sth.shape)+","+str(curfix.shape))
        try:
            allthreshes = np.arange(0.0, np.max(
                np.append(Sth, curfix)), stepSize)[::-1]
        except:
            print("allthreshes wrong")
        # allthreshes = np.sort(Sth)[::-1]    # descend
        tp = np.zeros(len(allthreshes)+2)
        fp = np.zeros(len(allthreshes)+2)
        tp[0] = 0.0
        tp[-1] = 1.0
        fp[0] = 0.0
        fp[-1] = 1.0

        for i, thresh in enumerate(allthreshes):
            #aboveth = np.sum(S >= thresh)
            tp[i+1] = np.sum(Sth >= thresh)/float(Nfixations)
            fp[i+1] = np.sum(curfix >= thresh)/float(Nfixations)

        auc.append(np.trapz(tp, fp))

        #allthreshes = np.concatenate(([1], allthreshes, [0]))
    score = np.mean(auc)
    if to_plot:
        plt.plot(fp, tp, 'b-')
        plt.title('Area under ROC curve: {}'.format(score))

    return score


def AUC_Judd(saliency_map, fixation_map, jitter=True, to_plot=False):
    # saliency_map is the saliency map
    # fixation_map is the human fixation map (binary matrix)
    # jitter = True will add tiny non-zero random constant to all map
    # locations to ensure ROC can be calculated robustly
    # if to_plot = True, displays ROC curve
    score = float('nan')

    if ~np.any(fixation_map):
        print('no fixation_map')
        exit()

    saliency_map = cv2.resize(saliency_map, (240, 120), cv2.INTER_LANCZOS4)
    fixation_map = cv2.resize(fixation_map, (240, 120), cv2.INTER_LANCZOS4)
    assert saliency_map.shape == fixation_map.shape

    if jitter:
        # jitter the saliency map slightly to disrupt ties of same numbers
        sshape = saliency_map.shape
        saliency_map = saliency_map+np.random.randn(sshape[0], sshape[1])/1e7

    # normalize saliency map
    #saliency_map[saliency_map > np.mean(saliency_map)+2*np.std(saliency_map)] = 1.0
    saliency_map = (saliency_map-np.min(saliency_map)) / \
        (np.max(saliency_map) - np.min(saliency_map))

    if np.sum(np.isnan(saliency_map)) == np.size(saliency_map):
        print('NaN saliency_map')
        exit()

    S = saliency_map
    F = fixation_map

    Sth = S[F > np.mean(F)+2*np.std(F)]  # sal map values at fixation locations
    Nfixations = np.size(Sth)
    Npixels = np.size(S)

    allthreshes = np.sort(Sth)[::-1]    # descend
    tp = np.zeros(Nfixations+2)
    fp = np.zeros(Nfixations+2)
    tp[0] = 0.0
    tp[-1] = 1.0
    fp[0] = 0.0
    fp[-1] = 1.0

    for i, thresh in enumerate(allthreshes):
        aboveth = np.sum(S >= thresh)
        tp[i+1] = i/Nfixations
        fp[i+1] = (aboveth-i)/(Npixels-Nfixations)

    score = np.trapz(tp, fp)
    allthreshes = np.concatenate(([1], allthreshes, [0]))

    if to_plot:
        plt.plot(fp, tp, 'b-')
        plt.title('Area under ROC curve: {}'.format(score))
    return score


def CorrCoeff(map1, map2):

    map1 = cv2.resize(map1, (240, 120), cv2.INTER_LANCZOS4)
    map2 = cv2.resize(map2, (240, 120), cv2.INTER_LANCZOS4)
    assert map1.shape == map2.shape

    map1 = (map1-np.mean(map1))/np.std(map1)
    map2 = (map2-np.mean(map2))/np.std(map2)

    k = np.shape(map1)
    H = k[0]
    W = k[1]
    c = np.zeros((H, W))
    d = np.zeros((H, W))
    e = np.zeros((H, W))

    # Calculating mean values
    AM = np.mean(map1)
    BM = np.mean(map2)
    # Vectorized versions of c,d,e
    c_vect = (map1-AM)*(map2-BM)
    d_vect = (map1-AM)**2
    e_vect = (map2-BM)**2

    # Finally get r using those vectorized versions
    r_out = np.sum(c_vect)/float(np.sqrt(np.sum(d_vect)*np.sum(e_vect)))

    return r_out


def similarity(map1, map2):

    map1 = cv2.resize(map1, (240, 120), cv2.INTER_LANCZOS4)
    map2 = cv2.resize(map2, (240, 120), cv2.INTER_LANCZOS4)
    assert map1.shape == map2.shape

    map1 = (map1-np.min(map1))/(np.max(map1)-np.min(map1))
    map1 = map1/np.sum(map1)
    map2 = (map2-np.min(map2))/(np.max(map2)-np.min(map2))
    map2 = map2/np.sum(map2)
    score = np.sum(np.minimum(map1, map2))
    return score


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Saliency evaluation')
    parser.add_argument('--input', dest='data_dir',
                        default='../cp', help='file path', type=str)
    parser.add_argument('--gt', dest='gt_dir',
                        default='../gt', help='gt path', type=str)
    args = parser.parse_args()

    return args


def get_vid_list(in_dir):
    out_list = []

    for item in os.listdir(in_dir):
        out_list.append(item)

    return out_list


def main():
    args = parse_args()

    fff = open('./test_25.txt', "r")
    ddd_list = fff.readlines()
    data_list = [x.split('\n')[0] for x in ddd_list]

    input_dir = args.data_dir
    gt_dir = args.gt_dir

    input_list = get_vid_list(input_dir)
    gt_list = get_vid_list(gt_dir)
    all_result = []
    tt_num = 0  # total steps
    data_cnt = 0

    tt_cc = 0.0
    tt_sim = 0.0
    tt_auc_judd = 0.0

    for data_item in data_list:

        tStart = time.time()

        cc = 0.0
        sim = 0.0
        auc_judd = 0.0
        vid_tt_num = 0  # video steps
        data_cnt += 1
        print(str(data_cnt)+"/"+str(len(data_list)))

        gt_path = os.path.join(gt_dir, data_item+'.mp4')
        input_path = os.path.join(input_dir, data_item)
        if not os.path.exists(gt_path):
            continue
        if not os.path.exists(input_path):
            continue

        targ_dir = './output/'+input_dir.split('/')[-1]+'/'+data_item
        if not os.path.exists(targ_dir):
            os.makedirs(targ_dir)
        gt_path_list = os.listdir(gt_path)
        gt_path_list.sort()

        for data_npy in gt_path_list:
            if not '.npy' in data_npy:
                continue
            try:
                gt = np.load(os.path.join(gt_path, data_npy))
                zp = np.load(os.path.join(
                    input_path, 'cube_feat/feat_0'+data_npy))
            except:
                print("skip "+data_npy)
                continue

            gt_sal = cv2.resize(gt, (200, 100), 2)
            if zp.shape[2] == zp.shape[3]:
                c2e = Cube2Equi(zp.shape[2])
                zp_equi = c2e.to_equi_cv2(zp)
                zp_sal = np.max(zp_equi, 0)
            else:
                zp_equi = np.squeeze(zp, 0)
                score1000 = np.mean(np.mean(zp_equi, 1), 1)
                camidx = np.where(score1000 == np.max(score1000))[0][0]
                zp_sal = zp_equi[camidx, :, :]
            zp_sal = cv2.resize(zp_sal, (200, 100), 2).astype(np.float)

            colorize = plt.get_cmap('jet')
            heatmap = zp_equi[camidx, :, :] - np.min(zp_equi[camidx, :, :])
            heatmap = heatmap / np.max(heatmap)
            heatmap = colorize(heatmap, bytes=True)
            heatmap = Image.fromarray(heatmap[:, :, :3], mode='RGB')
            heatmap.save(targ_dir + '/' +
                         '{:06}.png'.format(int(data_npy.split('.')[0])))

            #gt_sal = gt_sal-np.min(gt_sal)
            #gt_sal = gt_sal/np.max(gt_sal)
            zp_sal = zp_sal-np.min(zp_sal)
            zp_sal = zp_sal/np.max(zp_sal)
            zp_sal[zp_sal > np.mean(zp_sal)+2*np.std(zp_sal)] = 1
            #gt_sal = (gt_sal-np.mean(gt_sal))/np.std(gt_sal)
            #zp_sal = (zp_sal-np.mean(zp_sal))/np.std(zp_sal)
            cc += CC(gt_sal, zp_sal)
            sim += similarity(gt_sal, zp_sal)
            auc_judd += AUC_Judd(zp_sal, gt_sal)
            tt_num += 1
            vid_tt_num += 1
        tt_cc += cc
        tt_sim += sim
        tt_auc_judd += auc_judd

        print(data_item+": "+str(sim/vid_tt_num)+", " +
              str(cc/vid_tt_num)+", "+str(auc_judd/vid_tt_num))

        tEnd = time.time()
        print("It takes {0} sec, for {1} frames".format(
            tEnd - tStart, vid_tt_num))

        all_result.append([data_item, sim/vid_tt_num, cc /
                           vid_tt_num, auc_judd/vid_tt_num])

    print("total result:"+str(tt_sim/tt_num)+", " +
          str(tt_cc/tt_num)+", "+str(tt_auc_judd/tt_num))
    print("done!")


if __name__ == '__main__':
    main()
