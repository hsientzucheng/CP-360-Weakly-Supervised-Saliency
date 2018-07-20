from __future__ import print_function
import os
import pdb
import datetime
import argparse
import math
import time
import cv2
import skvideo.io
import numpy as np
import matplotlib.pyplot as plt

from equi2cube import equi_to_cube
from PIL import Image
from sph_utils import cube2equi


def overlay(img, heatmap, cmap='jet', alpha=0.5):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    if isinstance(heatmap, np.ndarray):
        colorize = plt.get_cmap(cmap)
        # Normalize
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / np.max(heatmap)
        heatmap = colorize(heatmap, bytes=True)
        heatmap = Image.fromarray(heatmap[:, :, :3], mode='RGB')

    # Resize the heatmap to cover whole img
    heatmap = heatmap.resize((img.size[0], img.size[1]), resample=Image.CUBIC)
    # Display final overlayed output
    result = Image.blend(img, heatmap, alpha)
    return result


def cube2equi_layer_cv2(input_data, gridf, face_map):
    '''
    input_data: 6 * w * w * c
    gridf: 2w * 4w * 2
    face_map: 2w * 4w

    output: 1 * 2w * 4w * c
    '''
    out_w = gridf.shape[1]
    out_h = gridf.shape[0]
    in_width=out_w/4   
    depth = input_data.shape[1]

    gridf = gridf.astype(np.float32)
    out_arr = np.zeros((out_h, out_w, depth),dtype='float32')
    
    input_data = np.transpose(input_data, (0, 2, 3, 1))

    for f_idx in range(0,6):
        for dept in range(1000/4):
            out_arr[face_map==f_idx, 4*dept:4*(dept+1)] = cv2.remap(input_data[f_idx,:,:,4*dept:4*(dept+1)], gridf[:,:,0], gridf[:,:,1],cv2.INTER_CUBIC)[face_map==f_idx]
    return np.transpose(out_arr, (2,0,1)) 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid', type=str, default=None, help='Test video path.')
    parser.add_argument('--feat', type=str, default='', help='Test video path.')
    #parser.add_argument('--out', type=str, default='./REC_INTERCUBE_224', help='Test video path.')
    args, unparsed = parser.parse_known_args()

    args.out = 'FINAL_'+args.feat
    TEST_ONLY = True
        
    in_file = open('./test_25.txt', "r")
    ddd_list = in_file.readlines()
    data_list = [x.split('\n')[0]+'.mp4' for x in ddd_list]
    vid_num=0
    for vid_n in data_list:
        tStart = time.time()
        vid_num+=1
        print(vid_n+' '+str(vid_num)+'/'+str(len(data_list)))
        vid_name = os.path.join(args.vid, vid_n)
        vid_out_dir = os.path.join(args.out, vid_n.split('.mp4')[0])
        #cap = skvideo.io.vreader(vid_name,inputdict={'-ss': '00:00:01'})
        cap = skvideo.io.vreader(vid_name)
        local_out_path = os.path.join(vid_out_dir,'img')

        if not os.path.exists(vid_out_dir):
            os.makedirs(vid_out_dir)

        cnt=0

        vid_feat_path = os.path.join(args.feat,vid_n.split('.mp4')[0],'cube_feat')
        FIRST_FRAME=True
        while(True):
        #for fidd, frame in enumerate(cap):
            try:
                frame = cap.next()
            except:
                break
            if FIRST_FRAME: # frame cnt & frame cnt+1
                cur_frame = frame
                FIRST_FRAME=False
                continue

            cnt+=1

            equi_img = Image.fromarray(cur_frame)
            equi_img = equi_img.convert('RGB')
            equi_img = equi_img.resize((1920, 960), resample=Image.LANCZOS)
            input_img = np.array(equi_img) / 255.0
            
            if ('equi' in vid_feat_path) or ('EQUI' in vid_feat_path):
                local_feat_name = os.path.join(vid_feat_path,'{0:06}.npy'.format(cnt))
            else: 
                local_feat_name = os.path.join(vid_feat_path,'feat_{0:06}.npy'.format(cnt))

            try:
                cube_1000scores = np.load(local_feat_name)
            except:
                print('skip ',os.path.join(vid_feat_path,'feat_{0:06}.npy'.format(cnt)))
                continue

            if ('equi' in vid_feat_path) or ('EQUI' in vid_feat_path):
                zp_sal = np.max(cube_1000scores[0],0)
                zp_sal = zp_sal[:,:]**2
            else:
                grid, face_map = cube2equi(cube_1000scores.shape[2])
                #grid, face_map = cube2equi(80)
                #pdb.set_trace()
                zp_equi = cube2equi_layer_cv2(cube_1000scores, grid, face_map)
                zp_sal = np.max(zp_equi,0)
                zp_sal = zp_sal[:,:]**2
            heatmap_img = overlay(equi_img, zp_sal)
            #plt.imshow(heatmap_img)
            #plt.show()
            #pdb.set_trace()
            '''motion'''
            #motion_sal, flow = calcOpticalFlow(cur_frame, frame)
            
            heatmap_img.save(os.path.join(vid_out_dir,'{0:06}.jpg'.format(cnt)))
            '''
            #output_cubeset = equi_to_cube(224, np.array(equi_img))
            for faceid in output_cubeset.keys():
                cc_img = Image.fromarray(output_cubeset[faceid])
                cc_img = cc_img.convert('RGB')
                cc_img.save(os.path.join(local_out_cube_path,'{0:06}_{1}.jpg'.format(cnt,faceid)))
            '''
            '''equi_img'''
            cur_frame = frame




        tEnd = time.time()
        print("It takes {0} sec for {1} frames".format(tEnd - tStart, cnt))

if __name__ == '__main__':
    main()


