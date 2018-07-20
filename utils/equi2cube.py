import pdb
import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math as m
import argparse

from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator as interp2d
from scipy.interpolate import interp1d
from pylab import *
from PIL import Image


def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def rotx(ang):
    return np.array([[1,0,0],
            [0, np.cos(ang), -np.sin(ang)],
            [0, np.sin(ang), np.cos(ang)]])

def roty(ang):
    return np.array([[np.cos(ang),0,np.sin(ang)], 
            [0,1,0],
            [-np.sin(ang),0,np.cos(ang)]])

def rotz(ang):
    return np.array([[np.cos(ang), -np.sin(ang), 0], 
            [np.sin(ang), np.cos(ang), 0], 
             [0,0,1]])

def equi_to_cube(output_width,in_image,vfov=90):

    out = {}
    assert in_image.shape[0]*2==in_image.shape[1]  
    cube_out = np.array([])
    views = [[180,0,0], # Back
         [0,-90,0],     # Bottom
         [0,0,0],   #Front
         [-90,0,0], # Left
         [90,0,0],  # Right
         [0,90,0]]  # Top

    vfov = vfov*np.pi/180
    views = np.array(views)*np.pi/180

    output_width = output_width
    output_height = output_width
    input_width = in_image.shape[1]
    input_height = in_image.shape[0]
    
    topLeft = np.array([-m.tan(vfov/2)*(output_width/output_height), -m.tan(vfov/2), 1])

    # Scaling factor for grabbing pixel co-ordinates
    uv = np.array([-2*topLeft[0]/output_width, -2*topLeft[1]/output_height, 0])

    # Equirectangular lookups 
    res_acos = 2*input_width
    res_atan = 2*input_height
    step_acos = np.pi / res_acos
    step_atan = np.pi / res_atan
    lookup_acos = np.append(np.array(-np.cos(np.array(np.arange(0,res_acos))*step_acos)),1.)
    lookup_atan = np.append(np.append(np.tan(step_atan/2-pi/2), np.tan(np.array(np.arange(1,res_atan))*step_atan-pi/2)),np.tan(-step_atan/2+pi/2))
    #lookup_atan = np.array([np.tan(step_atan/2-pi/2), np.tan(np.array(range(res_atan))*step_atan-pi/2), np.tan(-step_atan/2+pi/2)])

    X,Y = np.meshgrid(range(output_width),range(output_height)) # for output grid
    X = X.flatten()
    Y = Y.flatten()

    XImage, YImage = np.meshgrid(range(input_height), range(input_width))

    for idx in range(views.shape[0]):
        yaw = views[idx,0]
        pitch = views[idx,1]
        roll = views[idx,2]
        transform = np.dot(np.dot(roty(yaw),rotx(pitch)),rotz(roll))
        #transform = np.dot(np.dot(rotation_matrix([0,1,0],yaw),rotation_matrix([1,0,0],pitch)),rotation_matrix([0,0,1],roll))

        points = np.concatenate((np.concatenate((topLeft[0] + uv[0]*np.expand_dims(X,axis=0), topLeft[1] + uv[1]*np.expand_dims(Y,axis=0)), axis=0), topLeft[2] + uv[2]*np.ones((1, X.shape[0]))),axis=0 )
        moved_points = np.dot(transform,points)

        x_points = moved_points[0,:]
        y_points = moved_points[1,:]
        z_points = moved_points[2,:]
    
        nxz = sqrt(x_points**2 + z_points**2)
        phi = zeros(X.shape[0])
        theta = zeros(X.shape[0])

        ind = nxz < 10e-10
        phi[ind & (y_points > 0)] = pi/2
        phi[ind & (y_points <= 0)] = -pi/2

        ind =  np.logical_not(ind)
        phi_interp = interp1d(lookup_atan, np.arange(0,res_atan+1),'linear')
        phi[ind] = phi_interp(y_points[ind]/nxz[ind])*step_atan - (pi/2)
        theta_interp = interp1d(lookup_acos, np.arange(0,res_acos+1), 'linear')
        theta[ind] = theta_interp(-z_points[ind]/nxz[ind])*step_acos
        theta[ind & (x_points < 0)] = -theta[ind & (x_points < 0)]
        
        # Find equivalent pixel co-ordinates
        inX = (theta / pi) * (input_width/2) + (input_width/2) + 1
        inY = (phi / (pi/2)) * (input_height/2) + (input_height/2) + 1
        
        # Cap if out of bounds
        inX[inX < 1] = 1
        inX[inX >= input_width-1] = input_width-1 # not equl -> out of range
        inY[inY < 1] = 1
        inY[inY >= input_height-1] = input_height-1
        
        # Initialize output image
        out[idx] = np.zeros((output_height, output_width,in_image.shape[2]), in_image.dtype)

        out_pix = zeros((X.shape[0],in_image.shape[2]))

        inX = inX.reshape(output_width,output_height).astype('float32')
        inY = inY.reshape(output_width,output_height).astype('float32')
        for c in range(in_image.shape[2]):
            out[idx][:,:,c] = cv2.remap(in_image[:,:,c], inX, inY, cv2.INTER_LINEAR)

    return out

def get_vid_list(in_dir):
    out_list = []
    for item in os.listdir(in_dir):
        out_list.append(item)
    return out_list

# output demo generator
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None, help='Test video path.')
    parser.add_argument('--out', type=str, default='./Saliency_Visualization', help='Test video path.')
    args, unparsed = parser.parse_known_args()

    out_list = get_vid_list(args.dir)
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    output_path = os.path.join(args.out,args.dir.split('/')[-1])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    TEST_ONLY = True
    if TEST_ONLY:
        fff = open('./test_25.txt', "r")
        ddd_list = fff.readlines()
        #data_list = [x.split('\n')[0]+'.mp4' for x in ddd_list]
        data_list = [x.split('\n')[0] for x in ddd_list]
    vid_num=0
    for item in data_list:
        vid_num+=1
        print(item+' '+str(vid_num)+'/'+str(len(data_list)))
        if 'CLSTM' in args.dir:
            vid_path = os.path.join(args.dir,item,'overlay')
        elif 'dataset' in args.dir:
            vid_path = os.path.join(args.dir,item+'.mp4','overlay')
        else:
            vid_path = os.path.join(args.dir,item)
        frame_list = os.listdir(vid_path)
        local_out_path = os.path.join(output_path, item)
        if not os.path.exists(local_out_path):
            os.makedirs(local_out_path)
        for frame in frame_list:
            try:
                cnt = int(frame.split('.jpg')[0])
            except:
                continue
            frame_path = os.path.join(vid_path, frame)
            equi_img = Image.open(frame_path)
            equi_img = equi_img.convert('RGB')
            equi_img = equi_img.resize((1920,960),resample=Image.LANCZOS)
            output_cubeset = equi_to_cube(480, np.array(equi_img))
    
            out_td_equi = np.zeros((np.array(equi_img).shape[0],np.array(equi_img).shape[1]+480,np.array(equi_img).shape[2]),dtype=np.uint)
            out_td_equi[:480,1920:,:]=output_cubeset[5]
            out_td_equi[480:,1920:,:]=output_cubeset[1]
            out_td_equi[:,:1920,:]=np.array(equi_img)
            #for faceid in output_cubeset.keys():
            #    cc_img = Image.fromarray(output_cubeset[faceid])
            #    cc_img = cc_img.convert('RGB')
                #cc_img.save(os.path.join(local_out_cube_path,'{0:06}_{1}.jpg'.format(cnt,faceid)))
            '''equi_img'''
            out_td_equi = out_td_equi.astype(np.int)
            out_td_equi_img = Image.fromarray(np.uint8(out_td_equi))
            out_td_equi_img = out_td_equi_img.resize((1440, 480), resample=Image.LANCZOS)
            
            out_td_equi_img.save(os.path.join(local_out_path,'{0:06}.jpg'.format(cnt)))
            #equi_img.save(os.path.join(local_out_path,'{0:06}.jpg'.format(cnt)))
            print(os.path.join(local_out_path,'{0:06}.jpg'.format(cnt)))
        

if __name__ == "__main__":
    main()
