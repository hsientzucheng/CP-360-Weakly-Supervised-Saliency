import numpy as np
import cv2 
import math as m
import torch
from torch import nn
from torch.autograd import Variable
from sph_utils import xy2angle, pruned_inf, to_3dsphere, get_face
from sph_utils import face_to_cube_coord, norm_to_cube

class Cube2Equi:
    def __init__(self, input_w):
        scale_c = 1
        in_width = input_w * scale_c
        out_w = in_width * 4
        out_h = in_width * 2
        out_arr = np.zeros((out_h,out_w, 3), dtype='float32')

        face_map = np.zeros((out_h, out_w)) # for face indexing

        XX, YY = np.meshgrid(range(out_w), range(out_h)) # for output grid

        theta, phi = xy2angle(XX, YY, out_w, out_h)
        theta = pruned_inf(theta)
        phi = pruned_inf(phi)

        _x,_y,_z = to_3dsphere(theta, phi, 1)
        face_map = get_face(_x,_y,_z, face_map)
        x_o, y_o = face_to_cube_coord(face_map, _x, _y, _z)

        out_coord = np.transpose(np.array([x_o, y_o]), (1, 2, 0))  # h x w x 2
        out_coord = norm_to_cube(out_coord, in_width)

        self.out_coord = out_coord
        self.face_map = face_map


    def to_equi_nn(self, input_data):
        ''' 
        input_data: 6 * c * w * w 
        gridf: 2w * 4w * 2
        face_map: 2w * 4w

        output: 1 * c * 2w * 4w
        '''
        gridf = self.out_coord 
        face_map = self.face_map
        gridf = Variable(torch.Tensor(gridf).contiguous()).cuda(async=True)
        face_map = Variable(torch.LongTensor(face_map.astype(np.int)).contiguous()).cuda(async=True)

        out_w = int(gridf.size(1))
        out_h = int(gridf.size(0))
        in_width=out_w/4
        depth = input_data.size(1)
        warp_out = Variable(torch.Tensor(np.zeros((1, depth, out_h, out_w),dtype='float32')),requires_grad=True).cuda()

        gridf = (gridf-torch.max(gridf)/2)/(torch.max(gridf)/2)

        for f_idx in range(0,6):
            face_mask = face_map==f_idx
            expanded_face_mask = face_mask.expand(1,input_data.size(1),face_mask.size(0), face_mask.size(1))
            warp_out[expanded_face_mask] = nn.functional.grid_sample(torch.unsqueeze(input_data[f_idx], 0), torch.unsqueeze(gridf, 0))[expanded_face_mask]
        return warp_out

    def to_equi_cv2(self, input_data):
        ''' 
        input_data: 6 * w * w * c
        gridf: 2w * 4w * 2
        face_map: 2w * 4w
        output: 1 * 2w * 4w * c
        '''
        gridf = self.out_coord 
        face_map = self.face_map 
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


