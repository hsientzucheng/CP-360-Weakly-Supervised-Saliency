import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math as m
from pylab import *
import torch
import torchvision
import torch.nn as nn

from torch.autograd import Variable
from scipy.interpolate import RegularGridInterpolator as interp2d
from itertools import product, combinations

FACE_B = 0
FACE_D = 1
FACE_F = 2
FACE_L = 3
FACE_R = 4
FACE_T = 5


def rotx(ang):
    return np.array([[1, 0, 0],
                     [0, np.cos(ang), -np.sin(ang)],
                     [0, np.sin(ang), np.cos(ang)]])


def roty(ang):
    return np.array([[np.cos(ang), 0, np.sin(ang)],
                     [0, 1, 0],
                     [-np.sin(ang), 0, np.cos(ang)]])


def rotz(ang):
    return np.array([[np.cos(ang), -np.sin(ang), 0],
                     [np.sin(ang), np.cos(ang), 0],
                     [0, 0, 1]])


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


def xy2angle(XX, YY, im_w, im_h):
    _XX = 2*(XX+0.5)/float(im_w)-1
    _YY = 1-2*(YY+0.5)/float(im_h)

    theta = _XX*np.pi
    phi = _YY*np.pi/2

    return theta, phi


def to_3dsphere(theta, phi, R):
    x = R*np.cos(phi)*np.cos(theta)
    y = R*np.sin(phi)
    z = R*np.cos(phi)*np.sin(theta)
    return x, y, z


def pruned_inf(angle):
    float_err = 10e-9
    angle[angle == 0.0] = float_err
    angle[angle == np.pi] = np.pi-float_err
    angle[angle == -np.pi] = -np.pi+float_err
    angle[angle == np.pi/2] = np.pi/2-float_err
    angle[angle == -np.pi/2] = -np.pi/2+float_err
    return angle


def over_pi(angle):
    while(angle > np.pi):
        angle -= 2*np.pi
    while(angle < -np.pi):
        angle += 2*np.pi
    return angle


def get_face(x, y, z, face_map):
    eps = 10e-9

    max_arr = np.maximum(np.abs(x), np.abs(y), np.abs(z))

    x_faces = max_arr-np.abs(x) < eps
    y_faces = max_arr-np.abs(y) < eps
    z_faces = max_arr-np.abs(z) < eps

    face_map[(x >= 0) & x_faces] = FACE_F
    face_map[(x <= 0) & x_faces] = FACE_B
    face_map[(y >= 0) & y_faces] = FACE_T
    face_map[(y <= 0) & y_faces] = FACE_D
    face_map[(z >= 0) & z_faces] = FACE_R
    face_map[(z <= 0) & z_faces] = FACE_L

    # Write cubic_equi image
    '''
    face_show = face_map/6.*255 
    face_show = cv2.applyColorMap(face_show.astype(uint8), cv2.COLORMAP_RAINBOW)
    cv2.imshow('a',face_show)
    cv2.imwrite('cube_to_equi.jpg',face_show)
    '''
    return face_map


def face_to_cube_coord(face_gr, x, y, z):

    direct_coord = np.zeros((face_gr.shape[0], face_gr.shape[1], 3))

    direct_coord[face_gr == FACE_F, 0] = z[face_gr == FACE_F]
    direct_coord[face_gr == FACE_F, 1] = y[face_gr == FACE_F]
    direct_coord[face_gr == FACE_F, 2] = x[face_gr == FACE_F]

    direct_coord[face_gr == FACE_B, 0] = -z[face_gr == FACE_B]
    direct_coord[face_gr == FACE_B, 1] = y[face_gr == FACE_B]
    direct_coord[face_gr == FACE_B, 2] = x[face_gr == FACE_B]

    direct_coord[face_gr == FACE_T, 0] = z[face_gr == FACE_T]
    direct_coord[face_gr == FACE_T, 1] = -x[face_gr == FACE_T]
    direct_coord[face_gr == FACE_T, 2] = y[face_gr == FACE_T]

    direct_coord[face_gr == FACE_D, 0] = z[face_gr == FACE_D]
    direct_coord[face_gr == FACE_D, 1] = x[face_gr == FACE_D]
    direct_coord[face_gr == FACE_D, 2] = y[face_gr == FACE_D]

    direct_coord[face_gr == FACE_R, 0] = -x[face_gr == FACE_R]
    direct_coord[face_gr == FACE_R, 1] = y[face_gr == FACE_R]
    direct_coord[face_gr == FACE_R, 2] = z[face_gr == FACE_R]

    direct_coord[face_gr == FACE_L, 0] = x[face_gr == FACE_L]
    direct_coord[face_gr == FACE_L, 1] = y[face_gr == FACE_L]
    direct_coord[face_gr == FACE_L, 2] = z[face_gr == FACE_L]

    # Convert to top-left origin on faces coordinate
    x_oncube = (direct_coord[:, :, 0]/np.abs(direct_coord[:, :, 2])+1)/2
    y_oncube = (-direct_coord[:, :, 1]/np.abs(direct_coord[:, :, 2])+1)/2

    return x_oncube, y_oncube


def norm_to_cube(_out_coord, w):
    out_coord = _out_coord*(w-1)
    out_coord[out_coord < 0.] = 0.
    out_coord[out_coord > (w-1)] = (w-1)
    return out_coord


TF_TRANS = False


def naive_cube2equi_layer(input_data, gridf, face_map, no_interp):
    '''
    input_data: 6 * w * w * c
    gridf: 2w * 4w * 2
    face_map: 2w * 4w

    output: 1 * 2w * 4w * c
    '''
    out_w = gridf.shape[1]
    out_h = gridf.shape[0]
    in_width = out_w/4
    depth = input_data.shape[-1]
    out_coord = gridf
    out_arr = np.zeros((out_h, out_w, depth), dtype='float32')
    if no_interp:
        # Round to nearest coord -> no interpolation
        out_coord_round = np.rint(gridf).astype(int)

        cube2equi_coord = out_coord_round[:, :, 1] * \
            depth*in_width + out_coord_round[:, :, 0]*depth
        cube2equi_coord = np.tile(cube2equi_coord[..., None], [1, 1, depth])

        c_step = range(0, depth)
        cube2equi_coord = np.add(cube2equi_coord, c_step)

        for f_idx in range(0, 6):
            out_arr[face_map == f_idx] = np.take(
                input_data[f_idx].flatten(), cube2equi_coord[face_map == f_idx])

    else:  # With interpolation
        print("interpolation!!")
        fl_oc0 = np.floor(out_coord)
        fl_oc1 = fl_oc0+1
        max_amount = depth*in_width*in_width-1

        x1_f = fl_oc1[:, :, 0].astype('int')
        y1_f = fl_oc1[:, :, 1].astype('int')
        x0_f = fl_oc0[:, :, 0].astype('int')
        y0_f = fl_oc0[:, :, 1].astype('int')
        xx = out_coord[:, :, 0]
        yy = out_coord[:, :, 1]

        c2e_a = y0_f*depth*in_width + x0_f*depth
        c2e_a = np.tile(c2e_a[..., None], [1, 1, depth])
        c_step = range(0, depth)
        c2e_a = np.add(c2e_a, c_step)

        c2e_d = y1_f*depth*in_width + x1_f*depth
        c2e_d = np.tile(c2e_d[..., None], [1, 1, depth])
        c_step = range(0, depth)
        c2e_d = np.add(c2e_d, c_step)

        c2e_b = y0_f*depth*in_width + x1_f*depth
        c2e_b = np.tile(c2e_b[..., None], [1, 1, depth])
        c_step = range(0, depth)
        c2e_b = np.add(c2e_b, c_step)

        c2e_c = y1_f*depth*in_width + x0_f*depth
        c2e_c = np.tile(c2e_c[..., None], [1, 1, depth])
        c_step = range(0, depth)
        c2e_c = np.add(c2e_c, c_step)

        c2e_a[c2e_a > max_amount] = max_amount
        c2e_b[c2e_b > max_amount] = max_amount
        c2e_c[c2e_c > max_amount] = max_amount
        c2e_d[c2e_d > max_amount] = max_amount

        Ia = np.zeros((out_h, out_w, depth), dtype='float32')
        Ib = np.zeros((out_h, out_w, depth), dtype='float32')
        Ic = np.zeros((out_h, out_w, depth), dtype='float32')
        Id = np.zeros((out_h, out_w, depth), dtype='float32')

        for f_idx in range(0, 6):
            Ia[face_map == f_idx] = np.take(
                input_data[f_idx].flatten(), c2e_a[face_map == f_idx])
            Ib[face_map == f_idx] = np.take(
                input_data[f_idx].flatten(), c2e_b[face_map == f_idx])
            Ic[face_map == f_idx] = np.take(
                input_data[f_idx].flatten(), c2e_c[face_map == f_idx])
            Id[face_map == f_idx] = np.take(
                input_data[f_idx].flatten(), c2e_d[face_map == f_idx])

        wa = np.multiply((x1_f-xx), (y1_f-yy))
        wc = np.multiply((x1_f-xx), (yy-y0_f))
        wb = np.multiply((xx-x0_f), (y1_f-yy))
        wd = np.multiply((xx-x0_f), (yy-y0_f))

        wa3 = np.tile(wa[..., None], [1, 1, depth])
        wb3 = np.tile(wb[..., None], [1, 1, depth])
        wc3 = np.tile(wc[..., None], [1, 1, depth])
        wd3 = np.tile(wd[..., None], [1, 1, depth])

        out_arr = np.multiply(Ia, wa3)+np.multiply(Ib, wb3) + \
            np.multiply(Ic, wc3)+np.multiply(Id, wd3)

    out_arr = np.expand_dims(out_arr, axis=0)
    return np.array(out_arr, dtype=np.float32)
