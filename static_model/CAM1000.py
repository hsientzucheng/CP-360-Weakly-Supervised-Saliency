from __future__ import division
from __future__ import print_function
import os, sys
sys.path.append('..')

import pdb
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from utils.sph_utils import cube2equi
from torchvision import transforms
from torch.autograd import Variable

from PIL import Image

from collections import Counter

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


def CAM(input_cubemap, input_equi, model, feature_layer_name, weight_layer_name, USE_GPU=False, CLASS_CONST=False):
    """Compute Class Activation Map (CAM) of input_img

    Args:
        input_cubemap (np.array): cubic image feed into model, shape = 6 x H x W x C
        input_equi (np.array): equirectangular image feed into model, shape = 1 x H x W x C
        model (torchvision.models): the DCNN model
        feature_layer_name (string): should be the name of model's last conv layer
        weight_layer_name (string): should be the name of model's classifier weight layer
        USE_GPU (bool): GPU configuration

    Returns:
        The return image (PIL.Image.Image). With input_equi blended with CAM.

    """
    if(CLASS_CONST):
        aaa = np.load('./imagenet_labeldict.npz')
        class_objectness = aaa['arr_0'].item()
    img = torch.Tensor(input_cubemap)

    model.eval()

    # hook the feature extractor
    feature_maps = []
    def hook(module, input, output):
        if USE_GPU:
            feature_maps.append(output.cpu().data.numpy())
        else:
            feature_maps.append(output.data.numpy())
    handle = model._modules.get(feature_layer_name).register_forward_hook(hook)
    
    params = model.state_dict()[weight_layer_name]
    if USE_GPU:
        weight_softmax = np.squeeze(params.cpu().numpy())
    else:
        weight_softmax = np.squeeze(params.numpy())
    if np.min(weight_softmax) < 0:
        weight_softmax-=np.min(weight_softmax)

    # from BZ x H x W x C to BZ x C x H x W
    # pdb.set_trace()
    img = img.permute(0, 3, 1, 2).contiguous()

    if USE_GPU:
        img = Variable(img).cuda(async=True)
    else:
        img = Variable(img)     

    # forward
    tStart = time.time()
    output = model(img)
    tEnd = time.time()
    #print("It takes {0} sec".format(tEnd - tStart))

    cubic_feature = feature_maps[0]


    # compute CAM
    bz, nc, h, w = cubic_feature.shape
    #out_feature = np.transpose(cubic_feature, (0, 3, 1, 2))
    features = cubic_feature.reshape(bz, nc, h*w)


    cams = []
    
    '''
    # compute un-weighted CAM by class #
    for class_idx, _ in top5:
        cam = np.expand_dims(weight_softmax[class_idx], 0).dot(features)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam_img = np.uint8(255 * cam)
        result = overlay(input_equi, cam_img)
        cams.append(result)
    ####################################
    '''
    cube_score = np.array([])
    for idx in range(features.shape[0]):
        if cube_score.shape[0] ==0:
            cube_score = np.expand_dims(weight_softmax.dot(features[idx]),0)
        else:
            cube_score = np.concatenate((cube_score, np.expand_dims(weight_softmax.dot(features[idx]),0)),axis=0)
    cube_score = cube_score.reshape(cube_score.shape[0],1000,h,w)
    handle.remove()
    return cube_score, cubic_feature, weight_softmax
