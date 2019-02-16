from __future__ import division
from __future__ import print_function

import os, sys
sys.path.append('..')
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
#import ruamel.yaml as yaml

from utils.utils import overlay
from torch.autograd import Variable
from PIL import Image


def CAM(input_cubemap, input_equi, model, feature_layer_name,
        weight_layer_name, use_gpu=True, class_const=False, num_class=1000):
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
    if(class_const):
        aaa = np.load('./imagenet_labeldict.npz')
        class_objectness = aaa['arr_0'].item()
    img = torch.Tensor(input_cubemap)

    model.eval()

    # hook the feature extractor
    feature_maps = []
    def hook(module, input, output):
        if use_gpu:
            feature_maps.append(output.cpu().data.numpy())
        else:
            feature_maps.append(output.data.numpy())
    handle = model._modules.get(feature_layer_name).register_forward_hook(hook)

    params = model.state_dict()[weight_layer_name]
    if use_gpu:
        weight_softmax = np.squeeze(params.cpu().numpy())
    else:
        weight_softmax = np.squeeze(params.numpy())
    if np.min(weight_softmax) < 0:
        weight_softmax -= np.min(weight_softmax)

    # from BZ x H x W x C to BZ x C x H x W
    img = img.permute(0, 3, 1, 2).contiguous()

    if use_gpu:
        img = Variable(img).cuda(async=True)
    else:
        img = Variable(img)

    # forward
    tStart = time.time()
    output = model(img)
    tEnd = time.time()

    cubic_feature = feature_maps[0]

    # compute CAM
    bz, nc, h, w = cubic_feature.shape
    #out_feature = np.transpose(cubic_feature, (0, 3, 1, 2))
    features = cubic_feature.reshape(bz, nc, h*w)

    cams = []

    cube_score = np.array([])
    for idx in range(features.shape[0]):
        if cube_score.shape[0] ==0:
            cube_score = np.expand_dims(weight_softmax.dot(features[idx]),0)
        else:
            cube_score = np.concatenate((cube_score, np.expand_dims(weight_softmax.dot(features[idx]),0)),axis=0)
    cube_score = cube_score.reshape(cube_score.shape[0],num_class,h,w)
    handle.remove()
    return cube_score, cubic_feature, weight_softmax
