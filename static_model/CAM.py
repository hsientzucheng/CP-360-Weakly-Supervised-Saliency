from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import time
import sys, os
sys.path.append('..')
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
    img = img.permute(0, 3, 1, 2).contiguous()

    if USE_GPU:
        img = Variable(img).cuda(async=True)
    else:
        img = Variable(img)     

    # forward
    tStart = time.time()
    output = model(img)

    # select most common 5 of 5 from each face #
    top5s = []
    for idx in range(6):
        arr = output[idx].data.cpu().numpy()
        top5s.extend(arr.argsort()[-10:].tolist())
        #print(arr.argsort()[-5:].tolist())
    top5s = np.array(top5s)

    if(CLASS_CONST):
        objness_arr = np.array([class_objectness[x] for x in top5s])
        objs = top5s[objness_arr==1]
        nonobjs = top5s[objness_arr==0]
        objtop5 = [Counter(objs).most_common(5)[x][0] for x in range(len(Counter(objs).most_common(5)))]
        nonobjtop5 = [Counter(nonobjs).most_common(5)[x][0] for x in range(len(Counter(nonobjs).most_common(5)))]

        top5 = np.append(objtop5, nonobjtop5).astype(np.int)
        #top5 = np.array(objtop5).astype(np.int)
    else:
        top5 = np.array([Counter(top5s).most_common(5)[x][0] for x in range(len(Counter(top5s).most_common(5)))]).astype(np.int)
    ############### end of top5 ###############
    # select the most possible object from each face (6) in total #
    #class_indices = torch.max(output, 1)[1].data.cpu().numpy().tolist()
    #print(class_indices)
    ###############################################################


    #cubic_feature = feature_maps[0]
    feature_maps = np.transpose(feature_maps[0], (0, 2, 3, 1))
    grid, face_map = cube2equi(feature_maps.shape[1])
    out_feature = cube2equi_layer(feature_maps, grid, face_map, True)

    tEnd = time.time()
    #print("It takes {0} sec".format(tEnd - tStart))

    # compute CAM
    bz, h, w, nc = out_feature.shape
    out_feature = np.transpose(out_feature, (0, 3, 1, 2))
    features = out_feature.reshape(bz*nc, h*w)

    cams = []
    # test for max response #
    max_rspns = []
    for clss in top5:
        rspn = np.expand_dims(weight_softmax[clss], 0).dot(features)
        max_rspns.append(np.max(rspn))
    cam_weight = [rspn/sum(max_rspns) for rspn in max_rspns]
    ##end of test max rspn##
    
    ## compute weighted CAM ##
    cam = np.zeros((1, h*w))
    for i, (clss) in enumerate(top5):
        #cam += cam_weight[i]*np.expand_dims(weight_softmax[clss], 0).dot(features)
        #cam = np.expand_dims(weight_softmax[clss], 0).dot(features)   # => w/o class weight
        tmp = cam_weight[i]*np.expand_dims(weight_softmax[clss], 0).dot(features)
        cam+=tmp
    ## end of weighted CAM yo ##
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    # cam[cam>(np.mean(cam)+1*np.std(cam))] = 1
    cam_img = np.uint8(255 * cam)
    # result = overlay(input_equi, cam_img, cmap='viridis')
    result = overlay(input_equi, cam_img)
    # cams.append(result)
    handle.remove()
    return cam, feature_maps, weight_softmax
