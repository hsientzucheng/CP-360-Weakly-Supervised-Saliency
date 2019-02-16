import os, sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

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

def im_norm(in_img, mean, std):
    out_img = in_img
    out_img[:, :, 0] = (in_img[:, :, 0] - mean[0]) / std[0]
    out_img[:, :, 1] = (in_img[:, :, 1] - mean[1]) / std[1]
    out_img[:, :, 2] = (in_img[:, :, 2] - mean[2]) / std[2]
    return out_img

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
