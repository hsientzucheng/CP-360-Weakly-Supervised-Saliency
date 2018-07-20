import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calcOpticalFlow(prev_frame, cur_frame):

    prev_frame = cv2.resize(prev_frame[..., ::-1], (960, 480), interpolation=cv2.INTER_LANCZOS4)
    cur_frame = cv2.resize(cur_frame[..., ::-1], (960, 480), interpolation=cv2.INTER_LANCZOS4)
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    DF = cv2.optflow.createOptFlow_DeepFlow()
    h,w = prev_frame.shape
    flow_temp = np.zeros((h,w,2))
    flow = DF.calc(prev_frame, cur_frame, flow_temp)

    #flow = cv2.calcOpticalFlowFarneback(prev_frame, cur_frame, None, 0.5, 7, 15, 3, 5, 1.2, 0)
    absflow = np.sqrt(flow[:, :, 0]**2+flow[:, :, 1]**2)
    absflow = absflow - np.min(absflow)
    absflow = absflow / np.max(absflow)
    absflow[absflow < (np.mean(absflow)-1.5*np.std(absflow))] = 0

    return absflow, flow
