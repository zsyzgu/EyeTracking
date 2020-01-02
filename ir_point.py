from __future__ import division
import os
import cv2
import dlib
import time
import numpy as np
from eye import Eye
from scipy.optimize import minimize
import random
import utils

class IrPoint(object):
    def __init__(self):
        pass
    
    def __call__(self, image, iris, radius):
        IR_THRESHOLD = 1.4
        FOLD = 5.0
        X, Y = int(iris[0]), int(iris[1])
        R = int(radius * 0.5)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eye_image = gray_image[Y - R: Y + R + 1, X - R: X + R + 1].astype(np.float32)
        if (np.max(eye_image) / np.mean(eye_image) <= IR_THRESHOLD):
            return None

        H, W = eye_image.shape[:2]
        H = int(H * FOLD)
        W = int(W * FOLD)
        eye_image = cv2.resize(eye_image, (H, W))

        kernel = utils.gkern(3,1)
        kernel = kernel - 0.1
        k_size = kernel.shape[0]
        kernel = cv2.resize(kernel, (int(k_size * FOLD), int(k_size * FOLD)))

        matrix = utils.convolve2d(eye_image, kernel)
        max_sum = 0
        ir_point = None
        for x in range(W):
            for y in range(H):
                if matrix[y,x] > max_sum:
                    max_sum = matrix[y,x]
                    ir_point = (x / FOLD + X - R, y / FOLD + Y - R)

        return ir_point
