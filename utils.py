from __future__ import division
import cv2
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from numpy.lib.stride_tricks import as_strided
from scipy import signal

def dist(A, B):
    return ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2) ** 0.5

def convolve2d(img, kernel):
    edge = int(kernel.shape[0]/2)
    img = np.pad(img, [edge,edge], mode='constant', constant_values=0)
    sub_shape = tuple(np.subtract(img.shape, kernel.shape) + 1)
    conv_shape = sub_shape + kernel.shape
    strides = img.strides + img.strides
    submatrices = as_strided(img, conv_shape, strides)
    convolved_mat = np.einsum('ij, klij->kl', kernel, submatrices)
    return convolved_mat

def gkern(kernlen=5, std=1):
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d
