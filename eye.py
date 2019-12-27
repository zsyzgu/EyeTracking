from __future__ import division
import os
import cv2
import dlib
import time
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from numpy.lib.stride_tricks import as_strided

class Eye(object):
    def __init__(self, image, eye_points):
        margin = 5
        points = np.array(eye_points)
        x0 = np.min(points[:,0]) - margin
        x1 = np.max(points[:,0]) + margin
        y0 = np.min(points[:,1]) - margin
        y1 = np.max(points[:,1]) + margin
        self.offset = (x0, y0)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        height, width = gray_image.shape[:2]
        black_image = np.zeros((height, width), np.uint8)
        self.mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(self.mask, [points], (0, 0, 0))
        mask_image = cv2.bitwise_not(black_image, gray_image.copy(), mask=self.mask)

        self.eye_image = mask_image[y0 : y1, x0 : x1]
        self.eye_points = eye_points
    
    def image_processing(self, eye_image, threshold):
        kernel = np.ones((3, 3), np.uint8)
        new_image = cv2.bilateralFilter(eye_image, 10, 15, 15)
        new_image = cv2.erode(new_image, kernel, iterations=3)
        new_image = cv2.threshold(new_image, threshold, 255, cv2.THRESH_BINARY)[1]
        return new_image

    def iris_size(self, image):
        image = image[5:-5, 5:-5]
        height, width = image.shape[:2]
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(image)
        return nb_blacks / nb_pixels

    def find_best_threshold(self):
        eye_image = self.eye_image
        average_iris_size = 0.48
        st, en = 5, 100
        while (st < en):
            threshold = (st + en) // 2
            iris_image = self.image_processing(eye_image, threshold)
            iris_size = self.iris_size(iris_image)
            if iris_size >= average_iris_size:
                en = threshold
            else:
                st = threshold + 1
        return st

    def convolve2d(self, img, kernel):
        edge = int(kernel.shape[0]/2)
        img = np.pad(img, [edge,edge], mode='constant', constant_values=0)
        sub_shape = tuple(np.subtract(img.shape, kernel.shape) + 1)
        conv_shape = sub_shape + kernel.shape
        strides = img.strides + img.strides
        submatrices = as_strided(img, conv_shape, strides)
        convolved_mat = np.einsum('ij, klij->kl', kernel, submatrices)
        return convolved_mat

    def get_iris(self, radius):
        threshold = self.find_best_threshold()
        _, image = cv2.threshold(self.eye_image, threshold, 1, cv2.THRESH_BINARY)
        image = 1 - image
        H, W = self.eye_image.shape[:2]

        R = int(radius)
        N = 2 * R + 1
        kernel = np.zeros((N, N), dtype=np.float32)
        for x in range(N):
            for y in range(N):
                if ((x - R) ** 2 + (y - R) ** 2 <= R * R):
                    kernel[y, x] = 1
        
        sum_matrix = self.convolve2d(image, kernel)

        max_sum = 0
        for x in range(W):
            for y in range(H):
                if self.eye_image[y,x] != 255:
                    if sum_matrix[y,x] > max_sum:
                        max_sum = sum_matrix[y,x]
                        iris = (x + self.offset[0], y + self.offset[1])
        
        return iris
    
    def get_eye_center(self):
        p0 = self.eye_points[0]
        p1 = self.eye_points[3]
        return ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
    
    def contains(self, point):
        x = int(point[0])
        y = int(point[1])
        height, width = self.mask.shape[:2]
        if 0 <= x and x < width and 0 <= y and y < height:
            return self.mask[y, x] == 0
        else:
            return False
