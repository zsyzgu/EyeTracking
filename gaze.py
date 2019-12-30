from __future__ import division
import os
import cv2
import dlib
import time
import numpy as np
from eye import Eye
from scipy.optimize import minimize
import random

class Gaze(object):
    def __init__(self):
        self.iris_left = None
        self.iris_right = None
        self.radius_left = 0
        self.radius_right = 0
        self.landmarks = None
        self.grad_x = None
        self.grad_y = None
        self.eye_left = None
        self.eye_right = None
        self.edge_left = None
        self.edge_right = None
        
        self._face_detector = dlib.get_frontal_face_detector()
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    def dist(self, A, B):
        return ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2) ** 0.5

    def refresh(self, image):
        faces = self._face_detector(image)

        if len(faces) >= 1:
            self.landmarks = self._predictor(image, faces[0])
            self.eye_left = Eye(image, self.get_eye_left())
            self.eye_right = Eye(image, self.get_eye_right())
            eye_center_left = self.eye_left.get_eye_center()
            eye_center_right = self.eye_right.get_eye_center()
            self.radius_left = self.radius_right = ((eye_center_left[0] - eye_center_right[0]) ** 2 + (eye_center_left[1] - eye_center_right[1]) ** 2) ** 0.5 * 0.1
            self.iris_left = self.eye_left.get_iris(self.radius_left)
            self.iris_right = self.eye_right.get_iris(self.radius_right)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.grad_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0)
            self.grad_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1)
            for epoch in range(10):
                self.iris_left, self.radius_left, self.edge_left = self.refine_iris(self.iris_left, self.radius_left, self.eye_left)
                self.iris_right, self.radius_right, self.edge_right = self.refine_iris(self.iris_right, self.radius_right, self.eye_right)

    def func_iris_error(self, pts, x0, y0, r0):
        v = lambda x: sum((((e[0] - x[0]) ** 2 + (e[1] - x[1]) ** 2) ** 0.5 - x[2]) ** 2 for e in pts) / len(pts) + 0.1 * ((x[0] - x0) ** 2 + (x[1] - y0) ** 2 + (x[2] - r0) ** 2)
        return v

    def refine_iris(self, iris, radius, eye):
        grad_x = self.grad_x
        grad_y = self.grad_y
        iris_edge_points = []
        cos_25_2 = np.cos(25 / 180.0 * np.pi) ** 2
        SAMPLES = 100

        angles = []
        for angle in range(-90, 270, 1):
            if (-45 <= angle and angle <= 45) or (135 <= angle and angle <= 225):
                radian = angle * np.pi / 180.0
                dx = np.cos(radian)
                dy = np.sin(radian)
                pos_x = int(iris[0] + radius * dx)
                pos_y = int(iris[1] + radius * dy)
                if eye.contains((pos_x, pos_y)):
                    angles.append(angle)
        if len(angles) >= SAMPLES:
            random.shuffle(angles)
            angles = angles[:SAMPLES]

        for angle in angles:
            radian = angle * np.pi / 180.0
            dx = np.cos(radian)
            dy = np.sin(radian)
            d_2 = dx ** 2 + dy ** 2
            max_grad = 0
            max_corr = 0
            max_x = 0
            max_y = 0
            for ratio in range(70, 130, 5):
                r = radius * 0.01 * ratio
                pos_x = int(iris[0] + r * dx)
                pos_y = int(iris[1] + r * dy)
                gx = grad_x[pos_y, pos_x]
                gy = grad_y[pos_y, pos_x]
                g_2 = gx ** 2 + gy ** 2
                grad = gx * dx + gy * dy
                corr_direction = grad ** 2 >= cos_25_2 * d_2 * g_2
                if grad > max_grad:
                    max_grad = grad
                    max_corr = corr_direction
                    max_x = pos_x
                    max_y = pos_y
            if max_x != 0 and max_y != 0 and max_corr and eye.contains((max_x, max_y)):
                iris_edge_points.append((max_x, max_y))
        
        dists = []
        for i in range(len(iris_edge_points)):
            dists.append(self.dist(iris_edge_points[i], iris))
        mean_dist = np.mean(dists)
        std_dist = np.std(dists)
        new_iris_edge = []
        for i in range(len(iris_edge_points)):
            if (mean_dist - std_dist <= dists[i] and dists[i] <= mean_dist + std_dist):
                new_iris_edge.append(iris_edge_points[i])

        if (len(new_iris_edge) >= SAMPLES / 5):
            res = minimize(self.func_iris_error(new_iris_edge, iris[0], iris[1], radius), (iris[0], iris[1], radius), method='SLSQP')
            if eye.contains((res.x[0], res.x[1])):
                iris = (res.x[0], res.x[1])
                radius = res.x[2]
        return iris, radius, new_iris_edge

    def get_iris_left(self):
        return self.iris_left, self.radius_left, self.edge_left

    def get_iris_right(self):
        return self.iris_right, self.radius_right, self.edge_right

    def get_eye_left(self):
        eye = []
        for i in range(36, 42):
            point = self.landmarks.part(i)
            eye.append((point.x, point.y))
        return eye
    
    def get_eye_right(self):
        eye = []
        for i in range(42, 48):
            point = self.landmarks.part(i)
            eye.append((point.x, point.y))
        return eye
