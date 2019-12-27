import cv2
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
import os
from gaze import Gaze
import time

def annotate_iris(image, center, radius, color=(255, 255, 0)):
    center = (int(center[0]), int(center[1]))
    radius = int(radius)
    cv2.circle(image, center, radius, color)
    x = center[0]
    y = center[1]
    cv2.line(image, (x - 5, y), (x + 5, y), color)
    cv2.line(image, (x, y - 5), (x, y + 5), color)

def annotate_eye(image, eye, color = (0, 255, 255)):
    N = len(eye)
    for i in range(N):
        cv2.line(image, eye[i], eye[(i + 1) % N], color)

def annotate_edge(image, edge, color = (255, 0, 255)):
    if None != edge:
        for point in edge:
            cv2.circle(image, point, 0, color)

def annotate_image(gaze, image):
    gaze.refresh(image)
    result = image.copy()
    left_iris, left_radius, left_edge = gaze.get_iris_left()
    right_iris, right_radius, right_edge = gaze.get_iris_right()
    if left_iris != None:
        left_eye = gaze.get_eye_left()
        annotate_eye(result, left_eye)
        annotate_iris(result, left_iris, left_radius)
        annotate_edge(result, left_edge)
    if right_iris != None:
        right_eye = gaze.get_eye_right()
        annotate_eye(result, right_eye)
        annotate_iris(result, right_iris, right_radius)
        annotate_edge(result, right_edge)
    return result

if __name__ == "__main__":
    '''
    gaze = Gaze()
    image = cv2.imread('./output_data/raw_13.jpg')
    anotated_image = annotate_image(gaze, image)
    cv2.imshow('Demo', anotated_image)
    cv2.waitKey(0)
    '''
    gaze = Gaze()
    root = './test_data/'
    files = os.listdir(root)
    output_cnt = 0
    for file in files:
        image = cv2.imread(root + file)
        anotated_image = annotate_image(gaze, image)
        cv2.imshow('Demo', anotated_image)
        cv2.waitKey(1)
        
        output_cnt += 1
        cv2.imwrite('output_data/res_' + str(output_cnt) + '.jpg', anotated_image)
        cv2.imwrite('output_data/raw_' + str(output_cnt) + '.jpg', image)
    print(np.mean(gaze.estimated_error), np.std(gaze.estimated_error))
