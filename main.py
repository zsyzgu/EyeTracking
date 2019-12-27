import cv2
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
import os
from gaze import Gaze
import time
import test
import pyrealsense2 as rs

def set_laser(pipeline_profile, power):
    device = pipeline_profile.get_device()
    depth_sensor = device.query_sensors()[0]
    laser_pwr = depth_sensor.get_option(rs.option.laser_power)
    depth_sensor.set_option(rs.option.laser_power, power)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline_profile = pipeline.start(config)
set_laser(pipeline_profile, 0)

gaze = Gaze()

output_cnt = 0
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    anotated_image = test.annotate_image(gaze, color_image)
    cv2.imshow('Demo', anotated_image)

    output_cnt += 1
    cv2.imwrite('output_data/res_' + str(output_cnt) + '.jpg', anotated_image)
    cv2.imwrite('output_data/raw_' + str(output_cnt) + '.jpg', color_image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

pipeline.stop()
