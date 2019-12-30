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

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 6)
#config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)
#config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
#config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
#align2depth = rs.align(rs.stream.depth)
pipeline_profile = pipeline.start(config)
depth_sensor = pipeline_profile.get_device().query_sensors()[0]
depth_sensor.set_option(rs.option.laser_power, 0)
depth_sensor.set_option(rs.option.exposure, 20000)
depth_sensor.set_option(rs.option.gain, 100)

output_cnt = 0
while True:
    frames = pipeline.wait_for_frames()
    #frames = align2depth.process(frames)
    frame = frames.get_infrared_frame(1)
    if not frame:
        continue

    image = np.asanyarray(frame.get_data())
    cv2.imshow('Demo', image)
    
    output_cnt += 1
    cv2.imwrite('output_data/raw_' + str(output_cnt) + '.jpg', image)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
