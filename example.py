import cv2
import numpy as np
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(1)

while True:
    _, frame = webcam.read()
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    #print('Left pupil:', str(left_pupil), '; Right pupil:', str(right_pupil))

    cv2.imshow("Demo", frame)
    if cv2.waitKey(1) == 27:
        break
