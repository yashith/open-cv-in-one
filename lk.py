import sys
import numpy as np
import cv2 as cv


def lk(frame1, frame2):
    ########
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    ########

    next = frame2
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    gray = cv.cvtColor(bgr,cv.COLOR_BGR2GRAY)
    
    ret, mask = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
    
    return bgr

sys.modules[__name__] = lk