from __future__ import print_function
import cv2 as cv
import argparse
import lk

parser = argparse.ArgumentParser()
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=False)
else:
    backSub = cv.createBackgroundSubtractorKNN(history=100, dist2Threshold=400)
capture = cv.VideoCapture("V_340.mp4")
ret, frame1 = capture.read()
if not capture.isOpened():
    exit(0)
while True:
    ret, frame2 = capture.read()
    if frame2 is None:
        break
    
    frame = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    frame = cv.GaussianBlur(frame, (5, 5), 0)
    frame = cv.equalizeHist(frame)
    
    fgMask = backSub.apply(frame)
    
    
    frame = lk(frame1,frame)
    
    cv.imshow('Optical flow', frame)
    cv.imshow('Background Mask', fgMask)
    #cv.imshow('Original', frame2)
    frame1 = frame2
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    elif keyboard == 'p':
        keyboard = cv.waitKey(0)
    elif keyboard == 'c':
        cv.waitKey(0)
        keyboard = cv.waitKey(30)