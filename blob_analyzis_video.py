import lk
import cv2 as cv
import numpy as np


cap = cv.VideoCapture("Hands - 38079.mp4")
_,prev_frame = cap.read()

while True:
    ret,current_frame = cap.read()
    
    if not ret:
        break
    
    frame = cv.cvtColor(current_frame,cv.COLOR_BGR2GRAY)
    frame = cv.GaussianBlur(frame, (5, 5), 0)
    frame = cv.equalizeHist(frame)
    
    mf_frame = lk(prev_frame,frame)
    
    gray = cv.cvtColor(mf_frame, cv.COLOR_BGR2GRAY)

    # Threshold the image to get a binary image
    #_, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    # Set up the SimpleBlobDetector parameters
    params = cv.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 1000


    # Filter by Area.
    params.filterByArea = True
    params.minArea = 50

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(mf_frame)

    # Draw blobs on the image
    img_with_keypoints = cv.drawKeypoints(current_frame, keypoints, np.array([]), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show the output image
    cv.imshow("Blobs", gray)
    cv.waitKey(1)
    prev_frame = current_frame
    