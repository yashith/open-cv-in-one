import cv2 as cv
import numpy as np


cap = cv.VideoCapture("videoplayback.mp4")

current_frame = 1
file=open("scenes.txt","w")
while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv.imshow("Video", frame)
    key = cv.waitKey(100)
    if key == ord('p'):

        file.write(str(current_frame)+"\n")
        
    elif key == ord('q'):
        break
       
    current_frame += 1
    
file.close()