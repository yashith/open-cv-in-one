import glob
import os
import cv2
from configparser import ConfigParser

config = ConfigParser()
config.read("E:\OpenCV tests\yolo\configs.ini")
v_path = config['video']['path']

files = glob.glob('frame_by_id/*')
for f in files:
    os.remove(f)
    
frame_id = 0 
cap= cv2.VideoCapture(v_path)

while True:
    
    ret,frame = cap.read()
    
    if not ret:
        break
    
    # cv2.imshow("video", frame) 
    # key = cv2.waitKey(1)
    print(frame_id)
    
    cv2.imwrite(f"frame_by_id/{frame_id}.png",frame) 
        
    frame_id +=1 