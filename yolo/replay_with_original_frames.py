import cv2

with open("final_boundaries_cropped_obj.txt", 'r') as f:
    boundaries = [line.strip() for line in f.readlines()]
    
frame_id = 0 
cap= cv2.VideoCapture("../violencec_1.mp4")

while True:
    
    ret,frame = cap.read()
    
    if not ret:
        break
    
    cv2.imshow("video", frame) 
    key = cv2.waitKey(1)
    if str(frame_id) in boundaries:
        print(f"frame Id : {frame_id}")
        key = cv2.waitKey(0)
        
    if(key==ord('p')):
        key = cv2.waitKey(1)
    if(key==ord('o')):
        print(frame_id)
        
        
    frame_id +=1 