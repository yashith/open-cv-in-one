import cv2
from configparser import ConfigParser

config = ConfigParser()
config.read("E:\OpenCV tests\yolo\configs.ini")
v_path = config['video']['path']
file_name = v_path.split('\\')[-1].split(".")[0]

with open("final_boundaries_cropped_obj.txt", 'r') as f:
    boundaries = [line.strip() for line in f.readlines()]
   
frame_id = 0
last_boundary =-1
cap= cv2.VideoCapture(v_path)
output_path = f'cropped_vid/{file_name}_parts.mp4'
output_fps = cap.get(cv2.CAP_PROP_FPS)
output_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, output_fps, output_size)
while True:
    
    ret,frame = cap.read()
    
    if not ret:
        break
    
    cv2.imshow("video", frame) 
    key = cv2.waitKey(10)
    if str(frame_id) in boundaries:
        print(f"frame Id : {frame_id}")
        # key = cv2.waitKey(0)
        if (frame_id - last_boundary > 30):
            last_boundary = frame_id
            print(f"frame Id : {frame_id}")
            last_boundary = frame_id
            # key = cv2.waitKey(0)
            out.release()
            output_path= f"cropped_vid/{file_name}_parts{frame_id}.mp4"
            out = cv2.VideoWriter(output_path, fourcc, output_fps, output_size)
        else:
            last_boundary = frame_id
    out.write(frame)
    if(key==ord('p')):
        key = cv2.waitKey(50)
    if(key==ord('o')):
        print(frame_id)
        
        
    frame_id +=1 