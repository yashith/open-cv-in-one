from ast import Dict, List, Str
import cv2 
from yolo import get_yolo_objects
boundaries=None
classes = None

with open("yolo3_classes.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
with open("hsv_frames_news.txt", 'r') as f:
    boundaries = [line.strip() for line in f.readlines()]
    
frame_id = 0 
file = open("boundary_objects.txt","w")
file2  = open("final_boundaries.txt","w")
cap= cv2.VideoCapture("../news.mp4")


def prec_similar_obj(arr1,arr2=None):
    obj_arr1 = {}
    
    for key in arr1:
        
        object_name = classes[key]
        
        if( object_name in obj_arr1):
            obj_arr1[object_name]=obj_arr1[object_name]+1
        else:
            obj_arr1[object_name]=1
    
    #print(obj_arr1)
    return obj_arr1

def compair_dict(dict1,dict2):
    if dict1.keys() != dict2.keys():
        return False
    else:
        return True
    #     for key in dict1.keys():
    #         if dict1[key]!=dict2[key]:
    #             return False
    # return True

obj_arr=None
while True:
    
    ret,frame = cap.read()
    
    if not ret:
        break
    
    #check object in boundaries
    # str(frame_id-1) in boundaries or 
    if str(frame_id) in boundaries or str(frame_id+1) in boundaries :
        indices,class_ids,boxes,confidences = get_yolo_objects(frame)
            
        #write object to file
        
        file.write(f"frame - {frame_id}")
        file.write("\n")
        
        #draw boxes for 1st object
        color = (255, 0, 0)
        if(boxes):
            x=round(boxes[0][0])
            y=round(boxes[0][1])
            x_w=round(boxes[0][0]+boxes[0][2])
            y_h=round(boxes[0][0]+boxes[0][3])
            cv2.rectangle(frame, (x,y), (x_w,y_h),color, 2)
            crop_img = frame[y:y_h, x:x_w]
            cv2.imshow("cropped",crop_img)
        
        key = cv2.waitKey(0)
        if(key==ord('p')):
            key = cv2.waitKey(1)
        
        for i in indices:
            file.write(classes[class_ids[i]]+" ")
        file.write("\n")
        
        #check object similarity in boundaris
        
        if obj_arr != None and compair_dict(obj_arr,prec_similar_obj(class_ids)):
            print(f"frame {frame_id}")
        else:
            file2.write(f"{frame_id}")
            file2.write(f"\n")
        obj_arr = prec_similar_obj(class_ids)
    else:
        obj_arr =None        
        
    cv2.imshow("vid",frame)
    cv2.waitKey(1)
    
    frame_id+=1
