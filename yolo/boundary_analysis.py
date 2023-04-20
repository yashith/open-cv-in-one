from ast import Dict, List, Str
import cv2
from yolo import get_yolo_objects
from huediff import diff_hist
from motion_blur import check_blur,check_blur_wavelet
import numpy as np

boundaries = None
classes = None

with open("yolo3_classes.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
with open("hsv_frames.txt", 'r') as f:
    boundaries = [line.strip() for line in f.readlines()]

frame_id = 0
file = open("boundary_objects.txt", "w")
file2 = open("final_boundaries.txt", "w")
file3 = open("final_boundaries_cropped_obj.txt", "w")
cap = cv2.VideoCapture("../Videos/news.mp4")


def prec_similar_obj(arr1, arr2=None):
    obj_arr1 = {}

    for key in arr1:

        object_name = classes[key]

        if(object_name in obj_arr1):
            obj_arr1[object_name] = obj_arr1[object_name]+1
        else:
            obj_arr1[object_name] = 1

    # print(obj_arr1)
    return obj_arr1


def compair_dict(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    else:
        return True
    #     for key in dict1.keys():
    #         if dict1[key]!=dict2[key]:
    #             return False
    # return True


def crop_main_obj(frame, boxes, confidences,class_ids):
    big_enough_boxes=[]
    big_enough_box_confidences=[]
    big_enough_box_class_ids=[]
     # set minimum considering object size to 1/8 of frame
    min_object_size = int(frame.shape[0]*frame.shape[1]/8) 
    for i,box in enumerate(boxes):
        if(box[2]*box[3]>min_object_size and box[0]>0 and box[1]>0 and box[2]>0 and box[3]>0): # check big enough or remove negative values if exist
           big_enough_boxes.append(box)
           big_enough_box_confidences.append(confidences[i]) 
           big_enough_box_class_ids.append(class_ids[i])  
    if(len(big_enough_boxes) != 0):
        max_index = big_enough_box_confidences.index(max(big_enough_box_confidences))
        print(big_enough_box_confidences[max_index])
        x = round(big_enough_boxes[max_index][0])
        y = round(big_enough_boxes[max_index][1])
        x_w = round(big_enough_boxes[max_index][0]+big_enough_boxes[max_index][2])
        y_h = round(big_enough_boxes[max_index][1]+big_enough_boxes[max_index][3])
        crop_img = frame[y:y_h, x:x_w]
        return crop_img,big_enough_box_class_ids[max_index]
    return None,None
def crop_obj(frame, boxes, index):
    if(len(boxes) != 0):
        print(confidences[index])
        x = round(boxes[index][0])
        y = round(boxes[index][1])
        x_w = round(boxes[index][0]+boxes[index][2])
        y_h = round(boxes[index][1]+boxes[index][3])
        crop_img = frame[y:y_h, x:x_w]
        return crop_img
    return None
obj_arr = None
prev_frame_hco = None
prev_class_id = None
current_frame_hco = None
while True:
    ret, frame = cap.read()

    if not ret:
        break
    cv2.imshow("vid", frame)
    cv2.waitKey(1)
    
    testing_frames=[2069]
    if(frame_id in testing_frames):
        print("testing frame")
        ##
    
    color = (255, 0, 0)
    # check current frame in the saved list
    
    if str(frame_id+1) in boundaries:
        prev_frame_blur = False
        indices, class_ids, boxes, confidences = get_yolo_objects(frame)
        try:
            prev_frame_hco,prev_class_id = crop_main_obj(frame,boxes,confidences,class_ids)
            if(type(prev_frame_hco)is np.ndarray):
                if(prev_frame_hco.size==0):
                    prev_frame_blur = check_blur_wavelet(frame,200)
                    if(not prev_frame_blur):           
                        # file3.write(f"{frame_id+1}")
                        # file3.write("\n")
                        print("Blur not found in prev frame")
            elif(prev_frame_hco == None):
                prev_frame_blur = check_blur_wavelet(frame,200)
                if(not prev_frame_blur):           
                    # file3.write(f"{frame_id+1}")
                    # file3.write("\n")
                    print("Blur not found in prev frame")
        except:
            print(f"{frame_id} Prev farme error")
    elif str(frame_id) in boundaries:
        indices, class_ids, boxes, confidences = get_yolo_objects(frame)

        # write object to file
        file.write(f"frame - {frame_id}")
        file.write("\n")

        # get max confidence item
        max_index = None
        matching_object_exist =False
        ##testing
        is_blur = check_blur_wavelet(frame,100)
        if((len(class_ids)==0 or prev_class_id not in class_ids) and is_blur):           
            matching_object_exist =True
            print("Blur found")
        for i,class_id in enumerate(class_ids):
            if(class_id==prev_class_id):        
                current_frame_hco = crop_obj(frame,boxes,i)
                try:
                    cv2.destroyAllWindows()
                    cv2.imshow("vid", frame)
                    cv2.imshow("prev_frame",prev_frame_hco)
                    cv2.imshow("current_frame",current_frame_hco)
                    cv2.waitKey(1)
                    
                    distance = diff_hist(prev_frame_hco,current_frame_hco)
                    print(f'{frame_id} - {distance}')
                    if(distance<=0.8):
                        matching_object_exist= True
                        break   
                    
                except:
                    if(type(prev_frame_hco)is np.ndarray):
                        if(prev_frame_hco.size==0 and prev_frame_blur):
                            matching_object_exist= True
                            print("No obj or blur found")
                            break
                    elif(prev_frame_hco==None and prev_frame_blur):
                        matching_object_exist= True
                        print("No obj or blur found")
                        break
                    elif(current_frame_hco.size!=0):
                        break
                    
        if(not matching_object_exist):
            file3.write(f"{frame_id}")
            file3.write("\n") 
            matching_object_exist = False
        prev_frame_hco=None
        #for checking object confidence (Not functional)
        for index, i in enumerate(indices):
            file.write(
                f"{classes[class_ids[i]]} - confidence = {confidences[index]} | ")
        file.write("\n")

        # check object similarity in boundaris

        if obj_arr != None and compair_dict(obj_arr, prec_similar_obj(class_ids)):
            print(f"frame {frame_id}")
        else:
            file2.write(f"{frame_id}")
            file2.write(f"\n")
        obj_arr = prec_similar_obj(class_ids)
    else:
        obj_arr = None

    frame_id += 1
