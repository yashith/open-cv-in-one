from ast import Dict, List, Str
import cv2
from yolo import get_yolo_objects,draw_prediction
from huediff import diff_hist
from motion_blur import check_blur,check_blur_wavelet
from harr_wavelet_blur import blur_detect
import numpy as np
import math
from configparser import ConfigParser

config = ConfigParser()
config.read("E:\OpenCV tests\yolo\configs.ini")
v_path = config['video']['path']
main_obj_size = int(config['boundary']['main_obj_size'])
har_wavelet_tresh = float(config['boundary']['har_wavelet_tresh'])
har_decision_high_thresh = float(config['boundary']['har_decision_thresh_high_blur'])
har_decision_low_thresh = float(config['boundary']['har_decision_thresh_low_blur'])
bh_distance_thresh = float(config['boundary']['bh_distance_thresh'])


boundaries = None
classes = None
previos_hco_dict={}
with open("yolo3_classes.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
with open("hsv_frames.txt", 'r') as f:
    boundaries = [line.strip() for line in f.readlines()]

frame_id = 0
file = open("boundary_objects.txt", "w")
file2 = open("final_boundaries.txt", "w")
file3 = open("final_boundaries_cropped_obj.txt", "w")
cap = cv2.VideoCapture(v_path)


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

def get_mid_point(box):
    middle_point = (round(box[0]+box[2]/2),round(box[1]+box[3]/2))
    return middle_point

def get_nearest_object(middle_point,boxes,classes,prev_class):
    smallest_distance = math.inf
    nearest_object_index=None
    for i,box in enumerate(boxes):
        if(classes[i]==prev_class):
            b_mid = get_mid_point(box)
            distance_to_main_object = (abs(middle_point[0])-abs(b_mid[0]))**2  + (abs(middle_point[1])-abs(b_mid[1]))**2
            if(distance_to_main_object<smallest_distance):
                nearest_object_index=i
                smallest_distance = distance_to_main_object
    return nearest_object_index

def crop_main_obj(frame, boxes, confidences,class_ids):
    big_enough_boxes=[]
    big_enough_box_confidences=[]
    big_enough_box_class_ids=[]
     # set minimum considering object size to 1/100 of frame
    min_object_size = int(frame.shape[0]*frame.shape[1]/main_obj_size) 
    for i in range(2):
        for i,box in enumerate(boxes):
            if(box[2]*box[3]>min_object_size and box[0]>0 and box[1]>0 and box[2]>0 and box[3]>0 and confidences[i] >0.75): # check big enough or remove negative values if exist and confidence >0.75
                big_enough_boxes.append(box)
                big_enough_box_confidences.append(confidences[i]) 
                big_enough_box_class_ids.append(class_ids[i])  
        if(len(big_enough_boxes)==0): # check big enough boxes twice 1st time full size, if nothing found full size/2
            min_object_size = int(min_object_size/2)
    if(len(big_enough_boxes) != 0):
        max_index = big_enough_box_confidences.index(max(big_enough_box_confidences))
        print(big_enough_box_confidences[max_index])
        x = round(big_enough_boxes[max_index][0])
        y = round(big_enough_boxes[max_index][1])
        x_w = round(big_enough_boxes[max_index][0]+big_enough_boxes[max_index][2])
        y_h = round(big_enough_boxes[max_index][1]+big_enough_boxes[max_index][3])
        middle_point = get_mid_point(big_enough_boxes[max_index])
        crop_img = frame[y:y_h, x:x_w]
        return crop_img,big_enough_box_class_ids[max_index],middle_point
    return None,None,None
def crop_obj(frame, boxes, index):
    if(len(boxes) != 0):
        x = round(boxes[index][0])
        y = round(boxes[index][1])
        x_w = round(boxes[index][0]+boxes[index][2])
        y_h = round(boxes[index][1]+boxes[index][3])
        crop_img = frame[y:y_h, x:x_w]
        return crop_img
    return None
def get_hco_from_memory(dictionary,lb,ub):
    hco = []
    for key, value in dictionary.items():
        if isinstance(key, int) and key > lb and key < ub:
            hco.append(value)
    return hco
obj_arr = None
prev_frame_hco = None
prev_class_id = None
prev_hco_mid =None
current_frame_hco = None
while True:
    ret, frame = cap.read()

    if not ret:
        break
    cv2.imshow("vid", frame)
    cv2.waitKey(1)
    
    testing_frames=[2057,2232]
    if(frame_id in testing_frames):
        print("testing frame")
        ##
    
    color = (255, 0, 0)
    # check current frame in the saved list
    
    if str(frame_id+1) in boundaries:
        prev_frame_blur = False
        indices, class_ids, boxes, confidences = get_yolo_objects(frame)
        try:
            prev_frame_hco,prev_class_id, prev_hco_mid = crop_main_obj(frame,boxes,confidences,class_ids)
            if(type(prev_frame_hco)is np.ndarray):
                if(prev_frame_hco.size==0):
                    per, blurext = blur_detect(frame,har_wavelet_tresh)
                    prev_frame_blur = per < har_decision_low_thresh # Decision Threshold considered as 0.001 
                    if(not prev_frame_blur):           
                        # file3.write(f"{frame_id+1}")
                        # file3.write("\n")
                        print("Blur not found in prev frame_hco and size 0")
                    else:
                        boundaries.remove(frame_id)
                        #reset all
                        prev_frame_hco = None
                        prev_class_id = None
                        prev_hco_mid =None
                        current_frame_hco = None
            elif(prev_frame_hco == None):
                per, blurext = blur_detect(frame,har_wavelet_tresh)
                prev_frame_blur = per < har_decision_low_thresh # Decision Threshold considered as 0.001 
                if(not prev_frame_blur):           
                    # file3.write(f"{frame_id+1}")
                    # file3.write("\n")
                    print("Blur not found in prev frame_hco and None continuing search in next frame")
                    # if no no objects detected and not blur consider the next frame to detect objects
                    boundaries[boundaries.index(str(frame_id+1))] =str(frame_id+2)
                else:
                    boundaries.remove(frame_id)
                    #reset all
                    prev_frame_hco = None
                    prev_class_id = None
                    prev_hco_mid =None
                    current_frame_hco = None
        except:
            print(f"{frame_id} Prev farme error")
    elif str(frame_id) in boundaries:
        indices, class_ids, boxes, confidences = get_yolo_objects(frame)
        
        per, blurext = blur_detect(frame,har_wavelet_tresh)
        neareset_obj_index = get_nearest_object(prev_hco_mid,boxes,class_ids,prev_class_id)
        # write object to file
        file.write(f"frame - {frame_id}")
        file.write("\n")

        matching_object_exist =False
        is_blur = per < har_decision_high_thresh # Decision Threshold considered as 0.001 
        
        if((len(class_ids)==0 or prev_class_id not in class_ids) and is_blur):  # if high blur found dont consider object detection         
            matching_object_exist =True #Since considering motion blur is only in same scene
            print("High Blur found")
        for i,class_id in enumerate(class_ids):
            if(prev_class_id != None and neareset_obj_index != None and neareset_obj_index==i):        
                current_frame_hco = crop_obj(frame,boxes,i)
                # draw_prediction(frame,class_id,0,boxes[i][0],boxes[i][1],boxes[i][0]+boxes[i][2],boxes[i][1]+boxes[i][3])
                b_frame = cv2.rectangle(frame, (round(boxes[i][0]),round(boxes[i][1])), (round(boxes[i][0]+boxes[i][2]),round(boxes[i][1]+boxes[i][3])), (255,0,0), 2)
                # cv2.destroyAllWindows()
                cv2.imshow("vid", b_frame)
                cv2.imshow("prev_frame",prev_frame_hco)
                #cv2.imshow("current_frame",current_frame_hco)
                cv2.waitKey(1)
                if(current_frame_hco.size!=0):
                    distance = diff_hist(prev_frame_hco,current_frame_hco)
                    print(f'{frame_id} - {distance}')
                    cv2.imwrite(f"cropped_obj/{frame_id}.png",prev_frame_hco)
                    cv2.imwrite(f"cropped_obj/{frame_id}_1.png",current_frame_hco)
                    
                    #add extracted object into a dictionary to compare later
                    previos_hco_dict.update({(frame_id-1):prev_frame_hco})
                    previos_hco_dict.update({frame_id:current_frame_hco})
                    
                    if(distance<=bh_distance_thresh):
                        matching_object_exist= True   
                        break
                    
                    # check memory buffer within 20 frames
                    obj_from_memory = get_hco_from_memory(previos_hco_dict,(frame_id -100),(frame_id-1))
                    
                    if(len(obj_from_memory)!=0):
                        for hco in obj_from_memory:
                            distance = diff_hist(hco,current_frame_hco)
                            if(distance<=bh_distance_thresh):
                                matching_object_exist= True
                                print("Matching object found in the memory")
                                break
            else:
                if(type(prev_frame_hco)is np.ndarray):
                    if(prev_frame_hco.size==0 and prev_frame_blur):
                        matching_object_exist= True
                        print("No obj or blur found")
                        break
                elif(prev_frame_hco==None and prev_frame_blur):
                    matching_object_exist= True
                    print("No obj or blur found")
                    break
                elif(neareset_obj_index == None or current_frame_hco.size!=0):
                    break
                    
        if(not matching_object_exist):
            file3.write(f"{frame_id}")
            file3.write("\n") 
            matching_object_exist = False
        #reset all
        prev_frame_hco = None
        prev_class_id = None
        prev_hco_mid =None
        current_frame_hco = None
        #for checking object confidence (Not functional)
        for index, i in enumerate(indices):
            file.write(
                f"{classes[class_ids[i]]} - confidence = {confidences[index]} | ")
        file.write("\n")

        # check object similarity in boundaris

        if obj_arr != None and compair_dict(obj_arr, prec_similar_obj(class_ids)):
            print(f"frame {frame_id}")
        else:
            per, blurext = blur_detect(frame,har_wavelet_tresh)
            file2.write(f"{frame_id} - {per}")
            file2.write(f"\n")
        obj_arr = prec_similar_obj(class_ids)
    else:
        obj_arr = None

    frame_id += 1
