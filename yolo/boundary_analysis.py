from ast import Dict, List, Str
import cv2
from yolo import get_yolo_objects
from huediff import diff_hist
from motion_blur import check_blur,check_blur_wavelet
import traceback

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
cap = cv2.VideoCapture("../Videos/test5.mp4")


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
    if(len(boxes) != 0):
        max_index = confidences.index(max(confidences))
        print(confidences[max_index])
        x = round(boxes[max_index][0])
        y = round(boxes[max_index][1])
        x_w = round(boxes[max_index][0]+boxes[max_index][2])
        y_h = round(boxes[max_index][1]+boxes[max_index][3])
        crop_img = frame[y:y_h, x:x_w]
        return crop_img,class_ids[max_index]
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

    color = (255, 0, 0)
    # check current frame in the saved list
    if str(frame_id+1) in boundaries:
        indices, class_ids, boxes, confidences = get_yolo_objects(frame)
        try:
            prev_frame_hco,prev_class_id = crop_main_obj(frame,boxes,confidences,class_ids)
            if(type(prev_frame_hco) is None):
                is_blur = check_blur_wavelet(frame,150)
                if(not is_blur):           
                    # file3.write(f"{frame_id+1}")
                    # file3.write("\n")
                    print("Blur not found in prev frame")
        except:
            print("Prev farme error")
    elif str(frame_id) in boundaries:
        indices, class_ids, boxes, confidences = get_yolo_objects(frame)

        # write object to file
        file.write(f"frame - {frame_id}")
        file.write("\n")

        # get max confidence item
        max_index = None
        matching_object_exist =False
        for i in indices:
            if(class_ids[i]==prev_class_id):        
                current_frame_hco = crop_obj(frame,boxes,i)
                try:
                    
                    distance = diff_hist(prev_frame_hco,current_frame_hco)
                    print(f'{frame_id} - {distance}')
                    if(distance<=0.4):
                        matching_object_exist= True
                        cv2.imshow("prev_frame",prev_frame_hco)
                        cv2.imshow("current_frame",current_frame_hco)
                        break   
                    
                except:
                    is_blur = check_blur_wavelet(frame,150)
                    if(not is_blur):           
                        file3.write(f"{frame_id}")
                        file3.write("\n")
                        print("Blur not found")
                    print("No obj or blur found")
        if(not matching_object_exist):
            file3.write(f"{frame_id}")
            file3.write("\n") 
            matching_object_exist = False
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

    cv2.imshow("vid", frame)
    cv2.waitKey(1)

    frame_id += 1
