import cv2
import argparse
import numpy as np
import math

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=False,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=False,
                help = 'path to yolo config file',default="./yolov3.cfg")
ap.add_argument('-w', '--weights', required=False,
                help = 'path to yolo pre-trained weights', default="./yolov3.weights")
ap.add_argument('-cl', '--classes', required=False,
                help = 'path to text file containing class names', default="./yolo3_classes.txt")
args = ap.parse_args()

classes = None

with open("yolo3_classes.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = (18, 255, 255)

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    
def get_yolo_objects(image):
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    net = cv2.dnn.readNet(args.weights, args.config)

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    return(indices,class_ids,boxes,confidences)

''''    
cap= cv2.VideoCapture("../test_3.mp4")

file = open("newsframes2.txt",'w')
current_frame = 1

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
    
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

while True:
    ret,image = cap.read()
    if not ret:
        break
    image = cv2.resize(image,(int(image.shape[1]*0.5),int(image.shape[0]*0.5)))
    

    indices,class_ids,boxes,confidences = get_yolo_objects(image)
    
    file.write(str(current_frame)+" - ")
    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]
        
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        print(classes[class_ids[i]])
        file.write(str(classes[class_ids[i]])+", ")
    file.write("\n") 
    current_frame+=1
    cv2.imshow("object detection", image)
    key=cv2.waitKey(1)
    
    if(key==ord('q')):
        break
        
    #cv2.imwrite("object-detection.jpg", image)
'''