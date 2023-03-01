import cv2 as cv
import numpy as np

cap = cv.VideoCapture('V_392.mp4')

model_config = "yolo/yolov3-tiny.cfg"
model_weights = "yolo/yolov3-tiny.weights"
net = cv.dnn.readNetFromDarknet(model_config, model_weights)
output_layers = net.getUnconnectedOutLayersNames()

while cap.isOpened():
    _,frame =  cap.read()
    print(frame)
    height,width = frame.shape[:2]
    
    blob = cv.dnn.blobFromImage(frame, 1/255, (416,416), (0,0,0), swapRB=True, crop=False)
    
    net.setInput(blob)
    detections = net.forward(output_layers)
    
    for detection in detections:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores.max()
        if confidence > 0.5:
            for d in detection:
                center_x = int(d[0] * width)
                center_y = int(d[1] * height)
                w = int(d[2] * width)
                h = int(d[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # Draw the bounding box
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Add the class label and confidence score
                cv.putText(frame, f"{class_id}: {confidence:.2f}", (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv.imshow("test", frame)
    if cv.waitKey(100) & 0xFF == ord('q'):
        break