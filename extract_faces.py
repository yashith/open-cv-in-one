from email.mime import base
import os
import cv2 as cv
from cv2 import Laplacian
import numpy as np



base_DIR =r""
image_list = os.listdir(base_DIR)
fname=1

for file in image_list:
    try:
        i = cv.imread(os.path.join(base_DIR,file))
        dim=()
        if(i.shape[1]<480):
            dim = (i.shape[1],i.shape[0])
        else:
            dim=(480,int(480/i.shape[1]*i.shape[0]))
        img = cv.resize(i,dim,interpolation=cv.INTER_AREA)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(img,(7,7),0)
        haar_cascade = cv.CascadeClassifier('harr.xml')

        face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1,
                minNeighbors=10,
                minSize=(30, 30),
                flags=cv.CASCADE_SCALE_IMAGE)

        #print(f"number of faces = {len(face_rect)}")

        cropped=img
        for(x,y,w,h)in face_rect:
            #cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
            if(len(face_rect)==1):
                cropped = img[y:y+h,x:x+w]
                #cropped= cv.resize(int(cropped.shape[1]*2),int(cropped.shape[0]*2))
                p=os.path.join(base_DIR,"faces")
                file_name= os.path.join(p,str(fname)+".jpg")
                cv.imwrite(file_name,cropped)
                print(str(fname)+" - " + file)
                fname+=1
        cv.imshow("Detected face",cropped)
        
        
    except:
        print("Err")

    #cv.waitKey(0)