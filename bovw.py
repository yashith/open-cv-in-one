from typing import Any
import cv2 as cv
import numpy as np

i = cv.imread("jpg")
cap = cv.VideoCapture("Git Tutorial for Beginners- Learn Git in 1 Hour.mp4")

extractor = cv.xfeatures2d.SIFT_create()
def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image,None)
    return keypoints, descriptors

while True:
    ret, img = cap.read()
    if not ret:
        print("can't resolve frame(stream end)")
        break
    keypoints, descriptors = features(img,extractor)
    keypoints=cv.KeyPoint_convert(keypoints=keypoints)
    for keypoint in keypoints:
        cv.circle(img,(int(keypoint[0]),int(keypoint[1])),10,color=(0,255,0))
    cv.imshow("img",img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


# img = cv.resize(i,(int(i.shape[1]*0.5),int(i.shape[0]*0.5)),interpolation=cv.INTER_AREA)
# img = cv.cvtColor(img,cv.COLOR_RGB2BGRA)



cv.waitKey(0)