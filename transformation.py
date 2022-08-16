from turtle import heading, width
import cv2 as cv
from cv2 import dilate
import numpy as np


img = cv.imread("test.jpg")
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edge = cv.Canny(gray,175,100)
dialated= cv.dilate(edge,(7,7),iterations=7)
def resizeImg(img,wprec,hprec):
    width=int(img.shape[1]*wprec)
    height= int(img.shape[0]*hprec)
    dim=(width,height)
    return cv.resize(img,dim,interpolation=cv.INTER_AREA)

resized = resizeImg(img,0.5,0.5)

def translate(img,x,y):
    transtMat = np.float32([[1,0,x],[0,1,y]])
    dimention = (img.shape[1],img.shape[0])
    return cv.warpAffine(img,transtMat,dimention)



def rotate(img,angle,point=None):
    (height,width) = img.shape[:2]
    
    if point is None:
       point = (width//2,height//2)
    rotMat = cv.getRotationMatrix2D(point,angle,1.0)
    dimentions =(width,height)
    return cv.warpAffine(img,rotMat,dimentions)


# cv.imshow("img",rotate(resized,60))
cv.waitKey(0)