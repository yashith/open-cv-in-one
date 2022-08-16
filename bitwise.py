import cv2 as cv
from cv2 import rectangle
import numpy as np

blank = np.zeros((400,400),dtype="uint8")

rectangle = cv.rectangle(blank.copy(),(30,30),(370,370),255,-1)
circle = cv.circle(blank.copy(),(200,200),200,255,-1)

i = cv.imread("test.jpg")
img = cv.resize(i,(int(i.shape[0]*0.5),int(i.shape[1]*0.5)),interpolation=cv.INTER_AREA)

cv.imshow("rectangle",rectangle)
cv.imshow("circle",cv.bitwise_not(rectangle,circle))

cv.waitKey(0)