import cv2 as cv
from cv2 import Laplacian
import numpy as np

i = cv.imread("test.jpg")
img = cv.resize(i,(int(i.shape[0]*0.5),int(i.shape[1]*0.5)),interpolation=cv.INTER_AREA)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#cv.imshow("gray",gray)

#Laplasian
lap = cv.Laplacian(gray,cv.CV_64F)
lap = np.uint8(np.absolute(lap))
#cv.imshow("lap",lap)

#Sobel
sobelx =cv.Sobel(gray,cv.CV_64F,1,0)
sobely = cv.Sobel(gray,cv.CV_64F,0,1)

cv.imshow("Sobelx",sobelx)
cv.imshow("Sobely",sobely)

cv.waitKey(0)