import cv2 as cv
import numpy as np
from transformation import resizeImg
i = cv.imread("test.jpg")
img = resizeImg(i, 0.5, 0.5)

bw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret,tresh = cv.threshold(bw,125,255,cv.THRESH_BINARY)
canny = cv.Canny(bw,125,125)
blank =  np.zeros(img.shape,dtype='uint8')


contours, heirarchies = cv.findContours(tresh,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
cont=cv.drawContours(blank,contours,-1,(255,255,255),2)

cv.imshow("canny edges",canny)
cv.imshow("Thresh",tresh)
cv.imshow("contours",cont)
cv.waitKey(0)
