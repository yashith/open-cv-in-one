import cv2 as cv
import matplotlib.pyplot as plt

i = cv.imread("test.jpg")
img = cv.resize(i,(int(i.shape[0]*0.5),int(i.shape[1]*0.5)),interpolation=cv.INTER_AREA)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray)
threshold, thresh = cv.threshold(gray,150,255,cv.THRESH_BINARY)
cv.imshow("thresh",thresh)

adaptive_thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,3)

cv.imshow("adaptive",adaptive_thresh)

cv.waitKey(0)