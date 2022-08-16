import cv2 as cv
from cv2 import imshow
import numpy as np
imshow=cv.imshow
i = cv.imread("test.jpg")

img = cv.resize(i,(int(i.shape[0]*0.5),int(i.shape[1]*0.5)),interpolation=cv.INTER_AREA)
black= np.zeros(img.shape[:2],dtype='uint8')
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("colord",img)
# cv.imshow("Gray",gray)

# hsv= cv.cvtColor(img,cv.COLOR_BGR2HSV)
# cv.imshow("hsv",hsv)
# lab = cv.cvtColor(img,cv.COLOR_BGR2Lab)
# cv.imshow("lab",lab)

r,g,b = cv.split(img)
blue = cv.merge ([b,black,black])
red = cv.merge([black,black,r])
green = cv.merge([black,g,black])

imshow("blue",blue)
imshow("green",green)
imshow("red",red)
cv.waitKey(0)