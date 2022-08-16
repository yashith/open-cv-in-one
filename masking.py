import cv2 as cv
import numpy as np

i = cv.imread("test.jpg")
img = cv.resize(i,(int(i.shape[0]*0.5),int(i.shape[1]*0.5)),interpolation=cv.INTER_AREA)

blank = np.zeros(img.shape[:2],dtype="uint8")
cv.imshow("blank",blank)

mask = cv.circle(blank,(img.shape[1]//2,img.shape[0]//4),100,255,-1)
cv.imshow("masked",mask)
masked = cv.bitwise_and(img,img,mask=mask)
cv.imshow("masked",masked)


cv.waitKey(0)