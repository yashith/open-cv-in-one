
import cv2 as cv

i = cv.imread("test.jpg")
img = cv.resize(i,(int(i.shape[0]*0.5),int(i.shape[1]*0.5)),interpolation=cv.INTER_AREA)

avg = cv.blur(img,(7,7))
gauss= cv.GaussianBlur(img,(7,7),0)
median = cv.medianBlur(img,7)
bilateral = cv.bilateralFilter(img,50,100,100)
cv.imshow("average",avg)
cv.imshow("Gaussian",gauss)
cv.imshow("median",median)
cv.imshow("bilateral",bilateral)
cv.waitKey(0)

