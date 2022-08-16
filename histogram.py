import cv2 as cv
import matplotlib.pyplot as plt

i = cv.imread("test.jpg")
img = cv.resize(i,(int(i.shape[0]*0.5),int(i.shape[1]*0.5)),interpolation=cv.INTER_AREA)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray)

gray_hist = cv.calcHist([gray],[0],None,[256],[0,256])

plt.figure()
plt.title("Gray scale histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixels")
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()
cv.waitKey(0)