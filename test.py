from configparser import Interpolation
import cv2 as cv

# img = cv.imread("test.jpg")

# def rescale(frame,scale):
#     width=int(frame.shape[1]*scale)
#     height=int(frame.shape[0]*scale)
#     dimention = (width,height)
#     return cv.resize(frame,dimention,interpolation=cv.INTER_AREA)
# cv.imshow("img",rescale(img,0.1))
# cv.waitKey(0)

cap = cv.VideoCapture("Git Tutorial for Beginners- Learn Git in 1 Hour.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        print("can't resolve frame(stream end)")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur=cv.GaussianBlur(gray,(7,7),cv.BORDER_DEFAULT)
    canny = cv.Canny(blur,125,175)
    cv.imshow('frame', canny)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
