from asyncio.windows_events import NULL
import cv2 as cv
import mediapipe as mp
import numpy as np

video = cv.VideoCapture("V_1000.mp4")

solution = mp.solutions.pose
pose = solution.Pose()
mpDraw = mp.solutions.drawing_utils

# human detection part
hog = cv.HOGDescriptor()

hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
bg_subtractor = cv.createBackgroundSubtractorMOG2(history=1, varThreshold=10)
#############


while True:
    success, img = video.read()
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    mask = bg_subtractor.apply(img)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    cv.erode(mask,kernel,iterations = 100)
    #cv.bitwise_and(img,mask,img)
    img = mask
    
    
    #img= cv.Canny(img,50,200)
    
    frameTime = 100
    results = NULL
    # if(success):

    #     # remove background

    #     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #     # Human boxex
    #     boxes, weights = boxes, weights = hog.detectMultiScale(
    #         img, winStride=(5, 5))
    #     boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    #     mask = np.zeros(img.shape[:2], dtype="uint8")
    #     for (xA, yA, xB, yB) in boxes:
    #         # display the detected boxes in the colour picture
    #         m = cv.rectangle(mask, (xA, yA), (xB, yB), 255, -1)
    #         onemask = cv.bitwise_and(img, img, mask=m)
    #         cv.imshow("Image-one-Masked", onemask)
    #         results = pose.process(onemask)
    #         # combine all masks
    #         mask = cv.bitwise_and(mask, m)
    #         cv.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)
    #     ####

    #     masked = cv.bitwise_and(img, img, mask=mask)
    #     if(results != NULL):

    #         print(results.pose_landmarks)

    #         if(results.pose_landmarks):
    #             mpDraw.draw_landmarks(
    #                 img, results.pose_landmarks, solution.POSE_CONNECTIONS)

    #         cv.imshow("Image Masked", img)
    #         if cv.waitKey(frameTime) & 0xFF == ord('q'):
    #             break
    # else:
    #     break
    cv.imshow("Image Masked", img)
    if cv.waitKey(frameTime) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()