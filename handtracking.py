from multiprocessing.connection import wait
import cv2 as cv
import mediapipe as mp
import time

cap=cv.VideoCapture("Hands - 38079.mp4")

mpHands= mp.solutions.hands
hands = mpHands.Hands()

while True:
    success,img=cap.read()
    
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = hands.process(img)
    mpDraw = mp.solutions.drawing_utils
    
    if results.multi_hand_landmarks:
        for handMarks in results.multi_hand_landmarks:
            for id,lm in enumerate(handMarks.landmark):
                h,w,c = img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                
                #drow circle only in root
                if(id == 0):
                    cv.circle(img,(cx,cy),25,(255,0,255),cv.FILLED)
            mpDraw.draw_landmarks(img,handMarks,mpHands.HAND_CONNECTIONS)
    pTime = 8
    cTime= time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.imshow("Image",img)
    if cv.waitKey(1) ==ord('q'):
        break