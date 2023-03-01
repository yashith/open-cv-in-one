from asyncio.windows_events import NULL
import cv2 as cv
import mediapipe as mp
import numpy as np
import cv2

# Capture the video
cap = cv2.VideoCapture("V_340.mp4")

# Read the first frame
_, first_frame = cap.read()

# Convert the frame to grayscale
gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)

# Create a background model by taking the average of several frames
background = gray.astype('float')
for i in range(100):
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.accumulateWeighted(gray, background, 0.5)


frameTime = 100
# Loop through the video
while True:
    # Read a frame from the video
    _, frame = cap.read()
    if frame is None:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Subtract the background model from the frame to create a foreground mask
    foreground_mask = cv2.absdiff(background.astype('uint8'), gray)

    # Apply morphological transformations to the foreground mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foreground_mask = cv2.erode(foreground_mask, kernel, iterations=1)
    foreground_mask = cv2.dilate(foreground_mask, kernel, iterations=1)

    # Apply contour detection to the foreground mask
    contours, hierarchy = cv2.findContours(
        foreground_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Loop through the contours and draw them on the frame
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow("Image Masked", frame)
    if cv.waitKey(frameTime) & 0xFF == ord('q'):
        break
