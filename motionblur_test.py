import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture('test_3.mp4')

# Define the threshold for blur detection
threshold = 15

# Loop through each frame in the video
while cap.isOpened():
    # Read the frame
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate the gradient magnitude
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_x**2 + sobel_y**2)
    avg_mag = np.mean(mag)
    
    
    key=cv2.waitKey(1)
    # Apply the threshold
    if avg_mag < threshold:
        key=cv2.waitKey(0)
    
    if(key==ord('p')):
        key=cv2.waitKey(1)
    elif(key==ord('q')):
        break
    
    cv2.imshow("Mb",frame)
    # Check for end of video
    if not ret:
        break
    
# Release the video capture
cap.release()
