import cv2

# Load the video file
video = cv2.VideoCapture("E:\OpenCV tests\V_392.mp4")

# Set up the background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Set up the fast phase and slow phase detectors
fast_phase_detector = cv2.FastPhaseHumanMovementDetector()
slow_phase_detector = cv2.SlowPhaseHumanMovementDetector()

# Set up the pause flag
paused = False

# Iterate over each frame of the video
while True:
    # Read the next frame from the video
    _, frame = video.read()

    # Check if we have reached the end of the video
    if frame is None:
        break
    
    # Apply the background subtractor to the frame
    fg_mask = bg_subtractor.apply(frame)
    
    # Detect fast and slow phase human movement in the frame
    fast_phase_movement = fast_phase_detector.detect(fg_mask)
    slow_phase_movement = slow_phase_detector.detect(fg_mask)
    
    # Check if there is a scene boundary
    if fast_phase_movement or slow_phase_movement:
        # Set the pause flag to True
        paused = True
    elif paused:
        # If the pause flag is True, set it back to False
        paused = False
    
    # Check if the video is paused
    if paused:
        # Display the frame but don't update the window
        cv2.imshow("Video", frame)
        # Wait for the user to press a key
        cv2.waitKey(0)
    else:
        # Display the frame and update the window
        cv2.imshow("Video", frame)
        # Wait for a short period of time before displaying the next frame
        cv2.waitKey(25)

# Release the video file and destroy all windows
video.release()
cv2.destroyAllWindows()