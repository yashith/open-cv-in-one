import cv2

# Read the video file
video_capture = cv2.VideoCapture('V_392.mp4')

# Get the video frame width and height
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize the scene counter and the list of scene boundaries
scene_counter = 0
scene_boundaries = []

# Initialize the previous frame and the current frame
prev_frame = None
curr_frame = None

# Initialize the optical flow object
optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()

# Read the frames of the video one by one
while True:
    # Read the next frame
    ret, frame = video_capture.read()
    
    # Check if the video has ended
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Set the current frame as the previous frame
    prev_frame = curr_frame
    
    # Set the current frame as the current frame
    curr_frame = gray_frame
    
    # If we have both the previous frame and the current frame,
    # compute the optical flow between the two frames
    if prev_frame is not None and curr_frame is not None:
        # Compute the optical flow
        flow = optical_flow.calc(prev_frame, curr_frame, None)
        
        # Compute the sum of the absolute values of the flow
        flow_sum = abs(flow[:,:,0]).sum() + abs(flow[:,:,1]).sum()
        
        # If the flow sum is above a certain threshold,
        # consider this to be a shot boundary
        if flow_sum > 100000:
            # Increment the scene counter and add the frame number
            # to the list of scene boundaries
            scene_counter += 1
            scene_boundaries.append(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
    
    cv2.imshow('FG Mask', frame)
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

# Print the list of scene boundaries
print(scene_boundaries)

# Release the video capture object
video_capture.release()