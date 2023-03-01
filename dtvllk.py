import cv2
import matplotlib.pyplot as plt
import numpy as np
# Read the video file
video_capture = cv2.VideoCapture('V_340.mp4')

# Get the video frame width and height
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize the scene counter and the list of scene boundaries
scene_counter = 0
frame_counter =0
scene_boundaries = []
flow_sum_list=[]


# Initialize the previous frame and the current frame
prev_frame = None
curr_frame = None

# Initialize the optical flow object
optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()

# Read the frames of the video one by one
while True:
    # Read the next frame
    ret, frame = video_capture.read()
    frame_counter+=1
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
        #flow = optical_flow.calc(prev_frame, curr_frame, None)
        flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Compute the sum of the absolute values of the flow
        flow_sum = abs(flow[:,:,0]).sum() + abs(flow[:,:,1]).sum()
        
        # If the flow sum is above a certain threshold,
        # consider this to be a shot boundary
        print(flow_sum)
        flow_sum_list.append(flow_sum)
        if flow_sum > 100000:
            # Increment the scene counter and add the frame number
            # to the list of scene boundaries
            scene_counter += 1
            scene_boundaries.append(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
    
    cv2.imshow('FG Mask', frame)
    if frame_counter == 23:
        keyboard = cv2.waitKey(0)
    keyboard = cv2.waitKey(1)
    if keyboard == 'q' or keyboard == 27:
        break

# Print the list of scene boundaries
print(scene_boundaries)
print("Scene count", scene_counter)
print("Frames", frame_counter)

#histogram
max_fs = (max(flow_sum_list))
max_fs_index = flow_sum_list.index(max_fs)
min_fs = (min(flow_sum_list))
min_fs_index = flow_sum_list.index(min_fs)
norm_fs_list = norm = [float(i)/max(flow_sum_list) for i in flow_sum_list]
print("maxframe", max_fs_index)
print("minframe"),(min_fs_index)


plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.plot(norm_fs_list)
# Release the video capture object
video_capture.release()