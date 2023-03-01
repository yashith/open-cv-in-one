import cv2
import numpy as np

cap = cv2.VideoCapture("V_340.mp4")

# Read the input images and convert them to grayscale
re,image1 = cap.read()
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

while True:
    re,image2 = cap.read()
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Create a DualTVL1OpticalFlow object and set the necessary parameters
    flow = cv2.optflow.DualTVL1OpticalFlow_create()
    flow.setScalesNumber(5)
    flow.setWarpingsNumber(5)
    flow.setEpsilon(0.01)
    flow.setInnerIterations(2)
    flow.setOuterIterations(7)
    flow.setScaleStep(0.8)
    flow.setGamma(0.1)
    flow.setMedianFiltering(5)

    # Calculate the optical flow
    flow_vectors = flow.calc(gray1, gray2, None)

    # Create a blank image to draw the quiver plot on
    output_image = np.zeros_like(image1)

    # Get the width and height of the image
    height, width = image1.shape[:2]

    # Divide the image into a grid of 10x10 cells
    cell_size = 10
    grid_x = width // cell_size
    grid_y = height // cell_size

    # Iterate over the grid cells
    for i in range(grid_y):
        for j in range(grid_x):
            # Get the cell bounds
            x1 = j * cell_size
            y1 = i * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            # Get the flow vectors for the cell
            fx, fy = flow_vectors[y1:y2, x1:x2].mean(axis=0).mean(axis=0)

            # Calculate the magnitude and angle of the flow vector
            magnitude = np.sqrt(fx**2 + fy**2)
            angle = np.arctan2(fy, fx)

            # Convert the angle to degrees and map it to an RGB color value
            angle_degrees = angle * 180 / np.pi
            r = (angle_degrees + 180) % 180
            g = (angle_degrees + 90) % 180
            b = (angle_degrees + 270) % 180

            # Draw a rectangle on the output image with the color
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (r, g, b), -1)

# Show the output image with the color gradient
    cv2.imshow("Optical Flow", output_image)
    gray1 = gray2
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
