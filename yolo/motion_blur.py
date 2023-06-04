import cv2
import numpy as np
import pywt



def check_blur(frame,threshold):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except:
        return False
    
    
    # Calculate the gradient magnitude
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_x**2 + sobel_y**2)
    avg_mag = np.mean(mag)

    
    # Apply the threshold
    if avg_mag < threshold:
        return True
    else:
        return False
        
def check_blur_wavelet(image, threshold=70):
    # Load the image and convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the wavelet coefficients
    coeffs = pywt.dwt2(gray, 'haar')

    # Calculate the high frequency energy
    energy = np.sum(np.abs(coeffs[1]))

    # Compute the threshold
    h, w = gray.shape
    threshold_energy = (threshold * w * h) / 100.0

    # Check if the image is blurry
    if energy < threshold_energy:
        return True
    else:
        return False
    


# cap = cv2.VideoCapture("../test_3.mp4")

# while True:
    
#     ret,frame = cap.read()
#     print(check_blur_m2(frame))
#     cv2.imshow("frame",frame)
#     cv2.waitKey(1)
#     if not ret:
#         break
    