import cv2 as cv
import numpy as np


def get_hue_diff(prev_frame,current_frame):
    p_hue, p_sat, p_lum = cv.split(cv.cvtColor(prev_frame, cv.COLOR_BGR2HSV))
    hue, sat, lum = cv.split(cv.cvtColor(current_frame, cv.COLOR_BGR2HSV))
    num_pixels: float = float(current_frame.shape[0] * current_frame.shape[1])
    
    diff_hue = np.sum(np.abs(p_hue.astype(np.int32) - hue.astype(np.int32))) / num_pixels
    diff_sat = np.sum(np.abs(p_sat.astype(np.int32) - sat.astype(np.int32))) / num_pixels
    diff_lum = np.sum(np.abs(p_lum.astype(np.int32) - lum.astype(np.int32))) / num_pixels
    
    return(diff_hue,diff_sat,diff_lum)

def diff_hist(prev_frame,current_frame):
    # Convert the images to the HSV color space
    hsv1 = cv.cvtColor(prev_frame, cv.COLOR_BGR2HSV)
    hsv2 = cv.cvtColor(current_frame, cv.COLOR_BGR2HSV)

    # Calculate the hue histograms
    hist1 = cv.calcHist([hsv1], [0], None, [180], [0, 180])
    hist2 = cv.calcHist([hsv2], [0], None, [180], [0, 180])

    # Normalize the histograms
    cv.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    cv.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    # Compare the histograms using the Bhattacharyya distance
    distance = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    return distance