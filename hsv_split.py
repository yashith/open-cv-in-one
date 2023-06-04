import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.stats as stats
from configparser import ConfigParser

config = ConfigParser()
config.read("E:\OpenCV tests\yolo\configs.ini")
v_path = config['video']['path']
hue_difference = int(config['hue']['hue_difference'])
cap = cv.VideoCapture(v_path)
ret, prev_frame = cap.read()

hue_values=[]
sat_values=[]
lum_values=[]

frame_id = 1
file = open("hsv_frames.txt",'w')
def smooth_array(arr ,method):
    if method == 'n':
        window_size = 3
        window = np.ones(window_size) / window_size
        smoothed_arr = np.convolve(arr, window, 'same')
        return smoothed_arr
    elif method == 'g':
        return ndimage.gaussian_filter(arr, sigma=3)

def get_local_Zscore(arr,window):
    final=np.empty((0,))
    n_parts = len(arr)/window
    np_array = np.array(arr)
    chunks = np.array_split(np_array,n_parts)
    for chunk in chunks:
        z = stats.zscore(chunk)
        final = np.append(final,z)
    
    return final




while True:
    ret, current_frame = cap.read()
    if not ret:
        break
    
    p_hue, p_sat, p_lum = cv.split(cv.cvtColor(prev_frame, cv.COLOR_BGR2HSV))
    hue, sat, lum = cv.split(cv.cvtColor(current_frame, cv.COLOR_BGR2HSV))
    num_pixels: float = float(current_frame.shape[0] * current_frame.shape[1])
    
    diff_hue = np.sum(np.abs(p_hue.astype(np.int32) - hue.astype(np.int32))) / num_pixels
    diff_sat = np.sum(np.abs(p_sat.astype(np.int32) - sat.astype(np.int32))) / num_pixels
    diff_lum = np.sum(np.abs(p_lum.astype(np.int32) - lum.astype(np.int32))) / num_pixels
    
    hue_values.append(diff_hue)
    sat_values.append(diff_sat)
    lum_values.append(diff_lum)
    
    ######
    
    prev_frame = current_frame
    
    key = cv.waitKey(1)
    
    cv.imshow("Video",current_frame)
    if(diff_hue>hue_difference):
        print("frame")
        file.write(str(frame_id))
        file.write("\n")
        key = cv.waitKey(1) #change this to 0 to pause
        if key == ord('p'):
            key = cv.waitKey(10)
    if key == ord('q') or key == 27:
       break
    
    frame_id+=1
file.close()
##
hue_values_np = np.array(hue_values)   
centroid = hue_values_np.mean(axis=0)
#local_z = get_local_Zscore(hue_values,500)
print("diff :",np.subtract(hue_values,centroid))
##   
     
fig, ax = plt.subplots(figsize=(100,5))

# checking for zero zero crossing
hue_diff_smooth = smooth_array(np.diff(hue_values),"g")
zc_arr=[]
for i in range(len(hue_diff_smooth)-2):
    zc = hue_diff_smooth[i] * hue_diff_smooth[i+2] *100
    zc_arr.append(zc)



plt.axhline(y = centroid, color = 'y', linestyle = '-')
ax.plot(hue_values, color='g')
#ax.plot(get_local_Zscore(hue_values,500),color='b')
#ax.plot(smooth_array(np.diff(hue_values),"g"), color='red')
#ax.plot(zc_arr, color='y')

plt.show()