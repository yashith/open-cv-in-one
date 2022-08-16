import cv2 as cv
import os

folder_list = ['ciri','triss','yennefer']
base_dir ="H:/Mmmm/Yenna scrapte"
max_width=0
min_width=10000
less_than_80=0

for folder in folder_list:
    folder_path = os.path.join(base_dir,folder)
    folder_path = os.path.join(folder_path,"faces")
    for img in os.listdir(folder_path):
        i=cv.imread(os.path.join(folder_path,img))
        print(img + str(i.shape[0]))
        if(i.shape[0]>max_width):
            max_width=i.shape[0]
        if(i.shape[0]<min_width):
            min_width=i.shape[0]
        if(i.shape[0]<80):
            less_than_80+=1

print("Max width : "+ str(max_width))
print("Min width : "+ str(min_width))
print("Less than 80 : ",less_than_80)