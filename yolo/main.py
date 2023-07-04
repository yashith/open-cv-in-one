from configparser import ConfigParser
# import hsv_split
# import boundary_analysis
# import replay_with_original_frames

parser = ConfigParser()
parser.read("E:\OpenCV tests\yolo\configs.ini")

# for i in range(1,69):
#     parser.set('video', 'path', f'E:\\OpenCV tests\\Videos\\{i}.mp4')

#     # Writing our configuration file to 'example.ini'
#     with open("E:\OpenCV tests\yolo\configs.ini", 'w') as configfile:
#         parser.write(configfile)
        
#     print("Stage 1 splitting boundaries based on hue values")
exec(open('hsv_split.py').read())
print("Stage 2 Boundary comparision based on objects")
exec(open('boundary_analysis.py').read())
#exec(open('replay_with_original_frames.py').read())
print("splitting into clips")