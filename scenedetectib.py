from scenedetect import SceneManager, open_video, ContentDetector
import cv2 as cv

def find_scenes(video_path, threshold=27.0):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))
    # Detect all scenes in video from current position to end.
    scene_manager.detect_scenes(video)
    # `get_scene_list` returns a list of start/end timecode pairs
    # for each scene that was found.
    return scene_manager.get_scene_list()


scenes = find_scenes("news.mp4")


framecount = 0
s = 0
scene_frames=[]  
    
for i, scene in enumerate(scenes):
    print('Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        i+1,
        scene[0].get_timecode(), scene[0].get_frames(),
        scene[1].get_timecode(), scene[1].get_frames(),))
    scene_frames.append(scene[1].get_frames())

video_capture = cv.VideoCapture('news.mp4')

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    framecount+=1    
    cv.imshow("Video",frame)
    
    key = cv.waitKey(1)
    
    if(framecount == scene_frames[s]):
        print("frame")
        key = cv.waitKey(0)
        if key == 'p':
            key = cv.waitKey(10)
        s+=1
    if key == 'q' or key == 27:
       break
    