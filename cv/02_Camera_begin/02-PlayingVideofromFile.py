# 동영상 게속 재생하기, a frame per 1/FPS

import os
import numpy as np
import cv2 as cv

#path = '/home/sky/sky/VisualStudio/OpenCV/data/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data'

fname = os.path.join(path, 'vtest.avi')
cap = cv.VideoCapture(fname)

# 33ms for a frame
#framerate = 33 

# Set on FPS
FPS = cap.get(cv.CAP_PROP_FPS)
#print("%d" %(FPS))
framerate = round(1000/FPS)

while cap.isOpened():

    frame_pos = cap.get(cv.CAP_PROP_POS_FRAMES)
    frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    print("%d/%d" %(frame_pos, frame_count))

    if(frame_pos == frame_count):
        cap.open(path+'vtest.avi')

    ret, frame = cap.read()
    cv.imshow("Video Frame", frame)
    cv.imshow("Inversed Video Frame", ~frame)

    if cv.waitKey(framerate) > 0: break
    # waitKey(time) : 지정한 시간(time,ms)마다 프레임을 재생
    # 어떠한 키라도 누를 경우 break

cap.release()
cv.destroyAllWindows()
