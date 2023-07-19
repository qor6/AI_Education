# 동영상 1회 재생하기, a frame per 1ms

import os 
import numpy as np
import cv2 as cv

path = '/home/sky/sky/VisualStudio/OpenCV/data/'
fname = os.path.join(path, 'vtest.avi')
cap = cv.VideoCapture(fname)

while cap.isOpened():
    ret, frame = cap.read()

    frame_pos = cap.get(cv.CAP_PROP_POS_FRAMES)
    frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    print("%d/%d" %(frame_pos, frame_count))

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Step 1 : Display Grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame_gray', gray)
    #cv.imshow('frame_color', frame)

    # Step 2 : Original frame + Grayscale frame
    gray_ = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    merged = np.hstack((frame, gray_))
    cv.imshow('Original + Gray frame', merged)

    # Step 3 : 라플라시안 필터 적용
    edge = cv.Laplacian(frame, -1)
    laplacian = np.hstack((frame, edge))
    cv.imshow('Laplacian filter', laplacian)

    # waitKey(1) : 1ms per a frame
    #if cv.waitKey(1) > 0: break
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

