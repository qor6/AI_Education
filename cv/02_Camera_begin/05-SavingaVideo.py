import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'DIVX')
# Fourcc (4-문자 코드, four character code), DIVX, XVID, FMP4 (FFMPEG MPEG-4), X264, MJPG

out = cv.VideoWriter('./output.avi', fourcc, 20.0, (640,  480))

'''
cv2.VideoWriter(filename, fourcc, fps, frameSize, isColor=None) -> retval
• filename: 비디오 파일 이름 (e.g. 'video.mp4')
• fourcc: fourcc (e.g. cv2.VideoWriter_fourcc(*'DIVX'))
• fps: 초당 프레임 수 (e.g. 30)
• frameSize: 프레임 크기. (width, height) 튜플.
• isColor: 컬러 영상이면 True, 그렇지않으면 False.
• retval: cv2.VideoWriter 객체
'''

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv.flip(frame, 1)
    # write the flipped frame, 0:상하반전, 1:좌우반전, -1;상하좌우반전
    out.write(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()

