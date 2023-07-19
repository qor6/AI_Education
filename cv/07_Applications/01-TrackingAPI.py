# https://bkshin.tistory.com/entry/OpenCV-32-%EA%B0%9D%EC%B2%B4-%EC%B6%94%EC%A0%81%EC%9D%84-%EC%9C%84%ED%95%9C-Tracking-API?category=1148027
# Tracker APIs

import os, sys 
import cv2

#path = '/home/sky/sky/VisualStudio/OpenCV/data2/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data2/'

# https://docs.opencv.org/4.3.0/d0/d0a/classcv_1_1Tracker.html
# 트랙커 객체 생성자 함수 리스트 ---①
trackers = [cv2.TrackerBoosting_create,     # AdaBoost 알고리즘 기반
            cv2.TrackerMIL_create,          # MIL(Multiple Instance Learning) 알고리즘
            cv2.TrackerKCF_create,          # KCF(Kernelized Correlation Filters) 알고리즘 
            cv2.TrackerTLD_create,          # TLD(Tracking, Learning and Detection) 알고리즘 
            cv2.TrackerMedianFlow_create,   # 객체의 전방향/역방향을 추적해서 불일치성을 측정
            cv2.TrackerGOTURN_create,       # CNN(Convolutional Neural Networks) 기반, 버그로 오류 발생 (OpenCV 3.4)
            cv2.TrackerCSRT_create,         # CSRT(Channel and Spatial Reliability)
            cv2.TrackerMOSSE_create]        # MOSSE(Minimum Output Sum of Squared Error) tracker, 내부적으로 grayscale 사용

trackerIdx = 1  # 트랙커 생성자 함수 선택 인덱스
tracker = None
isFirst = True

video_src = 0 # 비디오 파일과 카메라 선택 ---②
video_src = path+"highway.mp4"

cap = cv2.VideoCapture(video_src)
fps = cap.get(cv2.CAP_PROP_FPS) # 프레임 수 구하기
delay = int(1000/fps)
print('delay=', delay)
win_name = 'Tracking APIs'

'''
retval = cv2.Tracker.init(img, boundingBox): Tracker 초기화
    img: 입력 영상
    boundingBox: 추적 대상 객체가 있는 좌표 (x, y)

초기화 후 새로운 영상 프레임에서 추적 대상 객체의 위치를 찾기 위해 update() 함수를 호출해야 한다.
retval, boundingBox = cv2.Tracker.update(img): 새로운 프레임에서 추적 대상 객체 위치 찾기
    img: 새로운 프레임 영상
    retval: 추적 성공 여부
    boundingBox: 새로운 프레임에서의 추적 대상 객체의 새로운 위치 (x, y, w, h)
'''

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('Cannot read video file')
        break
    img_draw = frame.copy()

    if tracker is None:                     # 트랙커 생성 안된 경우
        cv2.putText(img_draw, "Press the Space to set ROI!!", \
            (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)
    else:
        ok, bbox = tracker.update(frame)    # 새로운 프레임에서 추적 위치 찾기 ---③
        (x,y,w,h) = bbox
        if ok: # 추적 성공
            cv2.rectangle(img_draw, (int(x), int(y)), (int(x + w), int(y + h)), \
                          (0,255,0), 2, 1)
        else : # 추적 실패
            cv2.putText(img_draw, "Tracking fail.", (100,80), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)
    trackerName = tracker.__class__.__name__
    cv2.putText(img_draw, str(trackerIdx) + ":"+trackerName , (100,20), \
                 cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0),2,cv2.LINE_AA)

    cv2.imshow(win_name, img_draw)
    key = cv2.waitKey(delay) & 0xff

    # 스페이스 바 또는 비디오 파일 최초 실행 ---④
    if key == ord(' ') or (video_src != 0 and isFirst): 
        isFirst = False
        roi = cv2.selectROI(win_name, frame, False) # 초기 객체 위치 설정
        if roi[2] and roi[3]:                       # 위치 설정 값 있는 경우
            tracker = trackers[trackerIdx]()        # 트랙커 객체 생성 ---⑤
            isInit = tracker.init(frame, roi)
    elif key in range(48, 56):                      # 0~7 숫자 입력   ---⑥
        trackerIdx = key-48                         # 선택한 숫자로 트랙커 인덱스 수정
        if bbox is not None:
            tracker = trackers[trackerIdx]()        # 선택한 숫자의 트랙커 객체 생성 ---⑦
            isInit = tracker.init(frame, bbox)      # 이전 추적 위치로 추적 위치 초기화
    elif key == 27 : 
        break
else:
    print( "Could not open video")

cap.release()
cv2.destroyAllWindows()

