# https://bkshin.tistory.com/entry/OpenCV-33-HOG-%EB%94%94%EC%8A%A4%ED%81%AC%EB%A6%BD%ED%84%B0HOG-Descriptor?category=1148027
# HOG-SVM 보행자 검출 (svm_hog_pedestrian)

'''
HOG(Histogram of Oriented Gradient)
HOG : 보행자 검출을 위해 만들어진 특징 디스크립터 
      이미지 경계의 기울기 벡터 크기(magnitude)와 방향(direction)을 히스토그램으로 나타내 계산

https://towardsdatascience.com/hog-histogram-of-oriented-gradients-67ecd887675f
https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
'''

'''
OpenCV - HOG 디스크립터 함수

- descriptor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins): HOG 디스크립터 추출기 생성
    winSize: 윈도 크기, HOG 추출 영역
    blockSize: 블록 크기, 정규화 영역
    blockStride: 정규화 블록 겹침 크기
    cellSize: 셀 크기, 히스토그램 계산 영역
    nbins: 히스토그램 계급 수
    descriptor: HOG 특징 디스크립터 추출기
    hog = descriptor.compute(img): HOG 계산q
    img: 계산 대상 이미지
    hog: HOG 특징 디스크립터 결과

OpenCV는 보행자 인식을 위한 사전 훈련 API 제공 
cv2.HOGDescriptor는 HOG 디스크립터를 계산해 줄 수도 있고, 미리 훈련된 SVM 모델(pretrained SVM model)을 전달받아 보행자를 추출해줄 수도 있음.

- svmdetector = cv2.HOGDescriptor_getDefaultPeopleDetector(): 64 x 128 윈도 크기로 훈련된 모델
- svmdetector = cv2.HOGDescriptor_getDaimlerPeopleDetector(): 48 x 96 윈도 크기로 훈련된 모델

- descriptor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins): HOG 생성
- descriptor.setSVMDetector(svmdetector): 훈련된 SVM 모델 설정
- rects, weights = descriptor.detectMultiScale(img): 객체 검출
    img: 검출하고자 하는 이미지
    rects: 검출된 결과 영역 좌표 N x 4 (x, y, w, h)
    weights: 검출된 결과 계수 N x 1
'''

import os, sys 
import cv2

# default 검출기를 위한 HOG 객체 생성 및 설정--- ①
hogdef = cv2.HOGDescriptor()
hogdef.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# dailer 검출기를 위한 HOG 객체 생성 및 설정--- ②
hogdaim  = cv2.HOGDescriptor((48,96), (16,16), (8,8), (8,8), 9)
hogdaim.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())

#path = '/home/sky/sky/VisualStudio/OpenCV/data2/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data2/'
cap = cv2.VideoCapture(path + 'walking.avi')

mode = False  # 모드 변환을 위한 플래그 변수 
print('Toggle Space-bar to change mode.')

while cap.isOpened():
    ret, img = cap.read()
    if ret :
        if mode:
            # default 검출기로 보행자 검출 --- ③
            found, _ = hogdef.detectMultiScale(img)
            for (x,y,w,h) in found:
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255))
        else:
            # daimler 검출기로 보행자 검출 --- ④
            found, _ = hogdaim.detectMultiScale(img)
            for (x,y,w,h) in found:
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0))
        cv2.putText(img, 'Detector:%s'%('Default' if mode else 'Daimler'), \
                        (10,50 ), cv2.FONT_HERSHEY_DUPLEX,1, (0,255,0),1)
        cv2.imshow('frame', img)
        key = cv2.waitKey(1) 
        if key == 27:
            break
        elif key == ord(' '):
            mode = not mode
    else:
        break
cap.release()
cv2.destroyAllWindows()
