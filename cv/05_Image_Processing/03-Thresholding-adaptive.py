# https://bkshin.tistory.com/entry/OpenCV-8-%EC%8A%A4%EB%A0%88%EC%8B%9C%ED%99%80%EB%94%A9Thresholding?category=1148027
# 적응형 스레시홀딩 적용 (threshold_adapted.py)

'''
적응형 스레시홀딩(Adaptive Thresholding)
원본 이미지에서 조명이 일정하지 않거나 배경색이 여러 개인 경우에는 하나의 임계값으로 선명한 바이너리 이미지를 만들어내기 어려움. 
이미지를 여러 영역으로 나눈 뒤, 그 주변 픽셀 값을 활용하여 임계값을 구함

cv2.adaptiveThreshold(img, value, method, type_flag, block_size, C)
    img: 입력영상
    value: 임계값을 만족하는 픽셀에 적용할 값
    method: 임계값 결정 방법
    type_flag: 스레시홀딩 적용 방법 (cv2.threshod()와 동일)
    block_size: 영역으로 나눌 이웃의 크기(n x n), 홀수
    C: 계산된 임계값 결과에서 가감할 상수(음수 가능)

    method 값
        cv2.ADAPTIVE_THRESH_MEAN_C: 이웃 픽셀의 평균으로 결정
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C: 가우시안 분포에 따른 가중치의 합으로 결정
'''

import os, sys 
import cv2
import numpy as np
import matplotlib.pylab as plt

# 이미지를 그레이 스케일로 읽기
#path = '/home/sky/sky/VisualStudio/OpenCV/data2/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data2'

img = cv2.imread(os.path.join(path, "sudoku.png"), cv2.IMREAD_GRAYSCALE)

blk_size = 9        # 블럭 사이즈
C = 5               # 차감 상수 

# ---① 오츠의 알고리즘으로 단일 경계 값을 전체 이미지에 적용
ret, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# ---② Adaptive threshold를 평균과 가우시안 분포로 각각 적용
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                    cv2.THRESH_BINARY, blk_size, C)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                     cv2.THRESH_BINARY, blk_size, C)

# ---③ 결과를 Matplot으로 출력
imgs = {'Original': img, 'Global-Otsu:%d'%ret:th1, \
        'Adapted-Mean':th2, 'Adapted-Gaussian': th3}
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(2,2,i+1)
    plt.title(k)
    plt.imshow(v,'gray')
    plt.xticks([]),plt.yticks([])

plt.show()

