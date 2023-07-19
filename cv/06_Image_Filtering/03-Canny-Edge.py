# Canny Edge Detector 
'''
캐니 엣지 (Canny Edge)

https://docs.opencv.org/4.5.3/da/d5c/tutorial_canny_detector.html

캐니 엣지는 지금까지 살펴본 것처럼 한 가지 필터만 사용하는 것이 아니라 다음의 4단계 알고리즘에 따라 경계를 검출합니다. 

1. 노이즈 제거: 5 x 5 가우시안 블러링 필터로 노이즈 제거
2. 경계 그레디언트 방향 계산: 소벨 필터로 경계 및 그레디언트 방향 검출
3. 비최대치 억제(Non-Maximum Suppression): 그레디언트 방향에서 검출된 경계 중 가장 큰 값만 선택하고 나머지는 제거
4. Hysteresis 스레시홀딩: 두 개의 경계 값(Max, Min)을 지정해서 경계 영역에 있는 픽셀들 중 큰 경계 값(Max) 밖의 픽셀과 연결성이 없는 픽셀 제거

OpenCV에서 제공하는 캐니 엣지는 함수는 아래와 같습니다.

edges = cv2.Canny(img, threshold1, threshold2, edges, apertureSize, L2gardient)
    img: 입력 영상
    threshold1, threshold2: 이력 스레시홀딩에 사용할 Min, Max 값
    apertureSize: 소벨 마스크에 사용할 커널 크기
    L2gradient: 그레디언트 강도를 구할 방식 (True: 제곱 합의 루트 False: 절댓값의 합)
    edges: 엣지 결과 값을 갖는 2차원 배열
'''

from __future__ import print_function
import os 
import cv2 as cv
import argparse

max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3

#path = '/home/sky/sky/VisualStudio/OpenCV/data/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data/'

img = cv.imread(os.path.join(path, "building.jpg"))

# 케니 엣지 적용 
edges = cv.Canny(img,100,200)

# 결과 출력
cv.imshow('Original', img)
cv.imshow('Canny', edges)
cv.waitKey(0)
cv.destroyAllWindows()


# OpenCV tutorial
# Asks the user to enter a numerical value to set the lower threshold for our Canny Edge Detector (by means of a Trackbar).
# Applies the Canny Detector and generates a mask (bright lines representing the edges on a black background).
# Applies the mask obtained on the original image and display it in a wind

def CannyThreshold(val):
    low_threshold = val
    img_blur = cv.blur(src_gray, (3,3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))
    cv.imshow(window_name, dst)

parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
parser.add_argument('--input', help='Path to input image.', default=path+'building.jpg')
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))

if src is None:
    print('Could not open or find the image: ', args.input)
    exit(0)

src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.namedWindow(window_name)
cv.createTrackbar(title_trackbar, window_name , 0, max_lowThreshold, CannyThreshold)
CannyThreshold(0)
cv.waitKey()


