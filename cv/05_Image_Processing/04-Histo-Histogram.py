# https://bkshin.tistory.com/entry/OpenCV-10-%ED%9E%88%EC%8A%A4%ED%86%A0%EA%B7%B8%EB%9E%A8?category=1148027
# Histogram

'''
Histogram : 이미지의 밝기의 분포를 그래프로 표현 
히스토그램을 이용하면 이미지의 전체 밝기 분포와 채도(색의 밝고 어두움)를 알 수 있다.

cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
    image – 분석대상 이미지(uint8 or float32 type). Array형태.
    channels – 분석 채널(X축의 대상). 이미지가 graysacle이면 [0], color 이미지이면 [0],[0,1] 형태(1 : Blue, 2: Green, 3: Red)
    mask – 이미지의 분석영역. None이면 전체 영역.
    histSize – BINS 값. [256]
    ranges – Range값. [0,256]
'''

import os, sys
import cv2
import numpy as np
import matplotlib.pylab as plt

#--① 이미지 읽기 및 출력
#path = '/home/sky/sky/VisualStudio/OpenCV/data2/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data2'
img = cv2.imread(os.path.join(path, "mountain.jpg"))

# 그레이 이미지 히스토그램
#--② Grayscale 히스토그램 계산 및 그리기
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray], [0], None, [256], [0,256])
cv2.imshow('gray', gray)
plt.plot(hist)
print("hist.shape:", hist.shape)  #--③ 히스토그램의 shape (256,1)
print("hist.sum():", hist.sum(), "img.shape:",img.shape) #--④ 히스토그램 총 합계와 이미지의 크기
plt.show()

# 컬러 이미지 히스토그램
#--② 히스토그램 계산 및 그리기
cv2.imshow('img', img)
channels = cv2.split(img)
colors = ('b', 'g', 'r')
for (ch, color) in zip(channels, colors):
    hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
    plt.plot(hist, color = color)
plt.show()

