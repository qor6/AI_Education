# https://bkshin.tistory.com/entry/OpenCV-10-%ED%9E%88%EC%8A%A4%ED%86%A0%EA%B7%B8%EB%9E%A8?category=1148027
# 히스토그램 정규화 (histo_normalize.py)

'''
화소 분포가 특정 영역에 몰려 있는 경우 화질 개선
dst = cv2.normalize(src, dst, alpha, beta, type_flag)
    src: 정규화 이전의 데이터
    dst: 정규화 이후의 데이터
    alpha: 정규화 구간 1
    beta: 정규화 구간 2, 구간 정규화가 아닌 경우 사용 안 함
    type_flag: 정규화 알고리즘 선택 플래그 상수
        - cv2.NORM_MINMAX : alpha와 beta 구간으로 정규화  
        - cv2.NORM_L1 : 전체 합으로 나누기 
        - cv2.NORM_L2 :단위 벡터로 정규화 
        - cv2.NORM_INF : 최댓값으로 나누기 
'''

import os, sys 
import cv2
import numpy as np
import matplotlib.pylab as plt

#--① 그레이 스케일로 영상 읽기
#path = '/home/sky/sky/VisualStudio/OpenCV/data2/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data2'
img = cv2.imread(os.path.join(path, "abnormal.jpg"), cv2.IMREAD_GRAYSCALE)

#--② 직접 연산한 정규화
img_f = img.astype(np.float32)
img_norm = ((img_f - img_f.min()) * (255) / (img_f.max() - img_f.min()))
img_norm = img_norm.astype(np.uint8)

#--③ OpenCV API를 이용한 정규화
img_norm2 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

#--④ 히스토그램 계산
hist = cv2.calcHist([img], [0], None, [256], [0, 255])
hist_norm = cv2.calcHist([img_norm], [0], None, [256], [0, 255])
hist_norm2 = cv2.calcHist([img_norm2], [0], None, [256], [0, 255])

cv2.imshow('Before', img)
cv2.imshow('Manual', img_norm)
cv2.imshow('cv2.normalize()', img_norm2)

hists = {'Before' : hist, 'Manual':hist_norm, 'cv2.normalize()':hist_norm2}
for i, (k, v) in enumerate(hists.items()):
    plt.subplot(1,3,i+1)
    plt.title(k)
    plt.plot(v)
plt.show()

