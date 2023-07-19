# https://bkshin.tistory.com/entry/OpenCV-8-%EC%8A%A4%EB%A0%88%EC%8B%9C%ED%99%80%EB%94%A9Thresholding?category=1148027
# 전역 스레시홀딩 (threshold.py)

'''
cv2.threshold() 함수 사용법
ret, out = cv2.threshold(img, threshold, value, type_flag)
    img: 변환할 이미지
    threshold: 스레시홀딩 임계값
    value: 임계값 기준에 만족하는 픽셀에 적용할 값
    type_flag: 스레시홀딩 적용 방법

    ret : 스레시홀딩에 사용한 임계값 (파라미터 threshold와 동일 또는 임계값 -1일 때 type_flag에 적응적으로 계산한 값)
    out : 스레시홀딩이 적용된 바이너리 이미지 

    type_flag 값은 다음과 같습니다.
        cv2.THRESH_BINARY: 픽셀 값이 임계값을 넘으면 value로 지정하고, 넘지 못하면 0으로 지정
        cv2.THRESH_BINARY_INV: cv.THRESH_BINARY의 반대
        cv2.THRESH_TRUNC: 픽셀 값이 임계값을 넘으면 value로 지정하고, 넘지 못하면 원래 값 유지
        cv2.THRESH_TOZERO: 픽셀 값이 임계값을 넘으면 원래 값 유지, 넘지 못하면 0으로 지정
        cv2.THRESH_TOZERO_INV: cv2.THRESH_TOZERO의 반대
'''

import os, sys
import cv2
import numpy as np
import matplotlib.pylab as plt

#이미지를 그레이 스케일로 읽기
#path = '/home/sky/sky/VisualStudio/OpenCV/data2/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data2'
img = cv2.imread(os.path.join(path, "gray_gradient.jpg"), cv2.IMREAD_GRAYSCALE)

# --- ① NumPy API로 바이너리 이미지 만들기
thresh_np = np.zeros_like(img)   # 원본과 동일한 크기의 0으로 채워진 이미지
thresh_np[ img > 127] = 255      # 127 보다 큰 값만 255로 변경

# ---② OpenCV API로 바이너리 이미지 만들기
ret, thresh_cv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 
print(ret)  # 127.0, 바이너리 이미지에 사용된 문턱 값 반환

# ---③ 원본과 결과물을 matplotlib으로 출력
imgs = {'Original': img, 'NumPy API':thresh_np, 'cv2.threshold': thresh_cv}
for i, (key, value) in enumerate(imgs.items()):
    plt.subplot(1, 3, i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]); plt.yticks([])

plt.show()


### 다양한 type_flag 사용
_, t_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
_, t_bininv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
_, t_truc = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
_, t_2zr = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
_, t_2zrinv = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

imgs = {'origin':img, 'BINARY':t_bin, 'BINARY_INV':t_bininv, \
        'TRUNC':t_truc, 'TOZERO':t_2zr, 'TOZERO_INV':t_2zrinv}
for i, (key, value) in enumerate(imgs.items()):
    plt.subplot(2,3, i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]);    plt.yticks([])
    
plt.show()

