# https://bkshin.tistory.com/entry/OpenCV-7-%E3%85%87%E3%85%87?category=1148027
# BGR 색상 이미지를 회색조 이미지로 변환 (bgr2gray.py) : 2가지 방법
# 1) 평균값을 이용해 직접 구현, 2) OpenCV에서 제공하는 cv2.cvtcolor() 함수 이용

import os, sys
import cv2
import numpy as np

#path = '/home/sky/sky/VisualStudio/OpenCV/data/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data'
fname = os.path.join(path, "home.jpg")

img = cv2.imread(fname)
if img is None:
    print('Image load failed')
    sys.exit()

# img : astype(속성), ndim(차원), shape(크기), dtype(원소 자료형)

# img.dtype : uint8 -> uint16 변경
img2 = img.astype(np.uint16)                    # dtype 변경 (uint16: 0~2^16-1) ---①
b,g,r = cv2.split(img2)                         # 채널 별로 분리하여 튜플로 반환 ---②
#b,g,r = img2[:,:,0], img2[:,:,1], img2[:,:,2]
gray1 = ((b + g + r)/3).astype(np.uint8)        # 평균 값 연산후 dtype 변경 ---③
#cv2.split() 함수는 비용이 많이 드는 함수로, 가능하다면 Numpy indexing방법을 사용하는 효율적임

gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # BGR을 그레이 스케일로 변경 ---④
cv2.imshow('original', img)
cv2.imshow('gray1', gray1)
cv2.imshow('gray2', gray2)

cv2.waitKey(0)
cv2.destroyAllWindows()
