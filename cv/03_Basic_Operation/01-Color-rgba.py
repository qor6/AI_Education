# BGR, BGRA, Ahlpha 채널 (rgba.py)
# Color -> RGB, But OpenCV BGR

# https://bkshin.tistory.com/entry/OpenCV-7-%E3%85%87%E3%85%87?category=1148027

import os, sys
import cv2

#path = '/home/sky/sky/VisualStudio/OpenCV/data/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data'

fname = os.path.join(path, "opencv-logo.png")

# 기본 값 옵션
img = cv2.imread(fname)   
if img is None:
    print('Image load failed')
    sys.exit()

# IMREAD_COLOR 옵션                   
bgr = cv2.imread(fname, cv2.IMREAD_COLOR)    
# IMREAD_UNCHANGED 옵션
bgra = cv2.imread(fname, cv2.IMREAD_UNCHANGED) 
# 각 옵션에 따른 이미지 shape
print("default", img.shape, "color", bgr.shape, "unchanged", bgra.shape) 

cv2.imshow('bgr', bgr)
cv2.imshow('bgra', bgra)
cv2.imshow('alpha', bgra[:,:,0])  # 알파 채널만 표시
cv2.waitKey(0)
cv2.destroyAllWindows()
