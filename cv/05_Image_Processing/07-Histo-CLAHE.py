# https://opencv-python.readthedocs.io/en/latest/doc/20.imageHistogramEqualization/imageHistogramEqualization.html
# CLAHE (Contrast Limited Adaptive Histogram Equalization)

'''
히스토그램 평탄화를 하면 이미지의 밝은 부분이 Saturation 되는 현상이 발생함.
CLAHE : 어떤 영역이든 지정된 제한 값(clipLimit 파라미터)을 넘으면 그 픽셀은 다른 영역에 균일하게 배분하여 적용

clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    clipLimit: 대비(Contrast) 제한 경계 값, default=40.0
    tileGridSize: 영역 크기, default=8 x 8
    clahe: 생성된 CLAHE 객체
    clahe.apply(src): CLAHE 적용
    src: 입력 이미지
'''

import os, sys
import cv2
import numpy as np
import matplotlib.pylab as plt

#--① 대상 영상으로 그레이 스케일로 읽기
#path = '/home/sky/sky/VisualStudio/OpenCV/data2/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data2'

img = cv2.imread(os.path.join(path, "bright.jpg"))
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

#--② 밝기 채널에 대해서 이퀄라이즈 적용
img_eq = img_yuv.copy()
img_eq[:,:,0] = cv2.equalizeHist(img_eq[:,:,0])
img_eq = cv2.cvtColor(img_eq, cv2.COLOR_YUV2BGR)

#--③ 밝기 채널에 대해서 CLAHE 적용
img_clahe = img_yuv.copy()
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) #CLAHE 생성
img_clahe[:,:,0] = clahe.apply(img_clahe[:,:,0])           #CLAHE 적용
img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YUV2BGR)

#--④ 결과 출력
cv2.imshow('Before', img)
cv2.imshow('CLAHE', img_clahe)
cv2.imshow('equalizeHist', img_eq)
cv2.waitKey()
cv2.destroyAllWindows()

