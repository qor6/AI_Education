# https://bkshin.tistory.com/entry/OpenCV-11-2%EC%B0%A8%EC%9B%90-%ED%9E%88%EC%8A%A4%ED%86%A0%EA%B7%B8%EB%9E%A8%EA%B3%BC-%EC%97%AD%ED%88%AC%EC%98%81back-project?category=1148027
# 2D Histogram & Back Projeciton

'''
2D 히스토그램 (histo_2d.py)

Histogram은 1차원으로 grayscale 이미지의 pixel의 강도, 즉 빛의 세기를 분석한 결과이다. 
2D Histogrm은 축이 2개이고, 각 축이 만나는 지점의 개수를 표현

Color 이미지의 Hue(색상) & Saturation(채도)을 동시에 분석하는 방법 예시
image_hsv일 때
calcHist([image, ][channel, ]mask[, bins][, range])
    Histogram 분석 함수
    Parameters:	
        image – HSV로 변환된 이미지
        channel – 0-> Hue, 1-> Saturation
        bins – bin의 개수 [180,256] 첫번째는 Hue, 두번째는 Saturation
        range – [0,180,0,256] : Hue(0~180), Saturation(0,256)
'''

import os, sys
import cv2
import numpy as np
import matplotlib.pylab as plt

#path = '/home/sky/sky/VisualStudio/OpenCV/data2/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data2'
img = cv2.imread(os.path.join(path, "mountain.jpg"))

# --①컬러 스타일을 1.x 스타일로 사용
plt.style.use('classic')            

# 0 : Blue, 1 : Green
plt.subplot(131)
hist = cv2.calcHist([img], [0,1], None, [32,32], [0,256,0,256]) #--②
p = plt.imshow(hist)                                            #--③
plt.title('Blue and Green')                                     #--④
plt.colorbar(p)                                                 #--⑤

# 1 : Green, 1 : Red
plt.subplot(132)
hist = cv2.calcHist([img], [1,2], None, [32,32], [0,256,0,256]) #--⑥
p = plt.imshow(hist)
plt.title('Green and Red')
plt.colorbar(p)

# 0 : Blue, 2 : Red
plt.subplot(133)
hist = cv2.calcHist([img], [0,2], None, [32,32], [0,256,0,256]) #--⑦
p = plt.imshow(hist)
plt.title('Blue and Red')
plt.colorbar(p)

plt.show()

