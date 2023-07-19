# https://bkshin.tistory.com/entry/OpenCV-9-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%97%B0%EC%82%B0?category=1148027
# bitwise_and 연산으로 마스킹하기 (bitwise_masking.py)

import os, sys 
import numpy as np, cv2
import matplotlib.pylab as plt

#--① 이미지 읽기
#path = '/home/sky/sky/VisualStudio/OpenCV/data/2'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data2'
img = cv2.imread(os.path.join(path, "wing_wall.jpg"))

#--② 마스크 만들기
mask = np.zeros_like(img)

#img.shape = (480, 640, 3)

center = (int(img.shape[1]/2), (int)(img.shape[0]/2))
cv2.circle(mask, center, 100, (255,255,255), -1)
#cv2.circle(대상이미지, (원점x, 원점y), 반지름, (색상), thickness)

#--③ 마스킹
masked = cv2.bitwise_and(img, mask)

#--④ 결과 출력
cv2.imshow('original', img)
cv2.imshow('mask', mask)
cv2.imshow('masked', masked)
cv2.waitKey()
cv2.destroyAllWindows()

