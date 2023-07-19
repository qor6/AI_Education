# https://bkshin.tistory.com/entry/OpenCV-6-dd?category=1148027
# 관심영역 표시 (roi.py)

import os, sys
import cv2
import numpy as np

#path = '/home/sky/sky/VisualStudio/OpenCV/data/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data2'

fname = os.path.join(path, "sunset.jpg")

# cv2.imread() 함수를 실행하면 이미지를 numpy 배열로 반환
img = cv2.imread(fname)
if img is None:
    print('Image load failed')
    sys.exit()

x,y,w,h = 320,150,50,50         # roi 좌표
roi = img[y:y+h, x:x+w]         # roi 지정      ---①
img2 = roi.copy()               # roi 배열 복제

print(roi.shape)                # roi shape, (50,50,3)
cv2.rectangle(roi, (0,0), (h-1, w-1), (0,255,0)) # roi 전체에 사각형 그리기 ---②
# format : cv2.rectangle(roi, (x, y), (x+w, y+h), (0,255,0))
cv2.imshow("img", img)
cv2.imshow("roi", img2)         # roi 만 따로 출력

key = cv2.waitKey(0)
print(key)
cv2.destroyAllWindows()
