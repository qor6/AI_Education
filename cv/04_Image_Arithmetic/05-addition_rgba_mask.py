# 비트연산
# https://opencv-python.readthedocs.io/en/latest/doc/07.imageArithmetic/imageArithmetic.html

import os, sys
import cv2
import numpy as np

#path = '/home/sky/sky/VisualStudio/OpenCV/data/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data'

img1 = cv2.imread(os.path.join(path, "logo.png"))
img2 = cv2.imread(os.path.join(path, "lena.png"))

# 삽입할 이미지의 row, col, channel정보
print(img1.shape, img2.shape)   # (222, 180, 3), (512, 512, 3)
rows, cols, channels = img1.shape

# 대상 이미지에서 삽입할 이미지의 영역을 추출
roi = img2[0:rows, 0:cols]

#mask를 만들기 위해서 img1을 gray로 변경후 binary image로 전환
#mask는 logo부분이 흰색(255), 바탕은 검은색(0)
#mask_inv는 logo부분이 검은색(0), 바탕은 흰색(255)

img2gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

#bitwise_and 연산자는 둘다 0이 아닌 경우만 값을 통과 시킴.
#즉 mask가 검정색이 아닌 경우만 통과가 되기때문에 mask영역 이외는 모두 제거됨.
#아래 img1_fg의 경우는 bg가 제거 되고 fg(logo부분)만 남게 됨.
#img2_bg는 roi영역에서 logo부분이 제거되고 bg만 남게 됨.
img1_fg = cv2.bitwise_and(img1, img1, mask=mask)
img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

#2개의 이미지를 합치면 바탕은 제거되고 logo부분만 합쳐짐.
dst = cv2.add(img1_fg, img2_bg)

#합쳐진 이미지를 원본 이미지에 추가.
img2[0:rows, 0:cols] = dst

cv2.imshow('res', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# https://bkshin.tistory.com/entry/OpenCV-9-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%97%B0%EC%82%B0?category=1148027
#--① 합성에 사용할 영상 읽기, 전경 영상은 4채널 png 파일
#path = '/home/sky/sky/VisualStudio/OpenCV/data2/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data2'
img_fg = cv2.imread(os.path.join(path, "opencv_logo.png"),cv2.IMREAD_UNCHANGED)
img_bg = cv2.imread(os.path.join(path, "girl.jpg"))

#print(img_fg.shape) # (h,w,c)=(120, 98, 4)
#opencv_log.png -> img_fg[;,;,3] -> alpha channel (투명)

#--② 알파채널을 이용해서 마스크와 역마스크 생성
_, mask = cv2.threshold(img_fg[:,:,3], 1, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
# mask : 전경 1, 배경 0,    mask_inv : 전경 0, 배경 1

#--③ 전경 영상 크기로 배경 영상에서 ROI 잘라내기
img_fg = cv2.cvtColor(img_fg, cv2.COLOR_BGRA2BGR)
h, w = img_fg.shape[:2]         # (120, 98)
roi = img_bg[10:10+h, 10:10+w ] 

#--④ 마스크 이용해서 오려내기
masked_fg = cv2.bitwise_and(img_fg, img_fg, mask=mask)
masked_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

#--⑥ 이미지 합성
added = masked_fg + masked_bg
img_bg[10:10+h, 10:10+w] = added

cv2.imshow('mask', mask)
cv2.imshow('mask_inv', mask_inv)
cv2.imshow('masked_fg', masked_fg)
cv2.imshow('masked_bg', masked_bg)
cv2.imshow('added', added)
cv2.imshow('result', img_bg)
cv2.waitKey()
cv2.destroyAllWindows()