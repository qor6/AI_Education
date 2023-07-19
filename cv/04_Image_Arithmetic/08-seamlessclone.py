# https://bkshin.tistory.com/entry/OpenCV-9-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%97%B0%EC%82%B0?category=1148027
# SeamlessClone을 활용한 이미지 합성 (seamlessclone.py)

import os, sys
import cv2
import numpy as np
import matplotlib.pylab as plt

'''
블렌딩에서 알파값 선택과 마스킹을 위한 좌표, 색상 선택에는 많은 시간이 소요됨. 
OpenCV에서는 cv2.seamlessClone()이라는 함수가 있는데 이는 두 이미지의 특징을 살려 알아서 합성함

dst = cv2.seamlessClone(src, dst, mask, coords, flags, output)
    src: 입력 이미지, 일반적으로 전경
    dst: 대상 이미지, 일반적으로 배경
    mask: 마스크, src에서 합성하고자 하는 영역은 255, 나머지는 0
    coords: src가 놓이기 원하는 dst의 좌표 (중앙)
    flags: 합성 방식
    output(optional): 합성 결과

flags는 입력 원본을 유지하는 cv2.NORMAL_CLONE과 입력과 대상을 혼합하는 cv2.MIXED_CLONE이 있다
'''

#--① 합성 대상 영상 읽기
#path = '/home/sky/sky/VisualStudio/OpenCV/data2/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data2'
img1 = cv2.imread(os.path.join(path, "drawing.jpg"))    # (162, 200, 3)
img2 = cv2.imread(os.path.join(path, "my_hand.jpg"))    # (450, 600, 3)

#--② 마스크 생성, 합성할 이미지 전체 영역을 255로 셋팅
mask = np.full_like(img1, 255)

#--③ 합성 대상 좌표 계산(img2의 중앙)
height, width = img2.shape[:2]
center = (width//2, height//2)
 
#--④ seamlessClone 으로 합성 
normal = cv2.seamlessClone(img1, img2, mask, center, cv2.NORMAL_CLONE)
mixed = cv2.seamlessClone(img1, img2, mask, center, cv2.MIXED_CLONE)

#--⑤ 결과 출력
cv2.imshow('foregroud', img1)
cv2.imshow('background', img2)
cv2.imshow('mask', mask)
cv2.imshow('normal', normal)
cv2.imshow('mixed', mixed)
cv2.waitKey()
cv2.destroyAllWindows()

