# https://bkshin.tistory.com/entry/OpenCV-21-%EB%B8%94%EB%9F%AC%EB%A7%81%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%9C-%EB%AA%A8%EC%9E%90%EC%9D%B4%ED%81%AC-%EC%B2%98%EB%A6%AC-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%8A%A4%EC%BC%80%EC%B9%98-%ED%9A%A8%EA%B3%BC-%EC%A0%81%EC%9A%A9%ED%95%98%EA%B8%B0?category=1148027
# 카메라 스케치 효과 (Sketch camera)

import os, sys
import cv2
import numpy as np
import matplotlib.pylab as plt

cap = cv2.VideoCapture(0)   

while cap.isOpened():
    ret, frame = cap.read()

    # 속도 향상을 위해 영상 크기를 1/2으로 축소
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, \
                        interpolation=cv2.INTER_AREA)

    if cv2.waitKey(1) == 27: # 종료 esc키
        break

    # Gray 이미지로 변환 -> 가우시안 필터로 잡음 제거(블러링) ->
    # 라플라시안 필터로 에지 검출 -> 
    # Threshold로 경계값만 남기고 제거 후 화면 반전 (흰바탕 검은선)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (9,9), 0)
    edges = cv2.Laplacian(img_gray, -1, None, 5)
    ret, sketch = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)
    
    # 경계선 강조를 위해 침식 연산 후 median 필터로 경계선 자연스럽게 처리
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    sketch = cv2.erode(sketch, kernel)
    sketch = cv2.medianBlur(sketch, 5)

    # Gray -> BGR 변환
    img_sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    # Color 이미지 선명선을 없애기 위해 평균 블러 필터 적용 후 (img_blur)
    # sketch 영상과 합성
    img_blur = cv2.blur(frame, (10,10) )
    img_paint = cv2.bitwise_and(img_blur, img_blur, mask=sketch)
    
    # 결과 출력
    merged = np.hstack((img_blur, img_sketch, img_paint))
    cv2.imshow('Sketch Camera', merged)
    
cap.release()
cv2.destroyAllWindows()
