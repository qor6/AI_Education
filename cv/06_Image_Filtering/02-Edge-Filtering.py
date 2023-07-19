# https://bkshin.tistory.com/entry/OpenCV-18-%EA%B2%BD%EA%B3%84-%EA%B2%80%EC%B6%9C-%EB%AF%B8%EB%B6%84-%ED%95%84%ED%84%B0-%EB%A1%9C%EB%B2%84%EC%B8%A0-%EA%B5%90%EC%B0%A8-%ED%95%84%ED%84%B0-%ED%94%84%EB%A6%AC%EC%9C%97-%ED%95%84%ED%84%B0-%EC%86%8C%EB%B2%A8-%ED%95%84%ED%84%B0-%EC%83%A4%EB%A5%B4-%ED%95%84%ED%84%B0-%EB%9D%BC%ED%94%8C%EB%9D%BC%EC%8B%9C%EC%95%88-%ED%95%84%ED%84%B0-%EC%BA%90%EB%8B%88-%EC%97%A3%EC%A7%80?category=1148027

# 1) 미분 커널로 경계 검출 (edge_differential.py)

import os, sys
import cv2
import numpy as np
import matplotlib.pylab as plt

#path = '/home/sky/sky/VisualStudio/OpenCV/data2/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data2'
img = cv2.imread(os.path.join(path, "sudoku.jpg"))

#미분 커널 생성 ---①
gx_kernel = np.array([[ -1, 1]])
gy_kernel = np.array([[ -1],
                      [ 1]])

# 필터 적용 ---②
edge_gx = cv2.filter2D(img, -1, gx_kernel)
edge_gy = cv2.filter2D(img, -1, gy_kernel)

edge_mag = (edge_gx + edge_gy)

# 결과 출력
merged = np.hstack((img, edge_gx, edge_gy, edge_mag))

cv2.imshow('differential filter edge', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 2) 프리윗 필터 (Prewitt Filter)

# 프리윗 커널 생성
gx_k = np.array([[-1,0,1], [-1,0,1],[-1,0,1]])
gy_k = np.array([[-1,-1,-1],[0,0,0], [1,1,1]])

# 프리윗 커널 필터 적용
edge_gx = cv2.filter2D(img, -1, gx_k)
edge_gy = cv2.filter2D(img, -1, gy_k)

# 결과 출력
merged = np.hstack((img, edge_gx, edge_gy, edge_gx+edge_gy))
cv2.imshow('prewitt', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 3) 소벨 필터 (Sobel Filter)

'''
https://docs.opencv.org/4.5.3/d2/d2c/tutorial_sobel_derivatives.html

소벨 필터는 중심 픽셀의 차분 비중을 두 배로 준 필터임. 
따라서 소벨 필터는 x축, y축, 대각선 방향의 경계 검출에 모두 강함.

dst = cv2.Sobel(src, ddepth, dx, dy, dst, ksize, scale, delta, borderType)
    src: 입력 영상
    ddepth: 출력 영상의 dtype (-1: 입력 영상과 동일)
    dx, dy: 미분 차수 (0, 1, 2 중 선택, 둘 다 0일 수는 없음)
    ksize: 커널의 크기 (1, 3, 5, 7 중 선택)
    scale: 미분에 사용할 계수
    delta: 연산 결과에 가산할 값
'''

# 소벨 커널을 직접 생성해서 엣지 검출 ---①
## 소벨 커널 생성
gx_k = np.array([[-1,0,1], [-2,0,2],[-1,0,1]])
gy_k = np.array([[-1,-2,-1],[0,0,0], [1,2,1]])
## 소벨 필터 적용
edge_gx = cv2.filter2D(img, -1, gx_k)
edge_gy = cv2.filter2D(img, -1, gy_k)

# 소벨 API를 생성해서 엣지 검출
sobelx = cv2.Sobel(img, -1, 1, 0, ksize=3)
sobely = cv2.Sobel(img, -1, 0, 1, ksize=3) 

# 결과 출력
merged1 = np.hstack((img, edge_gx, edge_gy, edge_gx+edge_gy))
merged2 = np.hstack((img, sobelx, sobely, sobelx+sobely))
merged = np.vstack((merged1, merged2))
cv2.imshow('sobel', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 4) 라플라시안 필터 (Laplacian Filter)
'''
https://docs.opencv.org/4.5.3/d5/db5/tutorial_laplace_operator.html

라플라시안 필터는 2차 미분을 적용한 필터이다. 경계를 더 제대로 검출할 수 있다.

dst = cv2.Laplacian(src, ddepth, dst, ksize, scale, delta, borderType)
파라미터는 cv2.Sobel()과 동일하다.
'''

# 라플라시안 필터 적용 ---①
edge = cv2.Laplacian(img, -1)

# 결과 출력
merged = np.hstack((img, edge))
cv2.imshow('Laplacian', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
