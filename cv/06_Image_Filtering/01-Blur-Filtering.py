# https://bkshin.tistory.com/entry/OpenCV-17-%ED%95%84%ED%84%B0Filter%EC%99%80-%EC%BB%A8%EB%B3%BC%EB%A3%A8%EC%85%98Convolution-%EC%97%B0%EC%82%B0-%ED%8F%89%EA%B7%A0-%EB%B8%94%EB%9F%AC%EB%A7%81-%EA%B0%80%EC%9A%B0%EC%8B%9C%EC%95%88-%EB%B8%94%EB%9F%AC%EB%A7%81-%EB%AF%B8%EB%94%94%EC%96%B8-%EB%B8%94%EB%9F%AC%EB%A7%81-%EB%B0%94%EC%9D%B4%EB%A0%88%ED%84%B0%EB%9F%B4-%ED%95%84%ED%84%B0?category=1148027
# Image Blurring

import os, sys 
import cv2
import numpy as np

#path = '/home/sky/sky/VisualStudio/OpenCV/data2/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data2'

# img = cv2.imread(os.path.join(path, "yate.jpg")) # Nature Color image
img = cv2.imread(os.path.join(path, "gaussian_noise.jpg")) # Noised gray image

'''
Convolution 연산

dst = cv2.filter2D(src, ddepth, kernel, dst, anchor, delta, borderType)
    src: 입력 영상, Numpy 배열
    ddepth: 출력 영상의 dtype (-1: 입력 영상과 동일)
    kernel: 컨볼루션 커널, float32의 n x n 크기 배열
    dst(optional): 결과 영상
    anchor(optional): 커널의 기준점, default: 중심점 (-1, -1)
    delta(optional): 필터가 적용된 결과에 추가할 값
    borderType(optional): 외곽 픽셀 보정 방법 지정
'''

'''
#5x5 평균 필터 커널 생성    ---①
kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                   [0.04, 0.04, 0.04, 0.04, 0.04],
                   [0.04, 0.04, 0.04, 0.04, 0.04],
                   [0.04, 0.04, 0.04, 0.04, 0.04],
                   [0.04, 0.04, 0.04, 0.04, 0.04]])
'''

# 5x5 평균 필터 커널 생성  ---②
kernel = np.ones((5,5))/5**2

# 필터 적용             ---③
blured = cv2.filter2D(img, -1, kernel)

# 결과 출력
cv2.imshow('origin', img)
cv2.imshow('avrg blur', blured) 

cv2.waitKey(0)
cv2.destroyAllWindows()

'''
OpenCV 평균 블러링 함수를 제공

dst = cv2.blur(src, ksize, dst, anchor, borderType)
    src: 입력 영상, numpy 배열
    ksize: 커널의 크기
    나머지 파라미터는 cv2.filter2D()와 동일

dst = cv2.boxFilter(src, ddepth, ksize, dst, anchor, normalize, borderType)
    ddepth: 출력 영상의 dtype (-1: 입력 영상과 동일)
    normalize(optional): 커널 크기로 정규화(1/ksize²) 지정 여부 (Boolean), default=True
    나머지 파라미터는 cv2.filter2D()와 동일
'''

# blur() 함수로 블러링  ---①
blur1 = cv2.blur(img, (10,10))
# boxFilter() 함수로 블러링 적용 ---②
blur2 = cv2.boxFilter(img, -1, (10,10))

# 결과 출력
merged = np.hstack( (img, blur1, blur2))
cv2.imshow('blur', merged)

cv2.waitKey(0)
cv2.destroyAllWindows()


'''
가우시안 블러링(Gaussian Blurring)

cv2.GaussianBlur(src, ksize, sigmaX, sigmaY, borderType)
    src: 입력 영상
    ksize: 커널 크기 (주로 홀수)
    sigmaX: X 방향 표준편차 (0: auto)
    sigmaY(optional): Y 방향 표준편차 (default: sigmaX)
    borderType(optional): 외곽 테두리 보정 방식

ret = cv2.getGaussianKernel(ksize, sigma, ktype)
    ret: 가우시안 커널 (1차원이므로 ret * ret.T 형태로 사용해야 함)
'''

# 가우시안 커널을 직접 생성해서 블러링  ---①
k1 = np.array([[1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]]) *(1/16)
blur_gauss1 = cv2.filter2D(img, -1, k1)

# 가우시안 커널을 직접 생성해서 블러링  ---①
k1 = np.array([[1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]]) *(1/16)
blur_gauss1 = cv2.filter2D(img, -1, k1)

# 가우시안 커널을 API로 얻어서 블러링 ---②
k2 = cv2.getGaussianKernel(3, 0)
blur_gauss2 = cv2.filter2D(img, -1, k2*k2.T)

# 가우시안 블러 API로 블러링 ---③
blur_gauss3 = cv2.GaussianBlur(img, (3, 3), 0)

# 결과 출력
print('k1:', k1)
print('k2:', k2*k2.T)
merged = np.hstack((img, blur1, blur_gauss2, blur_gauss3))

cv2.imshow('gaussian blur', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
미디언 블러링(Median Blurring) : 커널의 픽셀 값 중 중앙값을 선택
dst = cv2.medianBlur(src, ksize)
    src: 입력 영상
    ksize: 커널 크기
'''

'''
바이레터럴 필터(Bilateral Filter)
https://en.wikipedia.org/wiki/Bilateral_filter
블러링은 잡음 제거 효과는 좋지만 경계도 흐릿하게 만드는 문제가 있음. 
바이레터럴 필터는 가우시안 필터와 경계 필터를 결합함. 
경계도 뚜렷하고 노이즈도 제거되는 효과가 있지만 속도가 느리다는 단점이 있음.
dst = cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace, dst, borderType)
    src: 입력 영상
    d: 필터의 직경(diameter), 5보다 크면 매우 느림
    sigmaColor: 색공간의 시그마 값
    sigmaSpace: 좌표 공간의 시그마 값
'''

#img = cv2.imread(os.path.join(path, "gaussian_noise.jpg"))

# 가우시안 필터 적용 ---①
gauss_blur = cv2.GaussianBlur(img, (5,5), 0)

# median 필터 적용 ---②
median_blur = cv2.medianBlur(img, 5)

# 바이레터럴 필터 적용 ---②
bilateral_blur = cv2.bilateralFilter(img, 5, 75, 75)

# 결과 출력
merged = np.hstack((img, gauss_blur, median_blur, bilateral_blur))
cv2.imshow('median & bilateral', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()