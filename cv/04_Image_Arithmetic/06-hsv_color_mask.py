# https://bkshin.tistory.com/entry/OpenCV-9-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%97%B0%EC%82%B0?category=1148027
# HSV 색상으로 마스킹 (hsv_color_mask.py)

import os, sys 
import cv2
import numpy as np
import matplotlib.pylab as plt

#--① 큐브 영상 읽어서 HSV로 변환
#path = '/home/sky/sky/VisualStudio/OpenCV/data2/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data2'
img = cv2.imread(os.path.join(path, "cube.jpg"))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#--② 색상별 영역 지정
blue1 = np.array([90, 50, 50])
blue2 = np.array([120, 255, 255])
green1 = np.array([45, 50, 50])
green2 = np.array([75, 255, 255])
red1 = np.array([0, 50, 50])
red2 = np.array([15, 255, 255])
red3 = np.array([165, 50, 50])
red4 = np.array([180, 255, 255])
yellow1 = np.array([20, 50, 50])
yellow2 = np.array([35, 255, 255])

# --③ 색상에 따른 마스크 생성
'''
cv2.inRange(src, lowerb, upperb, dst=None) -> dst; 특정 색상 영역을 추출
   • src: 입력 행렬
   • lowerb: 하한 값 행렬 또는 스칼라
   • upperb: 상한 값 행렬 또는 스칼라
   • dst: 입력 영상과 같은 크기의 마스크 영상. (numpy.uint8) 범위 안에 들어가는 픽셀은 255, 나머지는 0으로 설정
=> cv2.inRange(hsv, lower, upper) 함수는 hsv의 모든 값 중 lower와 upper 범위 사이에 있는 값은 255, 나머지 값은 0으로 변환한다.
'''
mask_blue = cv2.inRange(hsv, blue1, blue2)
mask_green = cv2.inRange(hsv, green1, green2)
mask_red = cv2.inRange(hsv, red1, red2)
mask_red2 = cv2.inRange(hsv, red3, red4)
mask_yellow = cv2.inRange(hsv, yellow1, yellow2)

#--④ 색상별 마스크로 색상만 추출
res_blue = cv2.bitwise_and(img, img, mask=mask_blue)
res_green = cv2.bitwise_and(img, img, mask=mask_green)
res_red1 = cv2.bitwise_and(img, img, mask=mask_red)
res_red2 = cv2.bitwise_and(img, img, mask=mask_red2)
res_red = cv2.bitwise_or(res_red1, res_red2)
res_yellow = cv2.bitwise_and(img, img, mask=mask_yellow)

#--⑤ 결과 출력
imgs = {'original': img, 'blue':res_blue, 'green':res_green, 
                            'red':res_red, 'yellow':res_yellow}
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(2,3, i+1)
    plt.title(k)
    plt.imshow(v[:,:,::-1])
    plt.xticks([]); plt.yticks([])
plt.show()
