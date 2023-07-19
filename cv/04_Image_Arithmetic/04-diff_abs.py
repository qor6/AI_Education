# https://bkshin.tistory.com/entry/OpenCV-9-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%97%B0%EC%82%B0?category=1148027
# 두 이미지의 차를 통해 도면의 차이 찾아내기 (diff_absolute.py)

import os, sys 
import numpy as np, cv2

#--① 연산에 필요한 영상을 읽고 그레이스케일로 변환
#path = '/home/sky/sky/VisualStudio/OpenCV/data2/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data2'

img1 = cv2.imread(os.path.join(path, "robot_arm1.jpg"))
img2 = cv2.imread(os.path.join(path, "robot_arm2.jpg"))

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#--② 두 영상의 절대값 차 연산
diff = cv2.absdiff(img1_gray, img2_gray)

#--③ 차 영상을 극대화 하기 위해 threshol 처리 및 컬러로 변환
ret, diff = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)
diff_red = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)

#--④ 두 번째 이미지에 변화 부분 표시
diff_red[:,:,2] = 0 
spot = cv2.bitwise_xor(img2, diff_red)
# B,G 채널 : diff_red가 0일때 img2 그대로 통과, 1일때 ~img2 변경 
# R채널 diff_red 0이므로, img2 R채널 그대로 통과

#--⑤ 결과 영상 출력
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('diff', diff)
cv2.imshow('spot', spot)
cv2.waitKey()
cv2.destroyAllWindows()

