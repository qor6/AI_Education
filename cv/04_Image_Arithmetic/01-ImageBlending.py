import os, sys
import cv2
import numpy as np

#path = '/home/sky/sky/VisualStudio/OpenCV/data/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data'

img1 = cv2.imread(os.path.join(path, "flower1.jpg"))
img2 = cv2.imread(os.path.join(path, "flower2.jpg"))

'''
cv2.addWeight(img1, alpha, img2, beta, gamma)
    img1, img2: 합성할 두 이미지
    alpha: img1에 지정할 가중치(알파 값)
    beta: img2에 지정할 가중치, 흔히 (1-alpha) 적용
    gamma: 연산 결과에 가감할 상수, 흔히 0 적용
'''

def nothing(x):
    pass

cv2.namedWindow('image')
cv2.createTrackbar('W', 'image', 0, 100, nothing)

while True:

    w = cv2.getTrackbarPos('W','image')
    dst = cv2.addWeighted(img1,float(100-w) * 0.01, img2,float(w) * 0.01, 0)

    cv2.imshow('dst', dst)

    if cv2.waitKey(1) &0xFF == 27:  # esc key
        break;

cv2.destroyAllWindows()


win_name = 'Alpha blending'     # 창 이름
trackbar_name = 'fade'          # 트렉바 이름

# ---① 트렉바 이벤트 핸들러 함수
def onChange(x):
    alpha = x/100
    dst = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0) 
    cv2.imshow(win_name, dst)

# ---② 합성 영상 읽기
#img1 = cv2.imread(cv2.imread(os.path.join(path, "man_face.jpg")))
#img2 = cv2.imread(cv2.imread(os.path.join(path, "lion_face.jpg")))

# ---③ 이미지 표시 및 트렉바 붙이기
cv2.imshow(win_name, img1)
cv2.createTrackbar(trackbar_name, win_name, 0, 100, onChange)

cv2.waitKey()
cv2.destroyAllWindows()


