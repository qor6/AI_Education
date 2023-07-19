# https://bkshin.tistory.com/entry/OpenCV-7-%E3%85%87%E3%85%87?category=1148027
# 마우스로 관심영역 지정 및 표시, 저장 (roi_crop_mouse.py)

import os, sys
import cv2
import numpy as np

#path = '/home/sky/sky/VisualStudio/OpenCV/data/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data2'

fname = os.path.join(path, "sunset.jpg")

# selectROI 사용 : 원본 이미지를 띄우고, 마우스 이벤트 처리도 도와줌
'''
ret = cv2.selectROI(win_name, img, showCrossHair=True, fromCenter=False)
- win_name: 관심영역을 표시할 창의 이름
- img: 관심영역을 표시할 이미지
- showCrossHair: 선택 영역 중심에 십자 모양 표시 여부
- fromCenter: 마우스 시작 지점을 영역의 중심으로 지정
- ret: 선택한 영역의 좌표와 크기 (x, y, w, h); 선택을 취소하면 모두 0으로 지정됨
'''

img = cv2.imread(fname)
if img is None:
    print('Image load failed')
    sys.exit()

x, y, w, h = cv2.selectROI("image", img, False)  # 선택 영역 표시

if w and h:
    roi = img[y:y+h, x:x+w]
    cv2.imshow('cropped', roi)              # ROI 지정 영역을 새창으로 표시
    cv2.moveWindow('cropped', 0, 0)         # 새창을 화면 좌측 상단에 이동
    #cv2.imwrite('./cropped2.jpg', roi)      # ROI 영역만 파일로 저장
    # 드래그 한 뒤 'c'를 누르면 취소

cv2.waitKey(0)
cv2.destroyAllWindows()


