# https://bkshin.tistory.com/entry/OpenCV-11-2%EC%B0%A8%EC%9B%90-%ED%9E%88%EC%8A%A4%ED%86%A0%EA%B7%B8%EB%9E%A8%EA%B3%BC-%EC%97%AD%ED%88%AC%EC%98%81back-project?category=1148027
# 역투영(Back Projection)

'''
역투영이란 관심 영역의 히스토그램과 유사한 히스토그램을 갖는 영역을 찾아내는 기법이다. 
역투영을 활용하면 이미지 내에서 특정 물체나 배경을 분리할 수 있다. 

예) 이미지 내에서 잔디만 분리하고 싶은 경우 잔디에 해당하는 관심 영역(ROI, region of interest)을 지정하고 역투영을 적용한다. 
그러면 잔디에 해당하는 부분은 흰색으로, 잔디가 아닌 부분은 검은색으로 서로 분리가 된다. 
다만 이 방법은 색상을 기준으로 분리하기 때문에 잔디와 비슷한 색상을 가진 다른 물체가 있는 경우 성능이 떨어지는 단점을 가진다.

cv2.calcBackProject(img, channel, hist, ranges, scale)
    img: 입력 이미지, [img]처럼 리스트로 감싸서 사용
    channel: 처리할 채널, 리스트로 감싸서 사용
    hist: 역투영에 사용할 히스토그램
    ranges: 각 픽셀이 가질 수 있는 값의 범위
    scale: 결과에 적용할 배율 계수
'''

import os, sys
import cv2
import numpy as np
import matplotlib.pylab as plt

win_name = 'back_projection'

#path = '/home/sky/sky/VisualStudio/OpenCV/data2/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data2'

img = cv2.imread(os.path.join(path, "pump_horse.jpg"))
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
draw = img.copy()

#--⑤ 역투영된 결과를 마스킹해서 결과를 출력하는 공통함수
# cv2.getStructuringElement()와 cv2.filter2D()는 마스크의 표면을 부드럽게 해주는 역할
def masking(bp, win_name):
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(bp,-1,disc,bp)
    _, mask = cv2.threshold(bp, 1, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow(win_name, result)

#--⑥ 직접 구현한 역투영 함수
def backProject_manual(hist_roi):
    #--⑦ 전체 영상에 대한 H,S 히스토그램 계산
    hist_img = cv2.calcHist([hsv_img], [0,1], None,[180,256], [0,180,0,256])
    print(hist_img)
    #--⑧ 선택영역과 전체 영상에 대한 히스토그램 비율계산
    hist_rate = hist_roi/(hist_img + 1)
    #print(hist_rate)
    #--⑨ 비율에 맞는 픽셀 값 매핑
    h,s,v = cv2.split(hsv_img)
    bp = hist_rate[h.ravel(), s.ravel()]
    #print(bp)

    # 비율은 1을 넘어서는 안되기 때문에 1을 넘는 수는 1을 갖게 함
    bp = np.minimum(bp, 1)
    # 1차원 배열을 원래의 shape로 변환
    bp = bp.reshape(hsv_img.shape[:2])
    cv2.normalize(bp,bp, 0, 255, cv2.NORM_MINMAX)
    bp = bp.astype(np.uint8)
    #--⑩ 역 투영 결과로 마스킹해서 결과 출력
    masking(bp,'result_manual')
 
# OpenCV API로 구현한 함수 ---⑪ 
def backProject_cv(hist_roi):
    # 역투영 함수 호출 ---⑫
    bp = cv2.calcBackProject([hsv_img], [0, 1], hist_roi,  [0, 180, 0, 256], 1)
    # 역 투영 결과로 마스킹해서 결과 출력 ---⑬ 
    masking(bp,'result_cv')

# ROI 선택 ---①
(x,y,w,h) = cv2.selectROI(win_name, img, False)
if w > 0 and h > 0:
    roi = draw[y:y+h, x:x+w]
    # 빨간 사각형으로 ROI 영역 표시
    cv2.rectangle(draw, (x, y), (x+w, y+h), (0,0,255), 2)
    #--② 선택한 ROI를 HSV 컬러 스페이스로 변경
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    #--③ H,S 채널에 대한 히스토그램 계산
    hist_roi = cv2.calcHist([hsv_roi],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    #--④ ROI의 히스토그램을 매뉴얼 구현함수와 OpenCV 이용하는 함수에 각각 전달
    backProject_manual(hist_roi)
    backProject_cv(hist_roi)

cv2.imshow(win_name, draw)
cv2.waitKey()
cv2.destroyAllWindows()

