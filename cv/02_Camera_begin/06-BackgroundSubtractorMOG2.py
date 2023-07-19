'''
AttributeError: module 'cv2' has no attribute 'bgsegm'
$ pip install opencv-contrib-python
'''

import os 
import cv2
import timeit

# 영상 정보 불러오기
#path = '/home/sky/sky/VisualStudio/OpenCV/data/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data'

fname = os.path.join(path, 'vtest.avi')
cap = cv2.VideoCapture(fname)

# frame 수 구하기
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)

# 배경 제거 객체 생성 --- ①
fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG()
fgbg2 = cv2.createBackgroundSubtractorMOG2()

'''
가우시안 혼합 배경제거 알고리즘 : Z.Zivkovic, 
"Improved adaptive Gausian mixture model for background subtraction", 2004 
"Efficient Adaptive Density Estimation per Image Pixel for the Task of 
Background Subtraction," 2006

cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)
history=500: 히스토리 개수
varThreshold=16: 분산 임계 값
detectShadows=True: 그림자 표시
'''

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 배경 제거 마스크 계산 --- ②
    fgmask1 = fgbg1.apply(frame)
    fgmask2 = fgbg2.apply(frame)
    cv2.imshow('frame',frame)
    cv2.imshow('bgsub-MOG',fgmask1)
    cv2.imshow('bgsub-MOG2',fgmask2)    

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


'''
# cv2.threshold : https://opencv-python.readthedocs.io/en/latest/doc/09.imageThresholding/imageThresholding.html
# cv2.findContoures : https://opencv-python.readthedocs.io/en/latest/doc/15.imageContours/imageContours.html
# cv2.rectangle : https://opencv-python.readthedocs.io/en/latest/doc/03.drawShape/drawShape.html

def MOG(frame):
    fgmask1 = fgbg1.apply(frame)
    _,fgmask1 = cv2.threshold(fgmask1, 175, 255, cv2.THRESH_BINARY)
    results = cv2.findContours(fgmask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in results[0]:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

while(True):

    ret, frame = cap.read()
    
    if ret is True:
        start_t = timeit.default_timer()    # 알고리즘 시작 시점
        
        """ 알고리즘 연산 """
        MOG(frame)
        
        terminate_t = timeit.default_timer()# 알고리즘 종료 시점
        cv2.imshow('video', frame)
        FPS = int(1./(terminate_t - start_t))
        print(FPS)       

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
'''