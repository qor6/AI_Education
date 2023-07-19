# https://bkshin.tistory.com/entry/OpenCV-10-%ED%9E%88%EC%8A%A4%ED%86%A0%EA%B7%B8%EB%9E%A8?category=1148027
# 히스토그램 평탄화 (Equalization)

'''
Histogram Equalization : 특정 영역에 집중되어 있는 분포를 골고루 분포하도록 함 
https://en.wikipedia.org/wiki/Histogram_equalization

이미지의 각 픽셀의 cumulative distribution function(cdf)값을 구하고, Histogram Equalization 공식에 대입하여 0 ~ 255 사이의 값으로 변환
h(v) = round( (cdf(f)-cdf_min)/(MxN-cdf_min) x (L-1) )

dst = cv2.equalizeHist(src, dst)
    src: 대상 이미지, 8비트 1 채널
    dst(optional): 결과 이미지
'''

import os, sys
import cv2
import numpy as np
import matplotlib.pylab as plt

#--① 대상 영상으로 그레이 스케일로 읽기
#path = '/home/sky/sky/VisualStudio/OpenCV/data2/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data2'
img = cv2.imread(os.path.join(path, "yate.jpg"))
rows, cols = img.shape[:2]

'''
# 회색조 이미지에 평탄화 적용
'''
# Grayscale로 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#--② 이퀄라이즈 연산을 직접 적용
hist = cv2.calcHist([gray], [0], None, [256], [0, 256]) # 히스토그램 계산
cdf = hist.cumsum()                                     # 누적 히스토그램 
cdf_m = np.ma.masked_equal(cdf, 0)                      # 0(zero)인 값을 NaN으로 제거
cdf_m = (cdf_m - cdf_m.min()) /(rows * cols) * 255      # 이퀄라이즈 히스토그램 계산
cdf = np.ma.filled(cdf_m, 0).astype('uint8')             # NaN을 다시 0으로 환원
print(cdf.shape)
img2 = cdf[gray]                                        # 히스토그램을 픽셀로 맵핑

#--③ OpenCV API로 이퀄라이즈 히스토그램 적용
img3 = cv2.equalizeHist(gray)

#--④ 이퀄라이즈 결과 히스토그램 계산
hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([img3], [0], None, [256], [0, 256])

#--⑤ 결과 출력
cv2.imshow('Before', gray)
cv2.imshow('Manual', img2)
cv2.imshow('cv2.equalizeHist()', img3)
hists = {'Before':hist, 'Manual':hist2, 'cv2.equalizeHist()':hist3}
for i, (k, v) in enumerate(hists.items()):
    plt.subplot(1,3,i+1)
    plt.title(k)
    plt.plot(v)
plt.show()


# 색상 이미지에 대한 평탄화 적용 : BRG -> YUV
'''
히스토그램 평탄화는 색상 이미지에도 적용할 때, BGR 3개 채널을 모두 평탄화해야 한다. 
하지만 YUV나 HSV를 활용하면 하나의 밝기 채널만 조절하면 된다. 
색상 이미지를 BGR에서 YUV 형식으로 변환하여 밝기 채널에 평탄화를 적용한 예제이다.
'''

#--① 컬러 스케일을 BGR에서 YUV로 변경
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) 

#--② YUV 컬러 스케일의 첫번째 채널에 대해서 이퀄라이즈 적용
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0]) 

#--③ 컬러 스케일을 YUV에서 BGR로 변경
img4 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) 

cv2.imshow('Before', img)
cv2.imshow('After', img4)
cv2.waitKey()
cv2.destroyAllWindows()

