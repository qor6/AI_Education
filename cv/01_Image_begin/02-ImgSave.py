import os
import sys
import cv2

#path = '/home/sky/sky/VisualStudio/OpenCV/data/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data'
fname = os.path.join(path, "lena.jpg")

#img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
img = cv2.imread(fname, cv2.IMREAD_COLOR)

if img is None:
    print('Image load failed')
    sys.exit()

cv2.imshow('image',img)

k = cv2.waitKey(0) & 0xFF
if k == 27: # esc key
    cv2.destroyAllWindow()
elif k == ord('s'): # 's' key
    cv2.imwrite('lenagray.png',img)
    cv2.destroyAllWindow()

