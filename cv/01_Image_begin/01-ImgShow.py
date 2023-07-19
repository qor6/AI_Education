import os
import cv2

#path = '/home/sky/sky/VisualStudio/OpenCV/data/'
path = 'C:/Users/sky/Documents/Python Scripts/OpenCV VSCode/data'
fname = os.path.join(path, "lena.jpg")

original = cv2.imread(fname, cv2.IMREAD_COLOR)
gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
unchange = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

cv2.imshow('Original', original)
cv2.imshow('Gray', gray)
cv2.imshow('Unchange', unchange)

cv2.waitKey(0)
cv2.destroyAllWindows()
