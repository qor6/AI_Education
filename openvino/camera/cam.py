import cv2
cap = cv2.VideoCapture(0)
while cap.isOpened():
	ret, image = cap.read()
	if not ret:
		cv2.waitKey(1)
		continue
		
	cv2.imshow("display", image)
	inp = cv2.waitKey(1)
	if inp==ord('q'):
		break
