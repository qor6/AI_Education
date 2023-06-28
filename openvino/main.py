import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core

ie = Core()

#model = ie.read_model(model="model/face-detection-0206.xml")	#FP32
model = ie.read_model(model="model/FP16-INT8/face-detection-0206.xml")
compiled_model = ie.compile_model(model=model, device_name="CPU")

input_layer_ir = compiled_model.input(0)
output_layer_ir_box = compiled_model.output("boxes")

# Text detection models expect an image in BGR format.
#image = cv2.imread("people.jpg")
cam = cv2.VideoCapture(0)

while cam.isOpened():
	ret, cap = cam.read()
	if not ret:
		cv2.waitKey(1)
		continue
		
	N, C, H, W = input_layer_ir.shape
	print(N,C,H,W)
	
	resized_cap = cv2.resize(cap, (W, H))
	input_image = np.expand_dims(resized_cap.transpose(2, 0, 1), 0)
	infer_result = compiled_model([input_image])
	boxes = infer_result[output_layer_ir_box]
	print(boxes.shape)
	
	boxes = boxes[~np.all(boxes == 0, axis=1)]

	for box in boxes:
    		print(box)
    		conf = box[-1]
    		leftTop = (int(box[0]), int(box[1]))
    		rightBottom = (int(box[2]), int(box[3]))
    		if conf > 0.3:
    			cv2.rectangle(resized_cap, leftTop, rightBottom, (0,0,255),3)
	
	
	cv2.imshow("display", resized_cap)
	
	cv2.imwrite("me.jpg", cap)
	inp = cv2.waitKey(1)
	if inp==ord('q'):
		break
	if inp==ord('c'):
		cv2.imwrite("me.jpg", cap)


## N,C,H,W = batch size, number of channels, height, width.
#N, C, H, W = input_layer_ir.shape

#print(N,C,H,W)
## Resize the image to meet network expected input sizes.
#resized_image = cv2.resize(cap, (W, H))

## Reshape to the network input shape.
#input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)

##plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB));

##Create an inference request
#infer_result = compiled_model([input_image])
#boxes = infer_result[output_layer_ir_box]

#print(boxes.shape)
##print(boxes)

#exit(0)

#Remove zero only boxes
#boxes = boxes[~np.all(boxes == 0, axis=1)]

#for box in boxes:
#    print(box)
#    conf = box[-1]
#    leftTop = (int(box[0]), int(box[1]))
#    rightBottom = (int(box[2]), int(box[3]))
#    if conf > 0.3:
#    	cv2.rectangle(resized_image, leftTop, rightBottom, (0,0,255),3)
    	
#cv2.imshow("display", resized_image)
#cv2.waitKey(0)


