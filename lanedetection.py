import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(input_image):
	gray_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
	blur_image = cv2.GaussianBlur(gray_image, (5,5), 0)
	canny_image = cv2.Canny(blur_image, 50, 150)
	return canny_image

def roi(input_image):
	height = input_image.shape[0]
	#print("height = ",height)
	widht = input_image.shape[1]
	a = int(widht * 0.1822)
	b = int(widht * 0.6514)
	c = int(widht * 0.555)
	d = int(height * 0.636)
	e = int(widht * 0.41)
	#print("widht = ", widht)
	poly = np.array([
		[(a,height), (b,height),(c,d),(e,d)]
		])
	mask = np.zeros_like(input_image)
	cv2.fillPoly(mask, poly, 255)
	masked_image = cv2.bitwise_and(input_image,mask)
	return masked_image

def display_lines(input_image, input_lines):
	line_image = np.zeros_like(input_image)

	if input_lines is not None:
		for lines in input_lines:
			x1,y1,x2,y2 = lines.reshape(4)
			cv2.line(line_image, (x1,y1), (x2,y2), (0,255,0), 10)
	else:
		print("No lines detected")

	return line_image

def make_coordinates(input_image, avgeraged_parameters):
	slope, intercept = avgeraged_parameters
	y1 = input_image.shape[0]
	y2 = int(input_image.shape[0] * 0.636)
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)

	return np.array([x1,y1,x2,y2])

def average_slope_intercept(input_image, input_lines):
	#filtered_lines = []
	left_fit = []
	#centre_fit = []
	right_fit = []
	
	#for lines in input_lines:
	#	x1,y1,x2,y2 = lines.reshape(4)
	#	if abs(y1 - y2) > 5:
	#		filtered_lines.append((x1,y1,x2,y2))
	
	for lines in input_lines:
		#print(lines)
		x1,y1,x2,y2 = lines.reshape(4)
		parameters = np.polyfit((x1,x2), (y1,y2), 1)
		#print(parameters)
		slope = parameters[0]
		intercept = parameters[1]
		if slope < 0:
		   left_fit.append((slope, intercept))
		else:
		   right_fit.append((slope, intercept))
	#print(left_fit)
	#print(right_fit)

	left_fit_average = np.average(left_fit, axis = 0)
	right_fit_average = np.average(right_fit, axis = 0)

	left_line = make_coordinates(input_image, left_fit_average)
	right_line = make_coordinates(input_image, right_fit_average)

	return np.array([left_line, right_line])

def final_line_coordinates(input_image, input_lines):
	height = input_image.shape[0]
	m1 = input_lines[0][0]
	c1 = input_lines[0][1]
	m2 = input_lines[1][0]
	c2 = input_lines[1][1]


	x1 = (height - c1)/m1
	x4 = (300 - c1)/m1

	x2 = (height - c2)/m2
	x3 = (300 - c2)/m2

	return np.int32([np.array([
		[(x1,height), (x2,height), (x3, 300), (x4, 300)]])
	])


# image = cv2.imread('laneimage3.jpg')
# lane_image_cp = np.copy(image)
# canny_image = canny(lane_image_cp)
# cropped_image = roi(canny_image)
# minLineLength=40
# maxLineGap=10
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, minLineLength, maxLineGap)
# averaged_lines = average_slope_intercept(lane_image_cp, lines)
# lined_image = display_lines(lane_image_cp, averaged_lines)
# mask = np.zeros_like(lane_image_cp)
# color_coordinates = final_line_coordinates(lane_image_cp, averaged_lines)
# coloured_image = cv2.fillPoly(lined_image, color_coordinates, (255,0,0))
# combo_image = cv2.addWeighted(lane_image_cp, 0.8, lined_image, 1, 1)


# #without matplotlib.pyplot, use this
# cv2.imshow('result', combo_image)

# #shows this to know coordinates
# #plt.imshow(cropped_image)

# #without matplotlib.pyplot, use this
# cv2.waitKey(0)

#shows this to know coordinates
#plt.show()

cap = cv2.VideoCapture("lane_video.mp4")
while(cap.isOpened()):
	ret, frame = cap.read()
	frame = np.copy(frame)
	canny_image = canny(frame)
	cropped_image = roi(canny_image)
	minLineLength=40
	maxLineGap=10
	lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, minLineLength, maxLineGap)
	averaged_lines = average_slope_intercept(frame, lines)
	lined_image = display_lines(frame, averaged_lines)
	#mask = np.zeros_like(lane_image_cp)
	#color_coordinates = final_line_coordinates(lane_image_cp, averaged_lines)
	#coloured_image = cv2.fillPoly(lined_image, color_coordinates, (255,0,0))
	combo_image = cv2.addWeighted(frame, 0.8, lined_image, 1, 1)
	cv2.imshow("Lane-markings:",combo_image)
	if cv2.waitKey(1) == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
