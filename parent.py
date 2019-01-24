from threading import Thread
import os
import tkinter as tk
from tkinter import *
import tkinter.messagebox
import winsound
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
from selenium import webdriver
import serial

default_lat = "19.000000"
default_long = "073.000000"

target_lat = "19.000001"
targt_long = "073.000001"

cruise_speed = "10"
ETA = "45"



def reachme():

	#s = serial.Serial()
	#res = s.readline()
	#if(res[0] == 80):
	#	CarLatitude = str(chr(res[10]))+str(chr(res[11]))+str(chr(res[12]))+str(chr(res[13]))+str(chr(res[14]))+str(chr(res[15]))+str(chr(res[16]))+str(chr(res[17]))+str(chr(res[18]))
	#	CarLongitude = str(chr(res[20]))+str(chr(res[21]))+str(chr(res[22]))+str(chr(res[23]))+str(chr(res[24]))+str(chr(res[25]))+str(chr(res[26]))+str(chr(res[27]))+str(chr(res[28]))+str(chr(res[29]))
	#	return CarLatitude,CarLongitude
	#else:
	#	return default_lat,default_long
	return default_lat,default_long

def current_location(position):

	if position == "start":
		glob_folder = os.path.join("C:/Users/Ebrahim/Desktop/Projects/finalyearproj/location.html")
		html_file_list = glob.glob(glob_folder)
		index = 1

		for html_file in html_file_list:
			
			temp_name = "file://" + html_file
			driver = webdriver.Chrome("C:/Program Files (x86)/Google/Chrome/Application/chromedriver.exe")
			driver.get(temp_name)
			save_name = '00' + str(index) + '.png'       
			driver.save_screenshot("screenshot1.png")
			driver.quit()
			index += 1

		 # crop as required
			img = Image.open("screenshot1.png")
			box = (1, 1, 1000, 1000)
			area = img.crop(box)
			area.save('cropped_image' + str(index), 'png')

	elif position == "end":

		glob_folder = os.path.join("C:/Users/Ebrahim/Desktop/Projects/finalyearproj/location2.html")
		html_file_list = glob.glob(glob_folder)
		index = 1

		for html_file in html_file_list:

			temp_name = "file://" + html_file
			driver = webdriver.Chrome("C:/Program Files (x86)/Google/Chrome/Application/chromedriver.exe")
			driver.get(temp_name)
			save_name = '00' + str(index) + '.png'       
			driver.save_screenshot("screenshot2.png")
			driver.quit()
			index += 1

		 # crop as required
			img = Image.open("screenshot2.png")
			box = (1, 1, 1000, 1000)
			area = img.crop(box)
			area.save('cropped_image' + str(index), 'png')


		





def GUI1():
	top = tk.Tk()
	top.wm_title("Autonomous Mode On!")
	top.attributes("-fullscreen", True)
	#C = Canvas(top, bg="blue", height=250, width=300)
	#filename = PhotoImage(file = "sovsct.PNG")
	#background_label = Label(top, image=filename)
	#background_label.place(x=0, y=0, relwidth=1, relheight=1)
	#C.pack()

	# L1 = Label(top, text="                 Welcome to Sovereign scout!              ").grid(row=0,column=0)
	# #L2 = Label(top, text="Sovereign scout!").grid(row=0,column=1)
	# L3 = Label(top, text=" Current Location(Grid no):").grid(row=1,column=0)
	# L4 = Label(top, text="0").grid(row=1,column=1)
	# L5 = Label(top, text="Target Location(Grid no):").grid(row=2,column=0)
	# E1 = Entry(top, bd =3)
	# E1.grid(row=2,column=1)
	# L5 = Label(top, text="Cruise Speed(km/hr): ").grid(row=3,column=0)
	# E2 = Entry(top, bd =3)
	# E2.grid(row=3,column=1)
	# def _quit():
	# 	top.quit()
	# 	top.destroy()
	# button = tk.Button(master=top, text='            Lets Ride!            ', command=_quit).grid(row = 4, column = 0)
	L1 = Label(top, text="                 Welcome to Sovereign scout!              ", font = ("Courier", 44), fg = 'red', bg = 'yellow')
	L1.pack(side = TOP)
	a,b = reachme()
	L2 = Label(top, text=" Current Location(Latitude,Longitude):"+str(a)+","+str(b), font = ("Courier", 22), fg = 'blue' )
	L2.pack()
	L3 = Label(top, text=" Target Location(Latitude,Longitude): ", font = ("Courier", 22), fg = 'blue' )
	L3.pack()
	E1 = Entry(top, bd =3)
	E1.pack()
	L4 = Label(top, text=" Cruise Speed(km/hr): ", font = ("Courier", 22), fg = 'blue' )
	L4.pack()
	E2 = Entry(top, bd =3)
	E2.pack()
	

	def _quit():
		top.quit()
		top.destroy()
		Thread(target = GUI2).start()
		#Thread(target = audio2).start()


	button = tk.Button(master = top, text = 'LETS RIDE !', command = _quit, font = ("Courier", 22), bg = 'gray', fg = 'green' )
	button.pack()
	
	current_location("start")
	img = PhotoImage(file = "screenshot1.png")
	panel = tk.Label(top, image = img)
	panel.pack(side = "bottom")

	Thread(target = audio1).start()
	top.mainloop()

	#Thread(target = GUI2).start()

	#Thread(target = audio2).start() 

	
def GUI2():
	root = tk.Tk()
	root.wm_title("Ride Parameters")
	root.attributes("-fullscreen", True)
	#C = Canvas(root, bg="blue", height=250, width=300)
	#filename = PhotoImage(file = "sovsct.PNG")
	#background_label = Label(root, image=filename)
	#background_label.place(x=0, y=0, relwidth=1, relheight=1)
	#C.pack()
	# L6 = Label(root, text="                Alrigt! Heading to location:      ").grid(row=0,column=0)
	# L7 = Label(root, text="20").grid(row=0,column=1)
	# L3 = Label(root, text="Cruise Speed:").grid(row=1,column=0)
	# L3 = Label(root, text="15km/hr").grid(row=1,column=1)
	# L3 = Label(root, text="ETA").grid(row=2,column=0)
	# L3 = Label(root, text="45sec").grid(row=2,column=1)
	# def _quit2():
	# 	root.quit()
	# 	root.quit()
	# button2 = tk.Button(master=root, text='            Confirm ride?            ', command=_quit2).grid(row = 3, column = 0)

	L1 = Label(root, text="                 RIDE PARAMETERS              ", font = ("Courier", 44), fg = 'red', bg = 'yellow')
	L1.pack(side = TOP)
	L2 = Label(root, text="Target Location: "+str(target_lat)+","+str(targt_long), font = ("Courier", 22), fg = 'blue' )
	L2.pack()
	L3 = Label(root, text="Cruise Speed(km/hr):"+str(cruise_speed), font = ("Courier", 22), fg = 'blue' )
	L3.pack()
	L4 = Label(root, text="ETA(sec): "+str(ETA), font = ("Courier", 22), fg = 'blue' )
	L4.pack()

	def _quit2():
		root.quit()
		root.destroy()
		import lanedetection

	button = tk.Button(master = root, text = 'CONFIRM RIDE ?', command = _quit2, font = ("Courier", 22), bg = 'gray', fg = 'green' )
	button.pack()

	current_location("end")
	img = PhotoImage(file = "screenshot2.png")
	panel = tk.Label(root, image = img)
	panel.pack(side = "bottom")

	Thread(target = audio2).start()

	root.mainloop()

	#import lanedetection













	


	

	




# def lanedetect():
# 	def canny(input_image):
# 		gray_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
# 		blur_image = cv2.GaussianBlur(gray_image, (5,5), 0)
# 		canny_image = cv2.Canny(blur_image, 50, 150)
# 		return canny_image


# 	def roi(input_image):
# 		height = input_image.shape[0]
# 		widht = input_image.shape[1]
# 		a = int(widht * 0.1822)
# 		b = int(widht * 0.6514)
# 		c = int(widht * 0.555)
# 		d = int(height * 0.636)
# 		e = int(widht * 0.41)
# 		poly = np.array([
# 		[(a,height), (b,height),(c,d),(e,d)]
# 		])
# 		mask = np.zeros_like(input_image)
# 		cv2.fillPoly(mask, poly, 255)
# 		masked_image = cv2.bitwise_and(input_image,mask)
# 		return masked_image


# 	def display_lines(input_image, input_lines):
# 		line_image = np.zeros_like(input_image)
# 		if input_lines is not None:
# 			for lines in input_lines:
# 				x1,y1,x2,y2 = lines.reshape(4)
# 				cv2.line(line_image, (x1,y1), (x2,y2), (0,255,0), 10)
# 		else:
# 			print("No lines detected")

# 		return line_image


# 	def make_coordinates(input_image, avgeraged_parameters):
# 		slope, intercept = avgeraged_parameters
# 		y1 = input_image.shape[0]
# 		y2 = int(input_image.shape[0] * 0.636)
# 		x1 = int((y1 - intercept)/slope)
# 		x2 = int((y2 - intercept)/slope)
# 		return np.array([x1,y1,x2,y2])


# 	def average_slope_intercept(input_image, input_lines):
# 	#filtered_lines = []
# 		left_fit = []
# 	#centre_fit = []
# 		right_fit = []
	
# 	#for lines in input_lines:
# 	#	x1,y1,x2,y2 = lines.reshape(4)
# 	#	if abs(y1 - y2) > 5:
# 	#		filtered_lines.append((x1,y1,x2,y2))
	
# 		for lines in input_lines:
# 		#print(lines)
# 			x1,y1,x2,y2 = lines.reshape(4)
# 			parameters = np.polyfit((x1,x2), (y1,y2), 1)
# 			#print(parameters)
# 			slope = parameters[0]
# 			intercept = parameters[1]
# 			if slope < 0:
# 				left_fit.append((slope, intercept))
# 			else:
# 				right_fit.append((slope, intercept))

# 			left_fit_average = np.average(left_fit, axis = 0)
# 			right_fit_average = np.average(right_fit, axis = 0)
			
# 			left_line = make_coordinates(input_image, left_fit_average)
# 			right_line = make_coordinates(input_image, right_fit_average)

# 		return np.array([left_line, right_line])
																		 



# 	cap = cv2.VideoCapture("lane_video.mp4")
# 	while(cap.isOpened()):
# 		ret, frame = cap.read()
# 		frame = np.copy(frame)
# 		canny_image = canny(frame)
# 		cropped_image = roi(canny_image)
# 		minLineLength=40
# 		maxLineGap=10
# 		lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, minLineLength, maxLineGap)
# 		averaged_lines = average_slope_intercept(frame, lines)
# 		lined_image = display_lines(frame, averaged_lines)
# 		#mask = np.zeros_like(lane_image_cp)
# 		#color_coordinates = final_line_coordinates(lane_image_cp, averaged_lines)
# 		#coloured_image = cv2.fillPoly(lined_image, color_coordinates, (255,0,0))
# 		combo_image = cv2.addWeighted(frame, 0.8, lined_image, 1, 1)
# 		cv2.imshow("Front Cmaera",combo_image)
# 		if cv2.waitKey(1) == ord('q'):
# 			break
# 	cap.release()
# 	cv2.destroyAllWindows()


def audio1():
	winsound.PlaySound("audio_1.wav",  winsound.SND_ALIAS)

def audio2():
	winsound.PlaySound("audio_2.wav",  winsound.SND_ALIAS)

Thread(target = GUI1).start() 
#Thread(target = audio1).start()
