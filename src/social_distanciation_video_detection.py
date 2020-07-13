from bird_view_transfo_functions import compute_perspective_transform,compute_point_perspective_transformation
from helping_functions import image_resize, create_blank, combine_image
from tf_model_object_detection import Model 
from colors import bcolors
from mutagen.mp3 import MP3
from playsound import playsound
import glob
import queue
import numpy as np
import itertools
import threading
import imutils
import time
import math
import glob
import yaml
import cv2
import os
import sys

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
BIG_CIRCLE = 25
SMALL_CIRCLE = 3


def get_human_box_detection(boxes,scores,classes,height,width):
	""" 
	For each object detected, check if it is a human and if the confidence >> our threshold.
	Return 2 coordonates necessary to build the box.
	@ boxes : all our boxes coordinates
	@ scores : confidence score on how good the prediction is -> between 0 & 1
	@ classes : the class of the detected object ( 1 for human )
	@ height : of the image -> to get the real pixel value
	@ width : of the image -> to get the real pixel value
	"""
	# print(boxes)
	array_boxes = list() # Create an empty list
	for i in range(boxes.shape[1]):
		# If the class of the detected object is 1 and the confidence of the prediction is > 0.75
		if int(classes[i]) == 1 and scores[i] > 0.75:
			# Multiply the X coordonnate by the height of the image and the Y coordonate by the width
			# To transform the box value into pixel coordonate values.
			box = [boxes[0,i,0],boxes[0,i,1],boxes[0,i,2],boxes[0,i,3]] * np.array([height, width, height, width])
			# Add the results converted to int
			array_boxes.append((int(box[0]),int(box[1]),int(box[2]),int(box[3])))
	return array_boxes


def get_centroids_and_groundpoints(array_boxes_detected):
	"""
	For every bounding box, compute the centroid and the point located on the bottom center of the box
	@ array_boxes_detected : list containing all our bounding boxes 
	"""
	array_centroids,array_groundpoints = [],[] # Initialize empty centroid and ground point lists 
	for index,box in enumerate(array_boxes_detected):
		# Draw the bounding box 
		# c
		# Get the both important points
		centroid,ground_point = get_points_from_box(box)
		array_centroids.append(centroid)
		array_groundpoints.append(ground_point)
	return array_centroids,array_groundpoints


def get_points_from_box(box):
	"""
	Get the center of the bounding and the point "on the ground"
	@ param = box : 2 points representing the bounding box
	@ return = centroid (x1,y1) and ground point (x2,y2)
	"""
	# Center of the box x = (x1+x2)/2 et y = (y1+y2)/2
	center_x = int(((box[1]+box[3])/2))
	center_y = int(((box[0]+box[2])/2))
	# Coordiniate on the point at the bottom center of the box
	center_y_ground = center_y + ((box[2] - box[0])/2)
	return (center_x,center_y),(center_x,int(center_y_ground))


def change_color_to_red_on_topview(pair):
	"""
	Draw red circles for the designated pair of points 
	"""
	cv2.circle(bird_view_img, (pair[0][0],pair[0][1]), BIG_CIRCLE, COLOR_RED, 2)
	cv2.circle(bird_view_img, (pair[0][0],pair[0][1]), SMALL_CIRCLE, COLOR_RED, -1)
	cv2.circle(bird_view_img, (pair[1][0],pair[1][1]), BIG_CIRCLE, COLOR_RED, 2)
	cv2.circle(bird_view_img, (pair[1][0],pair[1][1]), SMALL_CIRCLE, COLOR_RED, -1)

def change_color_to_green_on_topview(pair):
	"""
	Draw red circles for the designated pair of points 
	"""
	cv2.circle(bird_view_img, (pair[0][0],pair[0][1]), BIG_CIRCLE, COLOR_GREEN, 2)
	cv2.circle(bird_view_img, (pair[0][0],pair[0][1]), SMALL_CIRCLE, COLOR_GREEN, -1)
	cv2.circle(bird_view_img, (pair[1][0],pair[1][1]), BIG_CIRCLE, COLOR_GREEN, 2)
	cv2.circle(bird_view_img, (pair[1][0],pair[1][1]), SMALL_CIRCLE, COLOR_GREEN, -1)

def draw_rectangle(corner_points):
	# Draw rectangle box over the delimitation area
	cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[1][0], corner_points[1][1]), COLOR_BLUE, thickness=1)
	cv2.line(frame, (corner_points[1][0], corner_points[1][1]), (corner_points[3][0], corner_points[3][1]), COLOR_BLUE, thickness=1)
	cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[2][0], corner_points[2][1]), COLOR_BLUE, thickness=1)
	cv2.line(frame, (corner_points[3][0], corner_points[3][1]), (corner_points[2][0], corner_points[2][1]), COLOR_BLUE, thickness=1)





######################################### 
# # Load the config for the top-down view #
# #########################################
# print(bcolors.WARNING +"[ Loading config file for the bird view transformation ] "+ bcolors.ENDC)
# # with open("../conf/config_birdview.yml", "r") as ymlfile:
# with open("../conf/config_birdview.yml", "r") as ymlfile:
#     cfg = yaml.load(ymlfile)
# width_og, height_og = 0,0
# corner_points = []
# for section in cfg:
# 	corner_points.append(cfg["image_parameters"]["p1"])
# 	corner_points.append(cfg["image_parameters"]["p2"])
# 	corner_points.append(cfg["image_parameters"]["p3"])
# 	corner_points.append(cfg["image_parameters"]["p4"])
# 	width_og = int(cfg["image_parameters"]["width_og"])
# 	height_og = int(cfg["image_parameters"]["height_og"])
# 	img_path = cfg["image_parameters"]["img_path"]
# 	size_frame = cfg["image_parameters"]["size_frame"]
# print(bcolors.OKGREEN +" Done : [ Config file loaded ] ..."+bcolors.ENDC )


######################################### 
#		     Select the model 			#
#########################################
model_names_list = [name for name in os.listdir("../models/.") if name.find(".") == -1]
for index,model_name in enumerate(model_names_list):
    print(" - {} [{}]".format(model_name,index))
model_num = input(" Please select the number related to the model that you want : ")
if model_num == "":
	model_path="../models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb" 
else :
	model_path = "../models/"+model_names_list[int(model_num)]+"/frozen_inference_graph.pb"
print(bcolors.WARNING + " [ Loading the TENSORFLOW MODEL ... ]"+bcolors.ENDC)
model = Model(model_path)
print(bcolors.OKGREEN +"Done : [ Model loaded and initialized ] ..."+bcolors.ENDC)


######################################### 
#		     Select the video 			#
#########################################
# video_names_list = [name for name in os.listdir("../video/") if name.endswith(".mp4") or name.endswith(".avi")]
# for index,video_name in enumerate(video_names_list):
#     print(" - {} [{}]".format(video_name,index))
video_name = input("Enter the exact name of the video (including .mp4 or else) or 0 for webcam : ")
if video_name == "":
	video_path="../video/Demo1.mp4" 
elif video_name == "0":
	video_path = int(video_name)
else :
	video_path = "../video/"+video_name


######################################### 
#		    Minimal distance			#
#########################################
distance_minimum = input("Prompt the size of the minimal distance between 2 pedestrians : ")
if distance_minimum == "":
	distance_minimum = "110"

#Dictionary to save distance between pairs
distance_between_pairs = dict()
#Dictionary to start timer when the distance between pairs is less than the minimum distance defined.
timer_for_each_pairs = dict()
soundfile = "../sound/covid_message.mp3"
audio = MP3(soundfile)
audio_file_length = audio.info.length

#Take input for how many seconds do you want to wait when two people are close enough
second = input("How many seconds do you want to wait before playing warning even though pairs are not maintaining social distancing: ")
if second == "":
	seconds = 30 
else:
	seconds = int(second)

#Take input for how many seconds do you want to wait after playing warning.
wait = input("How many seconds do you want to wait before playing next warning: ")
if wait == "":
	waits = 30
else:
	waits = int(wait)

frame_per_second = input("How many frames per second is your webcam: ")
if frame_per_second == "":
	frame_per_seconds = 30
else:
	frame_per_seconds = int(frame_per_second)

def check_current_value(key,value):
    if value < float(distance_minimum):
        time.sleep(1)
        timer_for_each_pairs[key] += 1
    else:
        timer_for_each_pairs[key] = 0

# queue = []
# def check_latest_time(v):
#     if v >= seconds:
#         if len(queue) < 1:
# 	        queue.append(soundfile)
#         playsound(queue[0])

# def time_counter():
# 	time.sleep(1)
# 	time_to_wait += 1
time_to_wait = 0

def play_warning():
	playsound(soundfile)

def waiting_time():
	global time_to_wait
	for i in range(int(audio_file_length)+waits):
		for j in range(frame_per_seconds*2):
			to_sleep = 1/(frame_per_seconds*2)
			time.sleep(to_sleep)
			time_to_wait += to_sleep
	time_to_wait=0

# def check_to_play_warning():
# 	time_to_wait = 0

# 			playsound(soundfile)
# 			for i in range(waits):
# 				time.sleep(1)
# 				time_to_wait += 1
# 			time_to_wait=0
# 			break

# def time_counter():
# 	for i in range(int(audio_file_length) + 2 ):
# 		time.sleep(1)
# 		print(time_to_wait)
# 		time_to_wait += 1
		
# 	time_to_wait = 0




######################################################
#########									 #########
# 				START THE VIDEO STREAM               #
#########									 #########
######################################################
vs = cv2.VideoCapture(video_path)
output_video_1,output_video_2 = None,None

loop_count = 0
# Loop until the end of the video stream
while True:	
	# Load the frame
	(frame_exists, frame) = vs.read()
	# Test if it has reached the end of the video
	if not frame_exists:
		break
	else:
		# Resize the image to the correct size
		# frame = imutils.resize(frame, width=int(size_frame))
		frame = image_resize(frame, width = 600)
		height, width, channels = frame.shape
		# print(height, width,channels)
		# cv2.imshow("first_frame", frame)
		# cv2.waitKey(0)
		
		corner_points = [[0,0],[width, 0],[0,height],[width, height]]
		white = (255,255,255)

		bird_view_img = create_blank(width, height, rgb_color=white)
		######################################### 
		#     Compute transformation matrix		#
		#########################################
		# Compute  transformation matrix from the original frame
		matrix,imgOutput = compute_perspective_transform(corner_points,width,height,frame)
		# print(matrix)
		# blank_image = np.zeros((height,width,3), np.uint8)
		# height = blank_image.shape[0]
		# width = blank_image.shape[1] 
		dim = (width, height)
		
		# Make the predictions for this frame
		(boxes, scores, classes) =  model.predict(frame)

		if len(boxes)>0:
			# Get the human detected in the frame and return the 2 points to build the bounding box  
			array_boxes_detected = get_human_box_detection(boxes,scores[0].tolist(),classes[0].tolist(),frame.shape[0],frame.shape[1])
			if len(array_boxes_detected)>0:
				# Both of our lists that will contain the centroïds coordonates and the ground points
				array_centroids,array_groundpoints = get_centroids_and_groundpoints(array_boxes_detected)
				# for i in array_centroids:
				# 	cv2.circle(frame, (i[0],i[1]), SMALL_CIRCLE, COLOR_GREEN, 5)
				# for j in array_groundpoints:
				# 	cv2.circle(frame,(j[0],j[1]), SMALL_CIRCLE, COLOR_RED, 5)
				# cv2.imshow("Check", frame)
				# cv2.waitKey(10)
				# Use the transform matrix to get the transformed coordonates
				transformed_downoids = compute_point_perspective_transformation(matrix,array_groundpoints)
				
				# # Show every point on the top view image 
				# for point in transformed_downoids:
				# 	x,y = point
				# 	cv2.circle(bird_view_img, (x,y), BIG_CIRCLE, COLOR_GREEN, 2)
				# 	cv2.circle(bird_view_img, (x,y), SMALL_CIRCLE, COLOR_GREEN, -1)

				# Check if 2 or more people have been detected (otherwise no need to detect)
				if len(transformed_downoids) >= 2:
					for index,downoid in enumerate(transformed_downoids):
						if not (downoid[0] > width or downoid[0] < 0 or downoid[1] > height+200 or downoid[1] < 0 ):
							cv2.rectangle(frame,(array_boxes_detected[index][1],array_boxes_detected[index][0]),(array_boxes_detected[index][3],array_boxes_detected[index][2]),COLOR_GREEN,2)
							cv2.circle(bird_view_img, (downoid[0],downoid[1]), BIG_CIRCLE, COLOR_GREEN, 2)
							cv2.circle(bird_view_img, (downoid[0],downoid[1]), SMALL_CIRCLE, COLOR_GREEN, -1)
							# cv2.imshow("FramebyFrame", frame)
							# cv2.waitKey(10)
					# Iterate over every possible 2 by 2 between the points combinations 
					list_indexes = list(itertools.combinations(range(len(transformed_downoids)), 2))
					for i,pair in enumerate(itertools.combinations(transformed_downoids, r=2)):
					# for i,pair in enumerate(itertools.combinations(array_centroids, r=2)):
						# Check if the distance between each combination of points is less than the minimum distance chosen
						distance_between_pair = math.sqrt( (pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2 )
						# print(distance_between_pair)	
						#Pairs with probability that will not maintain social distancing.
						if distance_between_pair <= int(distance_minimum)*2:
							#Creating new dictionary containing distances between pairs
							distance_between_pairs[f"pairs{i}"] = distance_between_pair
							#Checking and creating timer for pairs from distance_between_pairs
							if f"pairs{i}" in timer_for_each_pairs.keys():
								continue
							else:
								timer_for_each_pairs[f"pairs{i}"] = 0
						if distance_between_pair < int(distance_minimum):
							# Change the colors of the points that are too close from each other to red
							if not (pair[0][0] > width or pair[0][0] < 0 or pair[0][1] > height+200  or pair[0][1] < 0 or pair[1][0] > width or pair[1][0] < 0 or pair[1][1] > height+200  or pair[1][1] < 0):
								change_color_to_red_on_topview(pair)
								# Get the equivalent indexes of these points in the original frame and change the color to red
								index_pt1 = list_indexes[i][0]
								index_pt2 = list_indexes[i][1]
								cv2.rectangle(frame,(array_boxes_detected[index_pt1][1],array_boxes_detected[index_pt1][0]),(array_boxes_detected[index_pt1][3],array_boxes_detected[index_pt1][2]),COLOR_RED,2)
								cv2.rectangle(frame,(array_boxes_detected[index_pt2][1],array_boxes_detected[index_pt2][0]),(array_boxes_detected[index_pt2][3],array_boxes_detected[index_pt2][2]),COLOR_RED,2)
								# cv2.imshow("RedBoxed", frame)
								# cv2.waitKey(10)
						# else:
						# 	# change_color_to_green_on_topview(pair)
						# 	for point in transformed_downoids:
						# 		x,y = point
						# 		cv2.circle(bird_view_img, (x,y), BIG_CIRCLE, COLOR_GREEN, 2)
						# 		cv2.circle(bird_view_img, (x,y), SMALL_CIRCLE, COLOR_GREEN, -1)
	# print(distance_between_pairs)
	# print(timer_for_each_pairs)
	# print("\n")

	if len(distance_between_pairs)>0:
		threading1 = []
		for key,value in distance_between_pairs.items():
			t1 = threading.Thread(target = check_current_value, args = [key,value])
			t1.start()
			threading1.append(t1)
		for thread1 in threading1:
			t1.join()

		
		# threading2 = []
		# for k,v in timer_for_each_pairs.items():
		# 	t2 = threading.Thread(target = check_latest_time, args = [v])
		# 	t2.start()
		# 	threading2.append(t2)
		# for thread2 in threading2:
		# 	t2.join()
		t = timer_for_each_pairs.values()
		t_max = max(t)
		if t_max >= seconds and time_to_wait==0:
			threading.Thread(target = play_warning).start()
			threading.Thread(target= waiting_time).start()
			
		# check_to_play_warning()

		
		#Update dictionary to remove far away pairs. Check for it in only 10 loop to save computation power.
		if loop_count >=10:
			for k,v in distance_between_pairs.items():
				if v > int(distance_minimum)*2:
					del distance_between_pairs[k]
					del timer_for_each_pairs[k]
			loop_count = 0
	loop_count += 1

	# Draw the green rectangle to delimitate the detection zone
	draw_rectangle(corner_points)
	h,w,c = frame.shape
	# Show both images	
	bird_view = image_resize(bird_view_img, width=int(w/4))
	# cv2.imshow("Bird view", bird_view_img)
	# cv2.imshow("Original picture", frame)
	##Combine bird_view_img and frame in one.
	final_frame = combine_image(frame, bird_view)
	cv2.imshow("Combined View", final_frame)


	key = cv2.waitKey(1) & 0xFF

	# Write the both outputs video to a local folders
	# if output_video_1 is None and output_video_2 is None:
	# 	fourcc1 = cv2.VideoWriter_fourcc(*"MJPG")
	# 	output_video_1 = cv2.VideoWriter("../output/video.avi", fourcc1, 25,(frame.shape[1], frame.shape[0]), True)
	# 	fourcc2 = cv2.VideoWriter_fourcc(*"MJPG")
	# 	output_video_2 = cv2.VideoWriter("../output/bird_view.avi", fourcc2, 25,(bird_view_img.shape[1], bird_view_img.shape[0]), True)
	# elif output_video_1 is not None and output_video_2 is not None:
	# 	output_video_1.write(frame)
	# 	output_video_2.write(bird_view_img)

	if output_video_1 is None:
		fourcc1 = cv2.VideoWriter_fourcc(*"MJPG")
		output_video_1 = cv2.VideoWriter("../output/video.avi", fourcc1, 25,(final_frame.shape[1], final_frame.shape[0]), True)
	elif output_video_1 is not None:
		output_video_1.write(final_frame)

	# Break the loop
	if key == ord("q"):
		break
