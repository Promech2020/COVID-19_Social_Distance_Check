from bird_view_transfo_functions import compute_perspective_transform,compute_point_perspective_transformation
from helping_functions import image_resize, create_blank, combine_image
from tf_model_object_detection import Model 
from colors import bcolors
from mutagen.mp3 import MP3
from playsound import playsound
import numpy as np
import itertools
import threading
import imutils
import time
import math
import yaml
import cv2
import os

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


def get_centroids(array_boxes_detected):
	"""
	For every bounding box, compute the centroid and the point located on the bottom center of the box
	@ array_boxes_detected : list containing all our bounding boxes 
	"""
	array_centroids = [] # Initialize empty centroid.
	for index,box in enumerate(array_boxes_detected):
		# Get the centroid
		centroid = get_points_from_box(box)
		array_centroids.append(centroid)
	return array_centroids


def get_points_from_box(box):
	"""
	Get the center of the bounding.
	@ param = box : 2 points representing the bounding box
	@ return = centroid (x1,y1)
	"""
	# Center of the box x = (x1+x2)/2 et y = (y1+y2)/2
	center_x = int(((box[1]+box[3])/2))
	center_y = int(((box[0]+box[2])/2))

	return (center_x,center_y)


# def change_color_to_red_on_topview(pair):
# 	"""
# 	Draw red circles for the designated pair of points 
# 	"""
# 	cv2.circle(bird_view_img, (pair[0][0],pair[0][1]), BIG_CIRCLE, COLOR_RED, 2)
# 	cv2.circle(bird_view_img, (pair[0][0],pair[0][1]), SMALL_CIRCLE, COLOR_RED, -1)
# 	cv2.circle(bird_view_img, (pair[1][0],pair[1][1]), BIG_CIRCLE, COLOR_RED, 2)
# 	cv2.circle(bird_view_img, (pair[1][0],pair[1][1]), SMALL_CIRCLE, COLOR_RED, -1)

# def change_color_to_green_on_topview(pair):
# 	"""
# 	Draw red circles for the designated pair of points 
# 	"""
# 	cv2.circle(bird_view_img, (pair[0][0],pair[0][1]), BIG_CIRCLE, COLOR_GREEN, 2)
# 	cv2.circle(bird_view_img, (pair[0][0],pair[0][1]), SMALL_CIRCLE, COLOR_GREEN, -1)
# 	cv2.circle(bird_view_img, (pair[1][0],pair[1][1]), BIG_CIRCLE, COLOR_GREEN, 2)
# 	cv2.circle(bird_view_img, (pair[1][0],pair[1][1]), SMALL_CIRCLE, COLOR_GREEN, -1)

# def draw_rectangle(corner_points):
# 	"""
# 	Draw rectangle box over the delimitation area.
# 	"""
# 	cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[1][0], corner_points[1][1]), COLOR_BLUE, thickness=1)
# 	cv2.line(frame, (corner_points[1][0], corner_points[1][1]), (corner_points[3][0], corner_points[3][1]), COLOR_BLUE, thickness=1)
# 	cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[2][0], corner_points[2][1]), COLOR_BLUE, thickness=1)
# 	cv2.line(frame, (corner_points[3][0], corner_points[3][1]), (corner_points[2][0], corner_points[2][1]), COLOR_BLUE, thickness=1)


######################################### 
#		     Select the model 			#
#########################################
model_path="../models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb" 

print(bcolors.WARNING + " [ Loading the TENSORFLOW MODEL ... ]"+bcolors.ENDC)
model = Model(model_path)
print(bcolors.OKGREEN +"Done : [ Model loaded and initialized ] ..."+bcolors.ENDC)

######################################### 
#		     Select the video 			#
#########################################

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

frame_per_second = input("How many frames per second is your video/webcam (Default is 30): ")
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
		frame = image_resize(frame, width = 600)
		height, width, channels = frame.shape

		# Make the predictions for this frame
		(boxes, scores, classes) =  model.predict(frame)

		if len(boxes)>0:
			box_and_centroid = dict()
			# Get the human detected in the frame and return the 2 points to build the bounding box  
			array_boxes_detected = get_human_box_detection(boxes,scores[0].tolist(),classes[0].tolist(),frame.shape[0],frame.shape[1])
			if len(array_boxes_detected)>0:
				# Both of our lists that will contain the centroïds coordonates and the ground points
				array_centroids = get_centroids(array_boxes_detected)


				#Only for checking
				# for i in array_centroids:
				# 	cv2.circle(frame, (i[0],i[1]), SMALL_CIRCLE, COLOR_GREEN, 5)
				# for j in array_groundpoints:
				# 	cv2.circle(frame,(j[0],j[1]), SMALL_CIRCLE, COLOR_RED, 5)
				# cv2.imshow("Check", frame)
				# cv2.waitKey(10)
				# Use the transform matrix to get the transformed coordonates
				# transformed_downoids = compute_point_perspective_transformation(matrix,array_groundpoints)
				
				# # Show every point on the top view image 
				# for point in transformed_downoids:
				# 	x,y = point
				# 	cv2.circle(bird_view_img, (x,y), BIG_CIRCLE, COLOR_GREEN, 2)
				# 	cv2.circle(bird_view_img, (x,y), SMALL_CIRCLE, COLOR_GREEN, -1)

				# Check if 2 or more people have been detected (otherwise no need to detect)
				if len(array_centroids) >= 2:
					# for index,centroids in enumerate(array_centroids):
					# 	if not (centroids[0] > width or centroids[0] < 0 or centroids[1] > height+200 or centroids[1] < 0 ):
					# 		cv2.rectangle(frame,(array_boxes_detected[index][1],array_boxes_detected[index][0]),(array_boxes_detected[index][3],array_boxes_detected[index][2]),COLOR_GREEN,2)
							# cv2.circle(bird_view_img, (downoid[0],downoid[1]), BIG_CIRCLE, COLOR_GREEN, 2)
							# cv2.circle(bird_view_img, (downoid[0],downoid[1]), SMALL_CIRCLE, COLOR_GREEN, -1)
							# cv2.imshow("FramebyFrame", frame)
							# cv2.waitKey(10)
					# Iterate over every possible 2 by 2 between the points combinations 
					# list_indexes = list(itertools.combinations(range(len(array_centroids)), 2))
					# print(list_indexes)
					close_pairs = []
					for i,pair in enumerate(itertools.combinations(array_centroids, r=2)):
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

						# index_pt1 = list_indexes[i][0]
						# index_pt2 = list_indexes[i][1]
						if distance_between_pair < int(distance_minimum):
							close_pairs.append(pair)
							for i in array_centroids:
								box_and_centroid[array_centroids[i]] = array_boxes_detected[i]
							# Change the colors of the points that are too close from each other to red
							# if not (pair[0][0] > width or pair[0][0] < 0 or pair[0][1] > height+200  or pair[0][1] < 0 or pair[1][0] > width or pair[1][0] < 0 or pair[1][1] > height+200  or pair[1][1] < 0):
								# Get the equivalent indexes of these points in the original frame and change the color to red
							# cv2.rectangle(frame,(array_boxes_detected[index_pt1][1],array_boxes_detected[index_pt1][0]),(array_boxes_detected[index_pt1][3],array_boxes_detected[index_pt1][2]),COLOR_RED,2)
							# cv2.rectangle(frame,(array_boxes_detected[index_pt2][1],array_boxes_detected[index_pt2][0]),(array_boxes_detected[index_pt2][3],array_boxes_detected[index_pt2][2]),COLOR_RED,2)
						# else:
						# 	cv2.rectangle(frame,(array_boxes_detected[index_pt1][1],array_boxes_detected[index_pt1][0]),(array_boxes_detected[index_pt1][3],array_boxes_detected[index_pt1][2]),COLOR_GREEN,2)
						# 	cv2.rectangle(frame,(array_boxes_detected[index_pt2][1],array_boxes_detected[index_pt2][0]),(array_boxes_detected[index_pt2][3],array_boxes_detected[index_pt2][2]),COLOR_GREEN,2)
					flat_list = []
					for sublist in close_pairs:
						for item in sublist:
							flat_list.append(item)
					common_close_pairs = list(set(flat_list))	
					for each in common_close_pairs:
						cv2.rectangle(frame,(box_and_centroid.keys[0],box_and_centroid.keys[1]),(box_and_centroid.keys[2],box_and_centroid.keys[3]),COLOR_GREEN,2)


				# else:
				# 	cv2.rectangle(frame,(array_boxes_detected[0][1],array_boxes_detected[0][0]),(array_boxes_detected[1][0],array_boxes_detected[1][1]),COLOR_GREEN,2)
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

		t = timer_for_each_pairs.values()
		t_max = max(t)
		if t_max >= seconds and time_to_wait==0:
			threading.Thread(target = play_warning).start()
			threading.Thread(target= waiting_time).start()

		#Update dictionary to remove far away pairs. Check for it in only 10 loop to save computation power.
		if loop_count >=10:
			for k,v in distance_between_pairs.items():
				if v > int(distance_minimum)*2:
					del distance_between_pairs[k]
					del timer_for_each_pairs[k]
			loop_count = 0
	loop_count += 1

	cv2.imshow("Final View", frame)

	key = cv2.waitKey(1) & 0xFF

	if output_video_1 is None:
		fourcc1 = cv2.VideoWriter_fourcc(*"MJPG")
		output_video_1 = cv2.VideoWriter("../output/video.avi", fourcc1, 25,(frame.shape[1], frame.shape[0]), True)
	elif output_video_1 is not None:
		output_video_1.write(frame)

	# Break the loop
	if key == ord("q"):
		break
