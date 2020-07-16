from helping_functions import image_resize, create_blank, get_human_box_detection, get_centroids, get_points_from_box
from tf_model_object_detection import Model 
from colors import bcolors
import numpy as np
import itertools
import imutils
import time
import math
import yaml
import cv2
import os

COLOR_BLUE = (255, 0, 0)

##Required Variables
distance_between_pair = dict()

######################################### 
#		     Select the model 			#
#########################################
def get_model():
	model_path="../models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb" 

	print(bcolors.WARNING + " [ Loading the TENSORFLOW MODEL ... ]"+bcolors.ENDC)
	m = Model(model_path)
	print(bcolors.OKGREEN +"Done : [ Model loaded and initialized ] ..."+bcolors.ENDC)
	return m
######################################### 
#		     Select the video 			#
#########################################
def get_image():
	image_name = input("Enter the path of the image: ")
	if image_name == "":
		image_p = "../input_image/cctv_two_people.jpg" 
	else :
		image_p = image_name
	return image_p


def start_checking(image_path, model,w):
	######################################################
	# 				START THE VIDEO STREAM               #
	######################################################
	frame = cv2.imread(image_path)
	# Resize the image to the correct size
	frame = image_resize(frame, width = int(w))
	height, width, channels = frame.shape

	# Make the predictions for this frame
	(boxes, scores, classes) =  model.predict(frame)

	if len(boxes)>0:
		# Get the human detected in the frame and return the 2 points to build the bounding box  
		array_boxes_detected = get_human_box_detection(boxes,scores[0].tolist(),classes[0].tolist(),frame.shape[0],frame.shape[1])

		if len(array_boxes_detected)>0:
			# Both of our lists that will contain the centroïds coordonates and the ground points
			array_centroids = get_centroids(array_boxes_detected)

			# Check if 2 or more people have been detected (otherwise no need to detect)
			if len(array_centroids) >= 2:
				for i,items in enumerate(array_boxes_detected):
					cv2.rectangle(frame,(array_boxes_detected[i][1],array_boxes_detected[i][0]),(array_boxes_detected[i][3],array_boxes_detected[i][2]),COLOR_BLUE,2)

				for i,pair in enumerate(itertools.combinations(array_centroids, r=2)):
				# for i,pair in enumerate(itertools.combinations(array_centroids, r=2)):
					# Check if the distance between each combination of points is less than the minimum distance choosen
					dbp = math.sqrt( (pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2 )	
					distance_between_pair[f"pair{i}"] = dbp
		else:
			print("There is only one person detected.")
	else:
		print("No human detected.")


#Start Execution
w = input("Frame size: ")
print(f"Frame Size: {w}")
model = get_model()
image_path = get_image()
start_checking(image_path, model,w)
print(distance_between_pair)
