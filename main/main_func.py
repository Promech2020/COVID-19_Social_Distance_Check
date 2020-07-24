from helping_functions import image_resize, create_blank, get_human_box_detection, get_centroids, get_points_from_box, rescale_image
from final_windows import Final
from vidgear.gears import CamGear
from playsound import playsound
import numpy as np
import itertools
import threading
import imutils
import time
import math
import yaml
import cv2
import sys
import os

#Defining red color rgb value
COLOR_RED = (0, 0, 255)

#Dictionary to save distance between pairs
distance_between_pairs = dict()

#Dictionary to start timer when the distance between pairs is less than the minimum distance defined.
timer_for_each_pairs = dict()

time_to_wait = 0



def start_checking(start_time, video_path, minimum_distance, seconds, waits, frame_width, audio_path, audio_length, camera_target, model):
	# event = threading.Event()
	good_to_run = False
	good_to_write = True
	output_video_1 = None
	loop_count = 0
# frame_count = 0
	######################################################
	# 				START THE VIDEO STREAM               #
	######################################################
	if video_path.startswith("http"):
		try:
			vs = CamGear(source=video_path, y_tube =True,  time_delay=1, logging=True).start() 
			# event.set()
			frame_per_seconds = 30
			good_to_run = True
			good_to_write = True
			
		except:
			# event.clear()
			good_to_run = False
			good_to_write = False
			end = time.time()
			time_elapsed = int(end - start_time)
			print(f"Time consumed: {time_elapsed} seconds.")
			print("Youtube Video Load Failed. Online video link stopped working.")
			# Final("Youtube Video Load Failed.",f"Time consumed: {time_elapsed} seconds. \n Online video link stopped working.")
			
	elif video_path == 0:
		try:
			# event.set()
			vs = cv2.VideoCapture(0)
			frame_per_seconds = int(vs.get(cv2.CAP_PROP_FPS))
			good_to_run = True
			good_to_write = True
		except:
			# event.clear()
			good_to_run = False
			good_to_write = False
			end = time.time()
			time_elapsed = int(end - start_time)
			# Final("WebCam Failed", f"Time consumed: {time_elapsed} \n Webcam not connected." )
			
	else:	
		try:
			# event.set()
			vs = cv2.VideoCapture(video_path)
			frame_per_seconds = int(vs.get(cv2.CAP_PROP_FPS))
			good_to_run = True
			good_to_write = True
		except:
			# event.clear()
			good_to_run = False
			good_to_write = False
			end = time.time()
			time_elapsed = int(end - start_time)
			print(f"Time consumed: {time_elapsed} seconds.")
			print("Webcam not connected.")
			os._exit(0)
			# Final("Video Load Failed", f"Time consumed: {time_elapsed} seconds. \n Something wrong in videopath provided." )

	# Loop until the end of the video stream
	while True and good_to_run == True:	
		if video_path.startswith("http"):
			try:
				frame = vs.read()
				# event.set()
				good_to_run = True
				good_to_write = True
			except:
				# event.clear()
				good_to_run = False
				good_to_write = False
				end = time.time()
				time_elapsed = int(end - start_time)
				print(f"Time consumed: {time_elapsed} seconds.")
				print("Online video link stopped working.")
				os._exit(0)
				# Final("Youtube Video Read Failed",f"Time consumed: {time_elapsed} seconds. \n Online video link stopped working.")
		else:
			try:
				(frame_exist, frame) = vs.read()
				# event.set()
				good_to_run = True
				good_to_write = True
			except:
				# event.clear()
				good_to_run = False
				good_to_write = False
				end = time.time()
				time_elapsed = int(end - start_time)
				print(f"Time consumed: {time_elapsed} seconds.")
				print("Could not get frames from video.")
				os._exit(0)
				# Final("Video load Failed.",f"Time consumed: {time_elapsed} seconds. \n Could not get frames from video.")
			

		if frame is None:
			break
		else:
			# event.set()
			good_to_run = True
			good_to_write = True
			# frame_count += 1
			# Resize the image to the correct size
			frame = image_resize(frame, width = frame_width)

			# Make the predictions for this frame
			(boxes, scores, classes) =  model.predict(frame)

			if len(boxes)>0:
				
				# Get the human detected in the frame and return the 2 points to build the bounding box  
				array_boxes_detected = get_human_box_detection(boxes,scores[0].tolist(),classes[0].tolist(),frame.shape[0],frame.shape[1])

				if len(array_boxes_detected)>0:
					# Both of our lists that will contain the centroïds coordonates and the ground points
					array_centroids = get_centroids(array_boxes_detected)
					box_and_centroid = list(zip(array_centroids,array_boxes_detected))

					# Check if 2 or more people have been detected (otherwise no need to detect)
					if len(array_centroids) >= 2:
						close_pairs = []
						for i,pair in enumerate(itertools.combinations(array_centroids, r=2)):
						# for i,pair in enumerate(itertools.combinations(array_centroids, r=2)):
							# Check if the distance between each combination of points is less than the minimum distance chosen
							distance_between_pair = math.sqrt( (pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2 )
							# print(distance_between_pair)	
							#Pairs with probability that will not maintain social distancing.
							if distance_between_pair <= int(minimum_distance)*2:
								#Creating new dictionary containing distances between pairs
								distance_between_pairs[f"pairs{i}"] = distance_between_pair
								#Checking and creating timer for pairs from distance_between_pairs
								if f"pairs{i}" not in timer_for_each_pairs.keys():
									timer_for_each_pairs[f"pairs{i}"] = 0
									

							if distance_between_pair < int(minimum_distance):
								close_pairs.append(pair)
						
						flat_list = []
						for sublist in close_pairs:
							for item in sublist:
								flat_list.append(item)
						common_close_pairs = list(set(flat_list))
						# print(common_close_pairs)	
						boxes_to_make_red = []
						for ccp in common_close_pairs:
							for b_and_c in box_and_centroid:
								if ccp == b_and_c[0]:
									boxes_to_make_red.append(b_and_c[1]) 
						# print(boxes_to_make_red)
						for i,items in enumerate(boxes_to_make_red):
							first_point = boxes_to_make_red[i][0]
							second_point = boxes_to_make_red[i][1]
							third_point = boxes_to_make_red[i][2]
							fourth_point = boxes_to_make_red[i][3]
							cv2.rectangle(frame,(second_point,first_point),(fourth_point,third_point),COLOR_RED,2)

						box_and_centroid.clear()
						close_pairs.clear()
						flat_list.clear()
						common_close_pairs.clear()
						boxes_to_make_red.clear()
			# else:
			# 	print(f"Something is wrong in frame {frame_count}.")

		if len(distance_between_pairs)>0:
			threading1 = []
			for key,value in distance_between_pairs.items():
				t1 = threading.Thread(target=check_current_value, args=[key,value, minimum_distance], daemon = True)
				t1.start()
				threading1.append(t1)
			for thread1 in threading1:
				t1.join()

			t = timer_for_each_pairs.values()
			t_max = max(t)
			if t_max >= seconds and time_to_wait==0:
				threading.Thread(target = play_warning, args = [start_time, audio_path, frame_per_seconds], daemon = True).start()
				threading.Thread(target= waiting_time, args=[audio_length, waits, frame_per_seconds], daemon = True).start()

			#Update dictionary to remove far away pairs. Check for it in only 10 loop to save computation power.
			if loop_count >=10:
				for k,v in distance_between_pairs.items():
					if v > int(minimum_distance)*2:
						del distance_between_pairs[k]
						del timer_for_each_pairs[k]
				loop_count = 0
		loop_count += 1

		cv2.imshow("Output", frame)
		# cv2.namedWindow("Final_Output", cv2.WND_PROP_FULLSCREEN)
		# cv2.setWindowProperty("Final_Output",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
		# cv2.imshow("Final_Output", frame)
		key = cv2.waitKey(1) & 0xFF
		if video_path != 0:
			if output_video_1 is None:
				fourcc1 = cv2.VideoWriter_fourcc(*"MJPG")
				output_video_1 = cv2.VideoWriter("../output_video/video.avi", fourcc1, 25,(frame.shape[1], frame.shape[0]), True)
			elif output_video_1 is not None:
				output_video_1.write(frame)

			# Break the loop
			if key == ord("q"):
				break

	if video_path != 0 and good_to_write == True:
		# while cv2.getWindowProperty('Output', cv2.WND_PROP_VISIBLE)==1.0:
		# 	cv2.destroyWindow("Output")
		end = time.time()
		time.sleep(1)
		time_elapsed = int(end -  start_time)
		print(f"Time consumed: {time_elapsed} seconds.")
		print("Successful execution. Video saved in output_folder.")
		os._exit(0)
		# Final("Success", f"Time consumed: {time_elapsed} seconds \n Successful execution. Video saved in output_video folder.")
	# complete_msg(video_path, good_to_write, start_time, frame_per_seconds)



def check_current_value(key,value, minimum_distance):
	if value < float(minimum_distance):
		time.sleep(1)
		timer_for_each_pairs[key] += 1
	else:
		timer_for_each_pairs[key] = 0

def play_warning(start_time, soundfile, fps):
	try:    
		playsound(soundfile)
		# eve.set()
		good_to_run = True
		good_to_write = True
	except:
		# eve.clear()
		good_to_run = False
		good_to_write = False
		end = time.time()
		time_elapsed = int(end - start_time)
		# print(threading.enumerate())
		# os._exit(0)
		# t1.stop()
		# print(threading.enumerate())
		# threading.Thread(target = close_cv2, args = [fps]).start()
		# Final("Playing Alert Message Failed", f"Time consumed: {time_elapsed} \n Could not play the sound.")
		# t_stop = threading.Thread(target = exit_all, daemon = True)
		# t_msg = threading.Thread(target = Final, args = ["Warning Play Failed", f"Time consumed: {time_elapsed} \n Could not play the sound."], daemon = True)
		# t_stop.start()
		# t_msg.start()
		print(f"Time consumed: {time_elapsed} seconds.")
		print("Could not play the sound.")
		os._exit(0)
	
# def exit_all():
# 	os._exit(0)

def waiting_time(audio, waits, frame_per_seconds):
	global time_to_wait
	for i in range(int(audio)+waits):
		for j in range(frame_per_seconds*2):
			to_sleep = 1/(frame_per_seconds*2)
			time.sleep(to_sleep)
			time_to_wait += to_sleep
	time_to_wait=0

# def close_cv2(frames_seconds):
# 	for i in range(frames_seconds):
# 		current_state = cv2.getWindowProperty('Output', cv2.WND_PROP_VISIBLE)
# 		if current_state == 1.0:
# 			cv2.destroyWindow("Output")
# 		else:
# 			time.sleep(0.5)
# def close_background():
# 	while cv2.getWindowProperty('Output', cv2.WND_PROP_VISIBLE)==1.0:
# 		cv2.destroyWindow("Output")

# def complete_msg(vp, gtw, start, fps):
# 	if vp != 0 and gtw == True:
# 		while cv2.getWindowProperty('Output', cv2.WND_PROP_VISIBLE)==1.0:
# 			cv2.destroyWindow("Output")
# 		end = time.time()
# 		time.sleep(1)
# 		time_elapsed = int(end -  start)
# 		Final("Success", f"Time consumed: {time_elapsed} seconds. \n Successful execution. Video saved in output_video folder.")