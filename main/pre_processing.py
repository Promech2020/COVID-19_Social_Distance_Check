from main_func import start_checking
from final_windows import Final
from tf_model_object_detection import Model 
from mutagen.mp3 import MP3
from colors import bcolors
import time
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def check_social_distance(v_path, min_dist, wait_time_before, wait_time_between, out_frame, a_path, cam_distance):
	start = time.time()
	video_path = select_video(v_path)
	minimum_distance = get_minimum_distance(min_dist, cam_distance, out_frame)
	seconds = wait_to_play_warning(wait_time_before)
	waits = wait_between_warning(wait_time_between)
	frame_size = refine_frame_size(out_frame)
	soundfile = select_audio(a_path)
	#Get length of audio file
	audio = MP3(soundfile)
	audio_file_length = audio.info.length
	target_distance = get_target_distance(cam_distance)
	model = get_model()
	start_checking(start, video_path, minimum_distance, seconds, waits, frame_size, soundfile, audio_file_length, target_distance, model)
	end = time.time()
	print(f"Runtime of the program is {end - start}")


######################################### 
#		     Select the video 			#
#########################################
def select_video(video_name):
	if video_name == "":
		video_p="../input_video/PETS2009_5S.avi" 
	elif video_name == "WebCam":
		video_p = 0
	else :
		video_p = video_name
	return video_p

######################################### 
#		    Minimal distance			#
#########################################
def get_minimum_distance(min, dist, fs):
	min_value = float(min.split(" ")[0])
	frame_w = int(fs.split(" ")[0])/100
	minimum = min_value*int(dist)* frame_w
	return minimum

######################################### 
#		    Time to wait			#
#########################################
def wait_to_play_warning(sec):
	#Take input for how many seconds do you want to wait when two people are close enough
	seconds = int(sec.split(" ")[0])
	return seconds

######################################### 
#		    Wait between Warning		#
#########################################
def wait_between_warning(secs):
	#Take input for how many seconds do you want to wait after playing warning.
	wait = int(secs.split(" ")[0])
	return wait

######################################### 
#		    Output Frame Size		#
#########################################
def refine_frame_size(size):
	#Take input for how many seconds do you want to wait after playing warning.
	frame_wide = int(size.split(" ")[0])
	return frame_wide

######################################### 
#		    Select Audio File		#
#########################################
def select_audio(audio):
	#Take input for how many seconds do you want to wait after playing warning.
	if audio == "":
		sound = "../sound/covid_message.mp3"
	else:
		sound = audio
	return sound
	
######################################### 
#		    Camera Target Distance		#
#########################################
def get_target_distance(distance):
	#Take input for how many seconds do you want to wait after playing warning.
	dist = int(distance)
	return dist

######################################### 
#		     Select the model 			#
#########################################
def get_model():
	try:
		model_path="../models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb" 

		print(bcolors.WARNING + " [ Loading the TENSORFLOW MODEL ... ]"+bcolors.ENDC)
		m = Model(model_path)
		print(bcolors.OKGREEN +"Done : [ Model loaded and initialized ] ..."+bcolors.ENDC)
		return m
	except:
		Final("Model load Failed","Please check your model file and folder.")
		sys.exit(0)