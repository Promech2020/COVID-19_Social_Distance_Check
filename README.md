# covid-social-distancing-detection

This project is a social distancing detector implemented in Python with OpenCV and Tensorflow. Original project is https://github.com/basileroth75/covid-social-distancing-detection
We have modified it to get more user input and play alert sound when people are not maintaining social distancing for certain period of time.
The result that can be obtained is the following :

![](/img/result.gif)

# Installation

### Python
This project is tested under python version 3.6

### OpenCV
If you are working under a Linux distribution or a MacOS, use this [tutorial](https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/) from Adrian Rosebrock to install this library.
Or following command should work.
```
pip install opencv-python
```
### Tensorflow

```
pip install tensorflow==2.1
pip install tensorflow-gpu==2.1
```
### imutils
```
pip install imutils
```

### Yaml
```
pip install PyYaml
```

### playsound
```
pip install playsound
```

### mutagen
```
pip install mutagen
```

# Download Tensorflow models

In my project I used the faster_rcnn_inception_v2_coco model. I could not upload it to github because it is to heavy. You can download this model and several others from the [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). 
Just download all the models you want to try out, put them in the models folder and unzip them. For example :
```bash
tar -xvzf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```

# Run project

### Start social distancing detection
Run 
```bash
python social_distanciation_video_detection.py
```
You will be asked as inputs :
- The name of the video (default value PETS2009.avi) or "0" for webcam.
- The distance (in pixels between 2 persons).
- Time to wait before playing alert sound.
- Frame per second of your video/cctv footage.

# Outputs
Video output (combined of normal frame and bird eye view) will be stored in the output file.
