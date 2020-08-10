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
pip install soundfile
conda install -c conda-forge python-sounddevice
```
### PyQt5 and PyQt5-tools
```
pip install pyqt5
```
### VidGear
```
pip install vidgear
```

# Download Tensorflow models

In my project I used the faster_rcnn_inception_v2_coco model. 
You can also download this model and several others from the [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). 
Just download all the models you want to try out, put them in the models folder and unzip them. For example :
```bash
tar -xvzf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```

# Run project
### Start social distancing detection
Run 
```bash
python Social_Distance_Check.py
```
You will be asked as inputs :
- Browse for input Video or WebCam or provide online video link.
- The minimum distance to maintain between 2 persons.
- Time to wait before playing alert sound.
- Time to wait between playing alert sound.
- Output frame size.
- Audio file(.wav format) to play as warning.

* Note: If you did not provide any information. It will run with defaults. *

# Outputs
Video output will be stored in the output file.

# In Progress

- Relation between minimum distance input, frame size and pixel value. 
- User input for warning message.
- Detect whether WebCam is connected or not.
- Show output messages in message boxes after completion of job or if some errors occurs.
