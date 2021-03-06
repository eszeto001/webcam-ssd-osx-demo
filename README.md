# webcam-ssd-osx-demo

This demonstrate a Single Shot Detector with a iMac webcam.

## Installation

I currently use OSX 10.3.x.
Python 3.6.4 via Anaconda is installed.

So the additional that need to be installed are the following:

---------------------
[Tensorflow](https://www.tensorflow.org/install/install_mac)

```
$ pip install --upgrade pip
$ pip install tensorflow 
```

---------------------
[OpenCV](https://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/)

With anaconda, you should be able to do the following:

```
$ conda install opencv
```

---------------------
Next you should install the tensorflow models somewhere.

$ git clone https://github.com/tensorflow/models

There seems to be a bug in the research/object_detection code.
It appears someone was cleaning out some files,
but broke the object_detection_tutorial.ipynb.

To fix this, the original file was retained from an earlier version.
You can copy this back in to the appropriate directory.

```
$ cd <webcam-ssd-osx-demo>

$ cp fix/string_int_label_map_pb2.py <tf-models-dir>/research/object_detection/protos

```

## Demo

The basic code for a "hello world" webcam is fairly simple.
(This is perhaps not the most robust code.  But for demonstration
 purposes, removing all the cruft, it should suffice.)

```python
import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    cv2.imshow('frame', frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): break

cap.release()
cv2.destroyAllWindows()

```

You should be able to run this, and have it work,
as a quick test that openCV is installed correctly,
and works with your webcam.


Next want draw bounding boxes around the objects, and
label them. We modify the original Tensorflow object detection tutorial for this.

[Object Detection Tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb)

The variable MODELS_DIR should be manually edited to 
point to the location where you've installed the Tensorflow
models/ directory.

To run

$ python webcam-ssd.py

I got something like the following:

<img style="float: center;" src="./images/cup.png" />


You shoud be able to find the list of supported classes in 

```
<tf-models-dir>/research/object_detection/data/mscoco_label_map.pbtxt
```

