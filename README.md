# Cinder-Yolo

Cinder block for running Darknet's YoloV3 object detection. This blocks uses the following fork of Darknet https://github.com/AlexeyAB/darknet

Tested on macOS with Cuda 10 and cuDNN 7.3.1.

Assuming you have cloned the block inside Cinder's block directory and you have already built Cinder in debug mode run:

`cd Cinder-Yolo/samples/BasicSample/proj/cmake && mkdir build && cd build && cmake .. && make -j4`
