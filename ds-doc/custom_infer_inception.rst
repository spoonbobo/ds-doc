Custom Model - FasterRCNN (Triton)
==================================

Sample deployment of Inception network in Triton Inference Server on AGX Xavier. For more details, see `FasterRCNN repo <https://github.com/NVIDIA-AI-IOT/deepstream_triton_model_deploy/tree/master/faster_rcnn_inception_v2>`_ for more details.

Prerequisites
-------------

* Jetson AGX Xavier
* DeepStream SDK 5.1

*Follow this sample in DeepStream NGC Container or on Nvidia AGX Xavier.*

1. Clone the repo
-----------------
::

	git clone https://github.com/NVIDIA-AI-IOT/deepstream_triton_model_deploy
	cd deepstream_triton_model_deploy/faster_rcnn_inception_v2

2. Download the models
----------------------

::

	wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
	tar xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

3. 
