Custom Model - Yolov4
=====================

This sample deployment of Yolov4 detection model describes how can we export Yolov4 detection model (with pretrain darknet weights as backbone) to ONNX model, and then convert it to TRT inference engine and deploy the engine on DeepStream. See `GitHub repository <https://github.com/NVIDIA-AI-IOT/yolov4_deepstream>`_ for more details of this deployment of Yolov4 detection model on Nvidia AGX Xavier.

Prerequisites
-------------

* Download pretrained darknet model weights

	* You can either donwload weights `here <https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT>`_, or

	* Pull the PyTorch image (based on PyTorch NGC container) containing model weights and GitHub repo directly::

		docker pull spoonnvidia/ds-demo:yolov4-trt-v1

	* It's recommended to export the pretrained darknet weights in PyTorch NGC container to meet dependencies requirement.

* Create PyTorch container

	* If you download weights by yourself, please clone the `GitHub repo <https://github.com/NVIDIA-AI-IOT/yolov4_deepstream>`_ put the darknet weights under :file:`pytorch-YOLOv4/models`, and create the container::

		docker run --gpus all -it --rm -v $(pwd)/pytorch-YOLOv4:/workspace/pytorch-YOLOv4 -v $(pwd)/outputs:/workspace/outputs nvcr.io/nvidia/pytorch:21.07-py3

	* If you pull the image from *spoonnvidia/ds-demo:yolov4-trt-v1*, create the container::

		docker run --gpus all -it --rm -v $(pwd)/outputs:/workspace/outputs spoonnvidia/ds-demo:yolov4-trt-v1

0. Run inference using pretrained darknet weights
-------------------------------------------------

With the downloaded `darknet weights <https://github.com/AlexeyAB/darknet>`_ trained with COCO dataset, you can run inference on an image::

	cd /workspace/pytorch-YOLOv4
	python demo.py -cfgfile cfg/yolov4-custom.cfg -weightfile models/yolov4.weights -imgfile <imgFile>

1. Convert the pretrained darknet weight to ONNX model
------------------------------------------------------

We use prepared python script :file: `demo_darknet2onnx.py` to convert :file:`models/yolov4.weights` with yolov4 detection backbone configuration file :file:`cfg/yolov4-custom.cfg` to run inference ONNX model.

::

	cd /workspace/pytorch-YOLOv4
	python demo_darknet2onnx.py cfg/yolov4-custom.cfg models/yolov4.weights data/dog.jpg 4

After running above block, you should see ONNX files with filenames similar to :file:`yolov4_1_3_608_608_static.onnx` or :file:`yolov4_4_3_608_608_static.onnx` in the same directory. We move the ONNX files and DeepStream-related files :file:`/workspace/pytorch-YOLOv4/DeepStream` to output directory :file:`/workspace/outputs`::

	cd /workspace/pytorch-YOLOv4
	mv DeepStream /workspace/outputs
	mv yolov4_1_3_608_608_static.onnx /workspace/outputs

2. Transfer ONNX file to AGX Xavier 
-----------------------------------

When ONNX files are ready, you can quit the container and ready to transfer the model outputs to AGX Xavier. Simply zip the outputs directory into :file:`yolov4-onnx.zip` and transfer :file:`yolov4-onnx.zip` to AGX Xavier::

	zip -r yolov4-onnx.zip outputs
	scp -r yolov4-onnx.zip <agx-xavier-user>@<agx-xavier-ip>:/home/<agx-xavier-user>

*NOTE: It's strongly recommended for you to convert ONNX to TRT engine on AGX Xavier (or where you deploy TRT engine) to prevent TRT serialization failure*


3. Convert ONNX file to TRT engine
----------------------------------

Upon successful transfer to AGX Xavier, we first unzip :file:`/home/<agx-xavier-user>/yolov4-onnx.zip`::

	cd /home/<agx-xavier-user>
	unzip yolov4-onnx.zip
	cd outputs

::

	.
	├── DeepStream
	│   ├── config_infer_primary_yoloV4.txt
	│   ├── deepstream_app_config_yoloV4.txt
	│   ├── labels.txt
	│   ├── nvdsinfer_custom_impl_Yolo
	│   │   ├── kernels.cu
	│   │   ├── Makefile
	│   │   ├── nvdsinfer_yolo_engine.cpp
	│   │   ├── nvdsparsebbox_Yolo.cpp
	│   │   ├── Readme.md
	│   │   ├── trt_utils.cpp
	│   │   ├── trt_utils.h
	│   │   ├── yolo.cpp
	│   │   ├── yolo.h
	│   │   ├── yoloPlugins.cpp
	│   │   └── yoloPlugins.h
	│   └── Readme.md
	└── yolov4_1_3_608_608_static.onnx


Convert ONNX to TRT engine using :file:`/usr/src/tensorrt/bin/trtexec`::

	/usr/src/tensorrt/bin/trtexec --onnx=yolov4_1_3_608_608_static.onnx --explicitBatch --saveEngine=/home/<agx-xavier-user>/outputs/DeepStream/primary-detector.trt --workspace=1000 --fp16

The process takes a while... grab a coffee.

4. Custom implementation of nvinfer
-----------------------------------

Create custom nvinfer plugin::

	cp -r /home/<agx-xavier-user>/outputs/DeepStream /opt/nvidia/deepstream/deepstream-5.1/sources
	cd /opt/nvidia/deepstream/deepstream-5.1/sources/DeepStream/nvdsinfer_custom_impl_Yolo
	export CUDA_VER=10.2
	make

A library file called :file:`libnvdsinfer_custom_impl_Yolo.so` will be generated. Move this file to :file:`/home/<agx-xavier-user>/outputs/DeepStream/nvdsinfer_custom_impl_Yolo`::

	mv libnvdsinfer_custom_impl_Yolo.so /home/<agx-xavier-user>/outputs/DeepStream/nvdsinfer_custom_impl_Yolo


5. Prepare DeepStream configs
-----------------------------

We will modify inference configuration file :file:`/home/<agx-xavier-user>/outputs/DeepStream/config_infer_primary_yoloV4.txt` and DeepStream configuration file :file:`/home/<agx-xavier-user>/outputs/DeepStream/deepstream_app_config_yoloV4.txt`

:file:`/home/<agx-xavier-user>/outputs/DeepStream/config_infer_primary_yoloV4.txt`: Modify :code:`model-engine-file`

::

	# [property]
	model-engine-file=primary-detector.trt

:file:`/home/<agx-xavier-user>/outputs/DeepStream/deepstream_app_config_yoloV4.txt`:

::

	# [source0]
	uri=file:///opt/nvidia/deepstream/deepstream-5.1/samples/streams/sample_720p.mp4
	# [primary-gie]
	model-engine-file=primary-detector.trt
	# [tracker]
	ll-lib-file=/opt/nvidia/deepstream/deepstream-5.1/lib/libnvds_mot_klt.so


6. Run DeepStream
-----------------

::

	cd /home/<agx-xavier-user>/outputs/DeepStream
	deepstream-app -c deepstream_app_config_yoloV4.txt