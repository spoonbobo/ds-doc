Custom Model - SSD
==================

This sample deployment of Yolov4 detection model describes how can we export SSD detection model with pretrained Resnet50 as backbone to ONNX model, and then convert it to TRT inference engine and deploy the engine on DeepStream. See `GitHub repository <https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD>`_ for more details of this deployment of SSD detection model on Nvidia AGX Xavier.

Prerequisites
-------------

* Nvidia docker
* PyTorch NGC container
* TensorRT >= 8.0.x
* Nvidia AGX Xavier
* GPU-based architecture
* `onnx2trt package <https://github.com/onnx/onnx-tensorrt>`_

1. Clone the github repo
------------------------
::

	git clone https://github.com/NVIDIA/DeepLearningExamples
	cd DeepLearningExamples/PyTorch/Detection/SSD

2. Donwload and preprocess the dataset
--------------------------------------
Download COCO dataset and perform data preprocessing.

::

	export COCO_DIR=$(pwd)/datasets
	download_dataset.sh $COCO_DIR

To see how data is preprocessed, see `here <https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD#getting-the-data>`_.

If you wish to fine-tune the preprocess process, edit :file:`src/coco_pipeline.py`.

3. Build the PyTorch container
------------------------------
::

	docker build . -t nvidia_ssd

4. Start the PyTorch container
------------------------------
Before starting the container, create a mounting output directory to store the outputs::

	!mkdir outputs

Start the container::

	nvidia-docker run --rm -it --ulimit memlock=-1 --ulimit stack=67108864 -v $COCO_DIR:/coco -v $(pwd)/outputs/:/outputs --ipc=host nvidia_ssd

5. Start training SSD detection model
-------------------------------------
Specify number of GPUs and precision used::

	export GPUS=<number-of-GPU-to-use>
	export PREC=<FP16-or-FP32>

You can choose 1, 4, or 8 value for :code:`GPUS`

start training SSD::

	bash ./examples/SSD300_${PREC}_${GPUS}GPU.sh . /coco --save /outputs

You can fine-tune the training. See `Training Parameters <https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD#parameters>`_ for more details.

For each epoch, the updated model weight will be stored to :file:`/outputs` mounted directory.

6. Convert checkpoint to ONNX model
-----------------------------------

To convert checkpoint to ONNX model, we write a python script :file:`/outputs/torch2onnx.py`.

.. code-block:: python

	import torch.onnx
	import torchvision
	import torch
	import sys
	import io

	sys.path.insert(1, "/workspace/src")
	from model import SSD300, ResNet, Loss

	def convert_onnx(filepath):
	    
	    with open(filepath, 'rb') as f:
	        buffer = io.BytesIO(f.read())

	    # checkpoint dict
	    checkpoint = torch.load(buffer)

	    # init SSD300
	    model = SSD300()

	    # load state dict
	    model.load_state_dict(checkpoint['model'])

	    # inference mode
	    model.eval().cuda().half()
	    
	    # dummy input to convert onnx model    
	    dummy_input = torch.ones(1, 3, 300, 300, dtype=torch.float16).cuda()
	    
	    # convert to onnx model
	    torch.onnx.export(model, dummy_input, "model.onnx")

	epoch = sys.argv[1]
	print("exporting checkpoint epoch_{}.pt to model.onnx".format(epoch))
	convert_onnx("epoch_{}.pt".format(epoch))

*Usage of this script*

.. code-block:: bash
	
	python3 /outputs/torch2onnx.py <epoch-no>

For example,

.. code-block:: bash

	python3 /outputs/torch2onnx.py 64


7. Convert ONNX model to TRT engine
-----------------------------------

Basically, you have two ways of accessing it

7.1 Use onnx2trt
~~~~~~~~~~~~~~~~

In this step, we need onnx2trt package. Exit PyTorch container, in TRT environment, run::

	git submodule update --init --recursive

	apt-get install libprotobuf-dev protobuf-compiler

	git submodule update --init --recursive

7.2 Use trtexec
~~~~~~~~~~~~~~~


8. Prepare labels file
----------------------

parse-bbox-func-name=NvDsInferParseCustomSSD
custom-lib-path=nvdsinfer_custom_impl_ssd/libnvdsinfer_custom_impl_ssd.so


9. Prepare app config file
--------------------------


10. Prepare gie config file
---------------------------