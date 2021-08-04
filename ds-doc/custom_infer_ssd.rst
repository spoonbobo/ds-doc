Custom Model - SSD
==================

This sample deployment of Yolov4 detection model describes how can we export SSD detection model with pretrained Resnet50 as backbone to ONNX model, and then convert it to TRT inference engine and deploy the engine on DeepStream. See `GitHub repository <https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD>`_ for more details of this deployment of SSD detection model on Nvidia AGX Xavier.

Prerequisites
-------------

* Nvidia docker
* PyTorch NGC container
* Nvidia AGX Xavier
* GPU-based architecture

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

6. 
