Custom Model - Custom Parser - Yolov2-coco
==========================================

This implementation describes how we parse the output layers of custom model, Yolov2-coco (from `ONNX model zoo <https://github.com/onnx/models>`_) and deploy the model in DeepStream on Xavier. The parser is extended version of :ref:`yolo_tiny_ver` which handles 80 detection classes (see `COCO labels <https://github.com/amikelive/coco-labels/blob/master/coco-labels-2014_2017.txt>`_). 


Download model
--------------
The yolov2-coco model :code:`yolov2-coco-9.onnx` can be downloaded from `Yolov2-coco model <https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov2-coco/model/yolov2-coco-9.onnx>`_.


.. _yolo_coco_outlayer:

Output layer of yolov2
----------------------
The output layer of :code:`yolov2-coco-9.onnx` is visualised with `Netron <https://github.com/lutzroeder/netron>`_

.. image:: assets/yolov2_coco_network.png
	:width: 800 px

The model has 1 output layer :code:`218`, its dim is :code:`425x13x13`, on which :code:`425` represents 425 parameters, that is

.. math::

	\text{no. of predicted bbox} \times (\text{bbox params} + \text{class prob}) = 5 \times (5 + 80) = 425

It is encouraged to read :ref:`yolo_tiny_ver` before going to implementation.

.. _yolov2_coco_impl:

Implementation
--------------
Start with modifying the implementation of :ref:`yolo_tiny_ver`. There are few important notes.

	1. In :ref:`yolov2_tiny_impl`, :code:`NvDsInferLayerInfo.dims.numDims` is deprecated. Use :code:`NvDsInferLayerInfo.inferDims.numDims` instead.


	2. This yolo model detects 80 classes. Thus, in :ref:`yolov2_tiny_impl`, change :code:`kNUM_CLASSES_YOLO` to *80* from *20*.


	3. The pixels of input images have not been normalised to [0, 1], so the pixels of input images could make `objectness <http://imgtec.eetrend.com/blog/2021/100060857.html>`_ very low. As a result, most bounding boxes could not pass a probability threshold for validation (no bounding boxes). To fix this, add :code:`net-scale-factor=0.003921569` in inference config file to normalise the pixels.


	4. In inference config file, we need to specify :code:`output-blob-names` for the name of this yolo model's output layer. Refer to :ref:`yolo_coco_outlayer`, we specify :code:`output-blob-name=218`.

.. _prepare_yolov2_coco_config:

Prepare DeepStream configs
--------------------------
Refer to :code:`prepare_yolov2_config`:

	* Copy app config file :file:`deepstream_app_config.txt`

	* Copy inference config file :file:`primary_infer_config.txt` and update

		* :code:`onnx-file`: update the name of yolov2-coco model (:file:`../tinyyolov2-8.onnx`)

		* :code:`output-blob-names`: update the name of yolov2-coco's output layer :code:`218`

	* Create a label file :file:`labels.txt` with COCO dataset labels::

		person
		bicycle
		car
		motorcycle
		airplane
		bus
		train
		truck
		boat
		traffic light
		fire hydrant
		stop sign
		parking meter
		bench
		bird
		cat
		dog
		horse
		sheep
		cow
		elephant
		bear
		zebra
		giraffe
		backpack
		umbrella
		handbag
		tie
		suitcase
		frisbee
		skis
		snowboard
		sports ball
		kite
		baseball bat
		baseball glove
		skateboard
		surfboard
		tennis racket
		bottle
		wine glass
		cup
		fork
		knife
		spoon
		bowl
		banana
		apple
		sandwich
		orange
		broccoli
		carrot
		hot dog
		pizza
		donut
		cake
		chair
		couch
		potted plant
		bed
		dining table
		toilet
		tv
		laptop
		mouse
		remote
		keyboard
		cell phone
		microwave
		oven
		toaster
		sink
		refrigerator
		book
		clock
		vase
		scissors
		teddy bear
		hair drier
		toothbrush

.. _prepare_yolov2_coco_lib:

Prepare custom library files
----------------------------
Refer to :ref:`prepare_yolov2_lib`:

	* Copy the source file of custom parser :file:`nvdsparsebbox_yolov2.cpp`, and update

		* Refer to first and second point :ref:`yolov2_coco_impl`

	* Copy the :file:`Makefile`


Transfer files to AGX Xavier
----------------------------
Create a directory named :file:`yolov2coco`, and put downloaded model and all files described in :ref:`prepare_yolov2_coco_config` and :ref:`prepare_yolov2_coco_lib` in a following structure::

	.
	├── configs
	│   ├── deepstream_app_config.txt
	│   ├── labels.txt
	│   └── primary_infer_config.txt
	├── Makefile
	├── nvdsparsebbox_yolov2.cpp
	└── yolov2-coco-9.onnx

Zip the folder into a zip file :file:`yolov2coco.zip` and transfer the file to AGX Xavier.

::

	zip -r yolov2coco.zip yolov2coco/*

Create custom library
---------------------
On AGX Xavier, unzip :code:`yolov2coco.zip` and navigate to the parser folder and create custom library of parser.

::

	unzip yolov2coco.zip
	cd yolov2coco
	make

Run DeepStream
--------------
Finally, DeepStream configuration files and custom library for parsing are ready. On AGX Xavier, navigate to :code:`configs` folder and run DeepStream

::

	cd yolov2coco/configs
	deepstream-app -c deepstream_app_config.txt
