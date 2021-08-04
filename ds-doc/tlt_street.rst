Transfer Learning Toolkit - StreetNet (TLT2)
============================================

StreetNet is an purpose-built model (detectnet_v2 detection model with pretrained resnet as backbone). It has three detection classes which are car, cyclist, and pedestrian. The dataset we use to re-train the pretrained resnet is Kitti Dataset. After re-train the model, we export it to a TRT inference engine and deploy it on DeepStream.

Prerequisites
-------------

Get your NGC API key ready. Use Jupyter notebook to run cells starting from section *0. Set up environment variables*.

Host jupyter server
-------------------

Create TLT2 container::
	
	docker run -it --network=host --rm --gpus all nvcr.io/nvidia/tlt-streamanalytics:v2.0_py3 

To start with, let's host jupyter server in TLT2 container::

	cd examples/detectnet_v2
	jupyter notebook --allow-root --ip=0.0.0.0

0. Set up environment variables
-------------------------------
.. code-block:: python

	%env KEY=<your-ngc-api-key>
	%env USER_EXPERIMENT_DIR=/workspace/tlt-experiments/detectnet_v2
	%env DATA_DOWNLOAD_DIR=/workspace/tlt-experiments/data
	%env SPECS_DIR=/workspace/examples/detectnet_v2/specs
	%env NUM_GPUS=1

Replace :code:`KEY` with your NGC API key. If you have more than 1 GPU (V100, A100), configure :code:`NUM_GPUS`.

1. Download Kitti dataset and pre-trained model
-----------------------------------------------

Before running cells in this section, download Kitti Dataset by executing cell::

	!wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip -O /workspace/tlt-experiments/data/data_object_image_2.zip
	!wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip -O /workspace/tlt-experiments/data/data_object_label_2.zip

After download, verify the datasets::

	!if [ ! -f $DATA_DOWNLOAD_DIR/data_object_image_2.zip ]; then echo 'Image zip file not found, please download.'; else echo 'Found Image zip file.';fi
	!if [ ! -f $DATA_DOWNLOAD_DIR/data_object_label_2.zip ]; then echo 'Label zip file not found, please download.'; else echo 'Found Labels zip file.';fi

.. code-block:: bash

	Found Image zip file.
	Found Labels zip file.

Unzip the datasets::

	!unzip -u $DATA_DOWNLOAD_DIR/data_object_image_2.zip -d $DATA_DOWNLOAD_DIR
	!unzip -u $DATA_DOWNLOAD_DIR/data_object_label_2.zip -d $DATA_DOWNLOAD_DIR

Verify unzipped datasets::

	import os

	DATA_DIR = os.environ.get('DATA_DOWNLOAD_DIR')
	num_training_images = len(os.listdir(os.path.join(DATA_DIR, "training/image_2")))
	num_training_labels = len(os.listdir(os.path.join(DATA_DIR, "training/label_2")))
	num_testing_images = len(os.listdir(os.path.join(DATA_DIR, "testing/image_2")))
	print("Number of images in the trainval set. {}".format(num_training_images))
	print("Number of labels in the trainval set. {}".format(num_training_labels))
	print("Number of images in the test set. {}".format(num_testing_images))

.. code-block:: bash

	Number of images in the trainval set. 7481
	Number of labels in the trainval set. 7481
	Number of images in the test set. 7518

Let's inspect a sample kitti label::

	!cat $DATA_DOWNLOAD_DIR/training/label_2/000110.txt

.. code-block::

	Car 0.27 0 2.50 862.65 129.39 1241.00 304.96 1.73 1.74 4.71 5.50 1.30 8.19 3.07
	Car 0.68 3 -0.76 1184.97 141.54 1241.00 187.84 1.52 1.60 4.42 22.39 0.48 24.57 -0.03
	Car 0.00 1 1.73 346.64 175.63 449.93 248.90 1.58 1.76 4.18 -5.13 1.67 17.86 1.46
	Car 0.00 0 1.75 420.44 170.72 540.83 256.12 1.65 1.88 4.45 -2.78 1.64 16.30 1.58
	Car 0.00 0 -0.35 815.59 143.96 962.82 198.54 1.90 1.78 4.72 10.19 0.90 26.65 0.01
	Car 0.00 1 -2.09 966.10 144.74 1039.76 182.96 1.80 1.65 3.55 19.49 0.49 35.99 -1.59
	Van 0.00 2 -2.07 1084.26 132.74 1173.25 177.89 2.11 1.75 4.31 26.02 0.24 36.41 -1.45
	Car 0.00 2 -2.13 1004.98 144.16 1087.13 178.96 1.64 1.70 3.91 21.91 0.30 36.47 -1.59
	Car 0.00 2 1.77 407.73 178.44 487.07 230.28 1.55 1.71 4.50 -5.35 1.76 24.13 1.55
	Car 0.00 1 1.45 657.19 166.33 702.65 198.71 1.50 1.71 4.44 3.39 1.22 35.96 1.55
	Car 0.00 1 -1.46 599.30 171.76 631.96 197.12 1.58 1.71 3.75 0.39 1.54 47.31 -1.45
	Car 0.00 0 -1.02 557.79 165.74 591.61 181.27 1.66 1.65 4.45 -3.89 0.91 80.12 -1.07

If you wish to train your detection models with your own datasets and TLT, you will need to prepare Kitti labels. See `Kitti Readme <https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt>`_ for more details.

For next step, prepare repare tf records from kitti format dataset
::
	
	!tlt-dataset-convert -d $SPECS_DIR/detectnet_v2_tfrecords_kitti_trainval.txt \
                     -o $DATA_DOWNLOAD_DIR/tfrecords/kitti_trainval/kitti_trainval

Verify tf records

::
	
	!ls -rlt $DATA_DOWNLOAD_DIR/tfrecords/kitti_trainval/

::

	total 7148
	-rw-r--r-- 1 root root  97081 Jul 29 05:41 kitti_trainval-fold-000-of-002-shard-00000-of-00010
	-rw-r--r-- 1 root root 100150 Jul 29 05:41 kitti_trainval-fold-000-of-002-shard-00001-of-00010
	-rw-r--r-- 1 root root 102296 Jul 29 05:41 kitti_trainval-fold-000-of-002-shard-00002-of-00010
	-rw-r--r-- 1 root root  98880 Jul 29 05:41 kitti_trainval-fold-000-of-002-shard-00003-of-00010
	-rw-r--r-- 1 root root 105123 Jul 29 05:41 kitti_trainval-fold-000-of-002-shard-00004-of-00010
	-rw-r--r-- 1 root root  99789 Jul 29 05:41 kitti_trainval-fold-000-of-002-shard-00005-of-00010
	-rw-r--r-- 1 root root 100097 Jul 29 05:41 kitti_trainval-fold-000-of-002-shard-00006-of-00010
	-rw-r--r-- 1 root root  99369 Jul 29 05:41 kitti_trainval-fold-000-of-002-shard-00007-of-00010
	-rw-r--r-- 1 root root  98669 Jul 29 05:41 kitti_trainval-fold-000-of-002-shard-00008-of-00010
	-rw-r--r-- 1 root root 107553 Jul 29 05:41 kitti_trainval-fold-000-of-002-shard-00009-of-00010
	-rw-r--r-- 1 root root 631361 Jul 29 05:41 kitti_trainval-fold-001-of-002-shard-00000-of-00010
	-rw-r--r-- 1 root root 619886 Jul 29 05:42 kitti_trainval-fold-001-of-002-shard-00001-of-00010
	-rw-r--r-- 1 root root 612923 Jul 29 05:42 kitti_trainval-fold-001-of-002-shard-00002-of-00010
	-rw-r--r-- 1 root root 629177 Jul 29 05:42 kitti_trainval-fold-001-of-002-shard-00003-of-00010
	-rw-r--r-- 1 root root 628648 Jul 29 05:42 kitti_trainval-fold-001-of-002-shard-00004-of-00010
	-rw-r--r-- 1 root root 628205 Jul 29 05:42 kitti_trainval-fold-001-of-002-shard-00005-of-00010
	-rw-r--r-- 1 root root 622273 Jul 29 05:42 kitti_trainval-fold-001-of-002-shard-00006-of-00010
	-rw-r--r-- 1 root root 635870 Jul 29 05:42 kitti_trainval-fold-001-of-002-shard-00007-of-00010
	-rw-r--r-- 1 root root 632348 Jul 29 05:42 kitti_trainval-fold-001-of-002-shard-00008-of-00010
	-rw-r--r-- 1 root root 623560 Jul 29 05:42 kitti_trainval-fold-001-of-002-shard-00009-of-00010


The last preparation is downloading the pre-trained model. Let's see what pretrained models are available for detectnet_v2

::

	!ngc registry model list nvidia/tlt_pretrained_detectnet_v2:*

::

	+-------+-------+-------+-------+-------+-------+-------+-------+-------+
	| Versi | Accur | Epoch | Batch | GPU   | Memor | File  | Statu | Creat |
	| on    | acy   | s     | Size  | Model | y Foo | Size  | s     | ed    |
	|       |       |       |       |       | tprin |       |       | Date  |
	|       |       |       |       |       | t     |       |       |       |
	+-------+-------+-------+-------+-------+-------+-------+-------+-------+
	| vgg19 | 82.6  | 80    | 1     | V100  | 153.8 | 153.7 | UPLOA | Apr   |
	|       |       |       |       |       |       | 7 MB  | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	| vgg16 | 82.2  | 80    | 1     | V100  | 113.2 | 113.2 | UPLOA | Apr   |
	|       |       |       |       |       |       | MB    | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	| squee | 65.67 | 80    | 1     | V100  | 6.5   | 6.46  | UPLOA | Apr   |
	| zenet |       |       |       |       |       | MB    | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	| resne | 82.7  | 80    | 1     | V100  | 294.5 | 294.5 | UPLOA | Apr   |
	| t50   |       |       |       |       |       | 3 MB  | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	| resne | 79.5  | 80    | 1     | V100  | 163.6 | 163.5 | UPLOA | Aug   |
	| t34   |       |       |       |       |       | 5 MB  | D_COM | 03,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	| resne | 79.0  | 80    | 1     | V100  | 89.0  | 89.02 | UPLOA | Apr   |
	| t18   |       |       |       |       |       | MB    | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	| resne | 79.2  | 80    | 1     | V100  | 38.3  | 38.34 | UPLOA | Apr   |
	| t10   |       |       |       |       |       | MB    | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	| mobil | 77.5  | 80    | 1     | V100  | 5.1   | 5.1   | UPLOA | Apr   |
	| enet_ |       |       |       |       |       | MB    | D_COM | 29,   |
	| v2    |       |       |       |       |       |       | PLETE | 2020  |
	| mobil | 79.5  | 80    | 1     | V100  | 13.4  | 13.37 | UPLOA | Apr   |
	| enet_ |       |       |       |       |       | MB    | D_COM | 29,   |
	| v1    |       |       |       |       |       |       | PLETE | 2020  |
	| googl | 82.2  | 80    | 1     | V100  | 47.7  | 47.74 | UPLOA | Apr   |
	| enet  |       |       |       |       |       | MB    | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	| effic | 77.11 | 80    | 1     | V100  | 16.9  | 16.9  | UPLOA | Jun   |
	| ientn |       |       |       |       |       | MB    | D_COM | 09,   |
	| et_b0 |       |       |       |       |       |       | PLETE | 2021  |
	| _swis |       |       |       |       |       |       |       |       |
	| h     |       |       |       |       |       |       |       |       |
	| effic | 77.11 | 80    | 1     | V100  | 16.9  | 16.9  | UPLOA | Jun   |
	| ientn |       |       |       |       |       | MB    | D_COM | 09,   |
	| et_b0 |       |       |       |       |       |       | PLETE | 2021  |
	| _relu |       |       |       |       |       |       |       |       |
	| darkn | 76.44 | 80    | 1     | V100  | 467.3 | 467.3 | UPLOA | Apr   |
	| et53  |       |       |       |       |       | 2 MB  | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	| darkn | 77.52 | 80    | 1     | V100  | 229.1 | 229.1 | UPLOA | Apr   |
	| et19  |       |       |       |       |       | 5 MB  | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	+-------+-------+-------+-------+-------+-------+-------+-------+-------+

Resnet18 is selected

::

	!mkdir -p $USER_EXPERIMENT_DIR/pretrained_resnet18/
	!ngc registry model download-version nvidia/tlt_pretrained_detectnet_v2:resnet18 \
    --dest $USER_EXPERIMENT_DIR/pretrained_resnet18

Verify installed Resnet18::

	!ls -rlt $USER_EXPERIMENT_DIR/pretrained_resnet18/tlt_pretrained_detectnet_v2_vresnet18

::

	total 91164
	-rw------- 1 root root 93345248 Jul 29 05:47 resnet18.hdf5

2. Provide training specification
---------------------------------

You can inspect prepared training specification::

	!cat $SPECS_DIR/detectnet_v2_train_resnet18_kitti.txt

3. Run TLT training
-------------------

.. code-block:: bash

	!tlt-train detectnet_v2 -e $SPECS_DIR/detectnet_v2_train_resnet18_kitti.txt \
	                        -r $USER_EXPERIMENT_DIR/experiment_dir_unpruned \
	                        -k $KEY \
	                        -n resnet18_detector \
	                        --gpus $NUM_GPUS

* :code:`-e`: your training specification file
* :code:`-r`: output directory
* :code:`-k`: your ngc api key
* :code:`-n`: name of your model

Sample training results using V100 cluster:
::

	Validation cost: 0.000045
	Mean average_precision (in %): 76.4810

	class name      average precision (in %)
	------------  --------------------------
	car                              81.5105
	cyclist                          80.9753
	pedestrian                       66.9571

	Median Inference Time: 0.005755
	2021-06-22 16:49:48,201 [INFO] modulus.hooks.sample_counter_hook: Train Samples / sec: 144.048
	Time taken to run iva.detectnet_v2.scripts.train:main: 1:33:16.548631.


Note that the training can take many hours... good time to go grab a coffee.

4. Evaluate the trained model
-----------------------------

::

	!tlt-evaluate detectnet_v2 -e $SPECS_DIR/detectnet_v2_train_resnet18_kitti.txt\
                           -m $USER_EXPERIMENT_DIR/experiment_dir_unpruned/weights/resnet18_detector.tlt \
                           -k $KEY

Sample evaluation results:

::

	Validation cost: 0.000262
	Mean average_precision (in %): 76.5375

	class name      average precision (in %)
	------------  --------------------------
	car                              81.5001
	cyclist                          81.1041
	pedestrian                       67.0082

	Median Inference Time: 0.005474
	2021-06-22 16:55:35,410 [INFO] iva.detectnet_v2.scripts.evaluate: Evaluation complete.
	Time taken to run iva.detectnet_v2.scripts.evaluate:main: 0:00:23.463324.

5. Prune the trained model
--------------------------

::

	!tlt-prune -m $USER_EXPERIMENT_DIR/experiment_dir_unpruned/weights/resnet18_detector.tlt \
           -o $USER_EXPERIMENT_DIR/experiment_dir_pruned/resnet18_nopool_bn_detectnet_v2_pruned.tlt \
           -eq union \
           -pth 0.0000052 \
           -k $KEY

* :code:`pth`: lower this value to improve model accuracy. If your current accuracy is good enough, increase it to have a smaller model.

6. Retrain pruned model
-----------------------

You can inspect retrain specification file::

	!cat $SPECS_DIR/detectnet_v2_retrain_resnet18_kitti.txt

Retrain pruned model

::

	!tlt-train detectnet_v2 -e $SPECS_DIR/detectnet_v2_retrain_resnet18_kitti.txt \
                        -r $USER_EXPERIMENT_DIR/experiment_dir_retrain \
                        -k $KEY \
                        -n resnet18_detector_pruned \
                        --gpus $NUM_GPUS

Sample retrain results using V100 cluster:

::

	Validation cost: 0.000044
	Mean average_precision (in %): 77.6199

	class name      average precision (in %)
	------------  --------------------------
	car                              82.7714
	cyclist                          82.2733
	pedestrian                       67.8151

	Median Inference Time: 0.005328
	2021-06-22 18:28:06,949 [INFO] modulus.hooks.sample_counter_hook: Train Samples / sec: 146.980
	Time taken to run iva.detectnet_v2.scripts.train:main: 1:32:00.270818.

7. Evaluate retrained model
---------------------------
::

	!tlt-evaluate detectnet_v2 -e $SPECS_DIR/detectnet_v2_retrain_resnet18_kitti.txt \
                           -m $USER_EXPERIMENT_DIR/experiment_dir_retrain/weights/resnet18_detector_pruned.tlt \
                           -k $KEY

Sample evaluation results:
::

	Validation cost: 0.000261
	Mean average_precision (in %): 77.6808

	class name      average precision (in %)
	------------  --------------------------
	car                              82.7801
	cyclist                          82.3879
	pedestrian                       67.8744

	Median Inference Time: 0.005269
	2021-06-22 18:36:59,560 [INFO] iva.detectnet_v2.scripts.evaluate: Evaluation complete.
	Time taken to run iva.detectnet_v2.scripts.evaluate:main: 0:00:23.347481.

8. Visualise inferences
-----------------------

Follow the notebook to visualise inferences.


9. Prepare files for final deployment
-------------------------------------

At this step, we will prepare TLT output files for int8 optimization.  Also, we prepare DeepStream config file, inference config file, labels file. When all files are prepared, we will transfer the files to AGX Xavier.

Export tlt-trained model to .etlt model.

.. code-block:: bash

	!mkdir -p $USER_EXPERIMENT_DIR/experiment_dir_final
	# Removing a pre-existing copy of the etlt if there has been any.
	import os
	output_file=os.path.join(os.environ['USER_EXPERIMENT_DIR'],
	                         "experiment_dir_final/resnet18_detector.etlt")
	if os.path.exists(output_file):
	    os.system("rm {}".format(output_file))
	!tlt-export detectnet_v2 \
	            -m $USER_EXPERIMENT_DIR/experiment_dir_retrain/weights/resnet18_detector_pruned.tlt \
	            -o $USER_EXPERIMENT_DIR/experiment_dir_final/resnet18_detector.etlt \
	            -k $KEY

* :code:`-m`: tlt-trained model.
* :code:`-o`: output path of etlt model :file:`resnet18_detector.etlt`

Int8 optimization
~~~~~~~~~~~~~~~~~

Generate :file:`calibration.tensor` file:

::

	!tlt-int8-tensorfile detectnet_v2 -e $SPECS_DIR/detectnet_v2_retrain_resnet18_kitti.txt \
                                  -m 10 \
                                  -o $USER_EXPERIMENT_DIR/experiment_dir_final/calibration.tensor

Export engine :file:`resnet18_detector.trt.int8`, and :file:`calibration.bin`.

::

	!rm -rf $USER_EXPERIMENT_DIR/experiment_dir_final/resnet18_detector.etlt
	!rm -rf $USER_EXPERIMENT_DIR/experiment_dir_final/calibration.bin
	!tlt-export detectnet_v2 \
	            -m $USER_EXPERIMENT_DIR/experiment_dir_retrain/weights/resnet18_detector_pruned.tlt \
	            -o $USER_EXPERIMENT_DIR/experiment_dir_final/resnet18_detector.etlt \
	            -k $KEY  \
	            --cal_data_file $USER_EXPERIMENT_DIR/experiment_dir_final/calibration.tensor \
	            --data_type int8 \
	            --batches 10 \
	            --batch_size 4 \
	            --max_batch_size 4\
	            --engine_file $USER_EXPERIMENT_DIR/experiment_dir_final/resnet18_detector.trt.int8 \
	            --cal_cache_file $USER_EXPERIMENT_DIR/experiment_dir_final/calibration.bin \
	            --verbose

* :code:`-m`: your pruned model
* :code:`-o`: your output model
* :code:`--cal_data_file`: your calibration.tensor file generated earlier
* :code:`--engine_file`	: output path of your engine file :file:`resnet18_detector.trt.int8`
* :code:`--cal_cache_file`: output path of your calibration cache file :file:`calibration.bin`.

Let's prepare :file:`labels.txt`, :file:`primary_infer.txt`, and :file:`stream_config.txt`:

::

	# labels.txt
	car
	cyclist
	pedestrain

::

	# primary_infer.txt
	[property]
	gpu-id=0
	net-scale-factor=0.0039215697906911373
	model-engine-file=resnet18_detector.trt
	labelfile-path=labels.txt
	int8-calib-file=calibration.bin
	uff-input-blob-name=input_1
	batch-size=1
	input-dims=3;384;1248;0
	process-mode=1
	model-color-format=0
	## 0=FP32, 1=INT8, 2=FP16 mode
	network-mode=1
	num-detected-classes=3
	interval=0
	gie-unique-id=1
	output-blob-names=output_cov/Sigmoid;output_bbox/BiasAdd
	is-classifier=0

	[class-attrs-all]
	threshold=0.1
	group-threshold=1
	## Set eps=0.7 and minBoxes for enable-dbscan=1
	eps=0.2
	minBoxes=5
	roi-top-offset=0
	roi-bottom-offset=0
	detected-min-w=0
	detected-min-h=0
	detected-max-w=0
	detected-max-h=0

	## Per class configuration
	#[class-attrs-2]
	#threshold=0.6
	#eps=0.5
	#group-threshold=3
	#roi-top-offset=20
	#roi-bottom-offset=10
	#detected-min-w=40
	#detected-min-h=40
	#detected-max-w=400
	#detected-max-h=800

::

	# stream_config.txt
	[application]
	enable-perf-measurement=1
	perf-measurement-interval-sec=5
	#gie-kitti-output-dir=streamscl

	[tiled-display]
	enable=1
	rows=1
	columns=1
	width=1600
	height=900

	[source0]
	enable=1
	#Type - 1=CameraV4L2 2=URI 3=MultiURI 4=RTSP
	type=3
	uri=file:///opt/nvidia/deepstream/deepstream-5.1/samples/streams/sample_1080p_h264.mp4
	num-sources=1
	#drop-frame-interval=2
	gpu-id=0
	# (0): memtype_device   - Memory type Devqice
	# (1): memtype_pinned   - Memory type Host Pinned
	# (2): memtype_unified  - Memory type Unified
	cudadec-memtype=0

	[sink0]
	enable=1
	#Type - 1=FakeSink 2=EglSink 3=File 4=RTSPStreaming 5=Overlay
	type=5
	sync=0
	display-id=0
	offset-x=0
	offset-y=0
	width=0
	height=0
	overlay-id=1
	source-id=0

	[sink1]
	enable=0
	type=3
	#1=mp4 2=mkv
	container=1
	#1=h264 2=h265 3=mpeg4
	codec=1
	sync=0
	bitrate=2000000
	output-file=out.mp4
	source-id=0

	[sink2]
	enable=0
	#Type - 1=FakeSink 2=EglSink 3=File 4=RTSPStreaming 5=Overlay
	type=4
	#1=h264 2=h265
	codec=1
	sync=0
	bitrate=4000000
	# set below properties in case of RTSPStreaming
	rtsp-port=8554
	udp-port=5400

	[osd]
	enable=1
	border-width=2
	text-size=12
	text-color=1;1;1;1;
	text-bg-color=0.3;0.3;0.3;1
	font=Serif
	show-clock=0
	clock-x-offset=800
	clock-y-offset=820
	clock-text-size=12
	clock-color=1;0;0;0

	[streammux]
	##Boolean property to inform muxer that sources are live
	live-source=0
	batch-size=1
	##time out in usec, to wait after the first buffer is available
	##to push the batch even if the complete batch is not formed
	batched-push-timeout=40000
	## Set muxer output width and height
	width=1920
	height=1080

	# config-file property is mandatory for any gie section.
	# Other properties are optional and if set will override the properties set in
	# the infer config file.
	[primary-gie]
	enable=1
	model-engine-file=resnet18_detector.trt
	batch-size=1
	#Required by the app for OSD, not a plugin property
	bbox-border-color0=1;0;0;1
	bbox-border-color1=0;1;1;1
	bbox-border-color2=1;0;1;1
	interval=10
	#Required by the app for SGIE, when used along with config-file property
	gie-unique-id=1
	config-file=primary_infer.txt

	[tests]
	file-loop=1

	[tracker]
	enable=1
	tracker-width=512
	tracker-height=128
	#ll-lib-file=/opt/nvidia/deepstream/deepstream-4.0/lib/libnvds_mot_iou.so
	#ll-lib-file=/opt/nvidia/deepstream/deepstream-4.0/lib/libnvds_nvdcf.so
	ll-lib-file=/opt/nvidia/deepstream/deepstream-5.1/lib/libnvds_mot_klt.so
	#ll-config-file required for DCF/IOU only
	#ll-config-file=tracker_config.yml
	#ll-config-file=iou_config.txt
	gpu-id=0
	#enable-batch-process applicable to DCF only
	enable-batch-process=1

When all files are prepared, gather files together and send to AGX Xavier.

::

	├── calibration.bin
	├── calibration.tensor
	├── labels.txt
	├── primary_infer.txt
	├── resnet18_detector.etlt
	├── resnet18_detector.trt.int8
	└── stream_config.txt

	0 directories, 7 files

10. Generate TRT engine using tlt-converter
-------------------------------------------

Once files were transferred to AGX Xavier, we will generate TRT engine on AGX Xavier. Let's download tlt-converter `here <https://docs.nvidia.com/tlt/tlt-user-guide/text/tensorrt.html#id2>`_.

Follow the readme file in the downloaded package to install tlt-converter, and verify converter:

::

	chmod u+x tlt-converter
	./tlt-converter -h

If everything is alright, let's generate TRT engine :file:`resnet18_detector.trt` with int8 optimization:

::

	./tlt-converter $USER_EXPERIMENT_DIR/experiment_dir_final/resnet18_detector.etlt \
               -k $KEY \
               -c $USER_EXPERIMENT_DIR/experiment_dir_final/calibration.bin \
               -o output_cov/Sigmoid,output_bbox/BiasAdd \
               -d 3,384,1248 \
               -i nchw \
               -m 64 \
               -t int8 \
               -e $USER_EXPERIMENT_DIR/experiment_dir_final/resnet18_detector.trt \
               -b 4

11. Run DeepStream
------------------

Once the TRT engine is ready, double-check the paths in config files, and run DeepStream

::

	deepstream-app -c stream_config.txt