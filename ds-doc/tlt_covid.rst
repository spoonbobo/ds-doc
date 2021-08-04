Transfer Learning Toolkit - CovidNet (TLT2)
===========================================

NewbieNet is an purpose-built model (ssd detection model with pretrained Resnet18 as backbone). It is used to detect Identify and localize COVID-19 abnormalities on chest radiographs. The dataset we use to re-train the pretrained resnet is SIIM-FISABIO-RSNA COVID-19 Detection Dataset. After re-train the model, we export it to a TRT inference engine and deploy it on DeepStream.

Prerequisites
-------------

* Get your NGC API key ready. Use Jupyter notebook to run cells starting from section *0. Set up environment variables*.

Host jupyter server
-------------------
Create TLT2 container::
	
	docker run -it --network=host --rm --gpus all nvcr.io/nvidia/tlt-streamanalytics:v2.0_py3 

To start with, let's host jupyter server in TLT2 container::

	jupyter notebook --allow-root --ip=0.0.0.0

0. Set up environment variables
-------------------------------
.. code-block::

	%set_env KEY=<your-ngc-api-key>
	%set_env USER_EXPERIMENT_DIR=/workspace/tlt-experiments/ssd
	%set_env DATA_DOWNLOAD_DIR=/workspace/tlt-experiments/data
	%set_env SPECS_DIR=/workspace/tlt-covid-specs

1. Kaggle API & prepare dataset
-------------------------------

To download SIIM-FISABIO-RSNA COVID-19 Detection Dataset, you will first need to register `Kaggle API account <https://www.kaggle.com/>`_, and join the `detection competition <https://www.kaggle.com/c/siim-covid19-detection>`_, then generate your API key (a json file).

Install Kaggle API package::

	!pip install kaggle
	!mkdir ~/.kaggle

Once the Kaggle API is set up, open the json file and copy the content of your API key to :file:`~/.kaggle/kaggle.json`::

	%%writefile ~/.kaggle/kaggle.json
	{"username":"<your-username>", "key":"<your-key>"}

Then, we download the dataset::

	!mkdir -p $DATA_DOWNLOAD_DIR
	!cd $DATA_DOWNLOAD_DIR && \
		kaggle competitions download -c siim-covid19-detection && \
		unzip siim-covid19-detection

Note that the dataset is very large in size, around 83G. Grab a coffee and comeback after a while.

Verify the dataset::

	!ls -l /workspace/tlt-experiments/data

	
2. Format the dataset
---------------------

This step, we format the dataset into Kitti format. See :ref:`tlt_specs` for more details::

	!pip install pandas pydicom gdcm pylibjpeg pylibjpeg-libjpeg
	!git clone https://github.com/spoonnvidia/converters
	!cd converters && \
		python siim2kitti.py --l /workspace/tlt-experiments/data/train_image_level.csv --t /workspace/tlt-experiments/data/train --px 512 --kf /workspace/tlt-experiments/data/SIIM-FISABIO-RSNA

Verify formatted dataset::

	!ls -l /workspace/tlt-experiments/data/SIIM-FISABIO-RSNA

3. Prepare TFRecords conversion specification file
--------------------------------------------------

Create a directory of storing specs files::

	!mkdir tlt-covid-specs

Prepare TFRecords conversion specification file :file:`covid_tfrecords_kitti_trainval.txt`::

	%%writefile tlt-covid-specs/covid_tfrecords_kitti_trainval.txt
	kitti_config {
	  root_directory_path: "/workspace/tlt-experiments/data/SIIM-FISABIO-RSNA/formatted"
	  image_dir_name: "images"
	  label_dir_name: "labels"
	  image_extension: ".png"
	  partition_mode: "random"
	  num_partitions: 2
	  val_split: 14
	  num_shards: 10
	}
	image_directory_path: "/workspace/tlt-experiments/data/SIIM-FISABIO-RSNA/formatted"

4. Prepare TF records
---------------------
::

	!mkdir -p $USER_EXPERIMENT_DIR/tfrecords
	!tlt-dataset-convert -d $SPECS_DIR/covid_tfrecords_kitti_trainval.txt \
	                     -o $DATA_DOWNLOAD_DIR/tfrecords/kitti_trainval/kitti_trainval

Verify TF records::

	!ls -l $DATA_DOWNLOAD_DIR/tfrecords/kitti_trainval/

::

	total 3980
	-rw-r--r-- 1 root root  56343 Aug  1 02:02 kitti_trainval-fold-000-of-002-shard-00000-of-00010
	-rw-r--r-- 1 root root  55794 Aug  1 02:02 kitti_trainval-fold-000-of-002-shard-00001-of-00010
	-rw-r--r-- 1 root root  56400 Aug  1 02:02 kitti_trainval-fold-000-of-002-shard-00002-of-00010
	-rw-r--r-- 1 root root  55847 Aug  1 02:02 kitti_trainval-fold-000-of-002-shard-00003-of-00010
	-rw-r--r-- 1 root root  56465 Aug  1 02:02 kitti_trainval-fold-000-of-002-shard-00004-of-00010
	-rw-r--r-- 1 root root  56277 Aug  1 02:02 kitti_trainval-fold-000-of-002-shard-00005-of-00010
	-rw-r--r-- 1 root root  56221 Aug  1 02:02 kitti_trainval-fold-000-of-002-shard-00006-of-00010
	-rw-r--r-- 1 root root  56158 Aug  1 02:02 kitti_trainval-fold-000-of-002-shard-00007-of-00010
	-rw-r--r-- 1 root root  55969 Aug  1 02:02 kitti_trainval-fold-000-of-002-shard-00008-of-00010
	-rw-r--r-- 1 root root  59846 Aug  1 02:02 kitti_trainval-fold-000-of-002-shard-00009-of-00010
	-rw-r--r-- 1 root root 346666 Aug  1 02:02 kitti_trainval-fold-001-of-002-shard-00000-of-00010
	-rw-r--r-- 1 root root 347483 Aug  1 02:02 kitti_trainval-fold-001-of-002-shard-00001-of-00010
	-rw-r--r-- 1 root root 347959 Aug  1 02:02 kitti_trainval-fold-001-of-002-shard-00002-of-00010
	-rw-r--r-- 1 root root 349149 Aug  1 02:02 kitti_trainval-fold-001-of-002-shard-00003-of-00010
	-rw-r--r-- 1 root root 346678 Aug  1 02:02 kitti_trainval-fold-001-of-002-shard-00004-of-00010
	-rw-r--r-- 1 root root 347554 Aug  1 02:02 kitti_trainval-fold-001-of-002-shard-00005-of-00010
	-rw-r--r-- 1 root root 347662 Aug  1 02:02 kitti_trainval-fold-001-of-002-shard-00006-of-00010
	-rw-r--r-- 1 root root 347538 Aug  1 02:02 kitti_trainval-fold-001-of-002-shard-00007-of-00010
	-rw-r--r-- 1 root root 348389 Aug  1 02:02 kitti_trainval-fold-001-of-002-shard-00008-of-00010
	-rw-r--r-- 1 root root 354036 Aug  1 02:02 kitti_trainval-fold-001-of-002-shard-00009-of-00010

5. Download pretrained models
-----------------------------
::

	!ngc registry model list nvidia/tlt_pretrained_object_detection:*

::

	+-------+-------+-------+-------+-------+-------+-------+-------+-------+
	| Versi | Accur | Epoch | Batch | GPU   | Memor | File  | Statu | Creat |
	| on    | acy   | s     | Size  | Model | y Foo | Size  | s     | ed    |
	|       |       |       |       |       | tprin |       |       | Date  |
	|       |       |       |       |       | t     |       |       |       |
	+-------+-------+-------+-------+-------+-------+-------+-------+-------+
	| vgg19 | 77.56 | 80    | 1     | V100  | 153.7 | 153.7 | UPLOA | Apr   |
	|       |       |       |       |       |       | 2 MB  | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	| vgg16 | 77.17 | 80    | 1     | V100  | 515.1 | 515.0 | UPLOA | Apr   |
	|       |       |       |       |       |       | 9 MB  | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	| squee | 65.13 | 80    | 1     | V100  | 6.5   | 6.46  | UPLOA | Apr   |
	| zenet |       |       |       |       |       | MB    | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	| resne | 77.91 | 80    | 1     | V100  | 294.2 | 294.2 | UPLOA | Apr   |
	| t50   |       |       |       |       |       | MB    | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	| resne | 77.04 | 80    | 1     | V100  | 170.7 | 170.6 | UPLOA | Apr   |
	| t34   |       |       |       |       |       | 5 MB  | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	| resne | 76.74 | 80    | 1     | V100  | 89.0  | 88.96 | UPLOA | Apr   |
	| t18   |       |       |       |       |       | MB    | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	| resne | 77.78 | 80    | 1     | V100  | 328.4 | 328.4 | UPLOA | Apr   |
	| t101  |       |       |       |       |       | 2 MB  | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	| resne | 74.38 | 80    | 1     | V100  | 38.3  | 38.31 | UPLOA | Apr   |
	| t10   |       |       |       |       |       | MB    | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	| mobil | 72.75 | 80    | 1     | V100  | 5.0   | 5.01  | UPLOA | Apr   |
	| enet_ |       |       |       |       |       | MB    | D_COM | 29,   |
	| v2    |       |       |       |       |       |       | PLETE | 2020  |
	| mobil | 79.5  | 80    | 1     | V100  | 26.2  | 26.22 | UPLOA | Apr   |
	| enet_ |       |       |       |       |       | MB    | D_COM | 29,   |
	| v1    |       |       |       |       |       |       | PLETE | 2020  |
	| googl | 77.11 | 80    | 1     | V100  | 47.6  | 47.64 | UPLOA | Apr   |
	| enet  |       |       |       |       |       | MB    | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	| effic | 77.11 | 80    | 1     | V100  | 26.8  | 26.78 | UPLOA | Jun   |
	| ientn |       |       |       |       |       | MB    | D_COM | 09,   |
	| et_b1 |       |       |       |       |       |       | PLETE | 2021  |
	| _swis |       |       |       |       |       |       |       |       |
	| h     |       |       |       |       |       |       |       |       |
	| effic | 77.11 | 80    | 1     | V100  | 26.8  | 26.78 | UPLOA | Jun   |
	| ientn |       |       |       |       |       | MB    | D_COM | 09,   |
	| et_b1 |       |       |       |       |       |       | PLETE | 2021  |
	| _relu |       |       |       |       |       |       |       |       |
	| effic | 77.9  | 80    | 1     | V100  | 16.9  | 16.9  | UPLOA | Feb   |
	| ientn |       |       |       |       |       | MB    | D_COM | 09,   |
	| et_b0 |       |       |       |       |       |       | PLETE | 2021  |
	| _swis |       |       |       |       |       |       |       |       |
	| h     |       |       |       |       |       |       |       |       |
	| effic | 77.6  | 80    | 1     | V100  | 16.9  | 16.9  | UPLOA | Feb   |
	| ientn |       |       |       |       |       | MB    | D_COM | 09,   |
	| et_b0 |       |       |       |       |       |       | PLETE | 2021  |
	| _relu |       |       |       |       |       |       |       |       |
	| darkn | 76.44 | 80    | 1     | V100  | 311.7 | 311.6 | UPLOA | Apr   |
	| et53  |       |       |       |       |       | 8 MB  | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	| darkn | 77.52 | 80    | 1     | V100  | 152.8 | 152.8 | UPLOA | Apr   |
	| et19  |       |       |       |       |       | 2 MB  | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	| cspda | 76.44 | 80    | 1     | V100  | 103.0 | 102.9 | UPLOA | Feb   |
	| rknet |       |       |       |       |       | 9 MB  | D_COM | 02,   |
	| 53    |       |       |       |       |       |       | PLETE | 2021  |
	| cspda | 77.52 | 80    | 1     | V100  | 62.9  | 62.86 | UPLOA | Feb   |
	| rknet |       |       |       |       |       | MB    | D_COM | 02,   |
	| 19    |       |       |       |       |       |       | PLETE | 2021  |
	+-------+-------+-------+-------+-------+-------+-------+-------+-------+

Let's download resnet18 for training::

	!mkdir -p $USER_EXPERIMENT_DIR/pretrained_resnet18/
	!ngc registry model download-version nvidia/tlt_pretrained_object_detection:resnet18 --dest $USER_EXPERIMENT_DIR/pretrained_resnet18

Verify the pretrained resnet18::

	!ls -l $USER_EXPERIMENT_DIR/pretrained_resnet18/tlt_pretrained_object_detection_vresnet18

6. Prepare training specification file
--------------------------------------
:file:`tlt-covid-specs/covid_tlt_train.txt`:

::

	%%writefile tlt-covid-specs/covid_tlt_train.txt
	random_seed: 42
	ssd_config {
	  aspect_ratios_global: "[1.0, 2.0, 0.5, 3.0, 1.0/3.0]"
	  scales: "[0.05, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85]"
	  two_boxes_for_ar1: true
	  clip_boxes: false
	  loss_loc_weight: 0.8
	  focal_loss_alpha: 0.25
	  focal_loss_gamma: 2.0
	  variances: "[0.1, 0.1, 0.2, 0.2]"
	  arch: "resnet"
	  nlayers: 18
	  freeze_bn: false
	  freeze_blocks: 0
	}
	training_config {
	  batch_size_per_gpu: 16
	  num_epochs: 80
	  enable_qat: false
	  learning_rate {
	  soft_start_annealing_schedule {
	    min_learning_rate: 5e-5
	    max_learning_rate: 2e-2
	    soft_start: 0.15
	    annealing: 0.8
	    }
	  }
	  regularizer {
	    type: L1
	    weight: 3e-5
	  }
	}
	eval_config {
	  validation_period_during_training: 10
	  average_precision_mode: SAMPLE
	  batch_size: 16
	  matching_iou_threshold: 0.5
	}
	nms_config {
	  confidence_threshold: 0.01
	  clustering_iou_threshold: 0.6
	  top_k: 200
	}
	augmentation_config {
	  preprocessing {
	    output_image_width: 512
	    output_image_height: 512
	    output_image_channel: 3
	    crop_right: 512
	    crop_bottom: 512
	    min_bbox_width: 1.0
	    min_bbox_height: 1.0
	  }
	  spatial_augmentation {
	    hflip_probability: 0.5
	    vflip_probability: 0.0
	    zoom_min: 0.7
	    zoom_max: 1.8
	    translate_max_x: 8.0
	    translate_max_y: 8.0
	  }
	  color_augmentation {
	    hue_rotation_max: 25.0
	    saturation_shift_max: 0.20000000298
	    contrast_scale_max: 0.10000000149
	    contrast_center: 0.5
	  }
	}
	dataset_config {
	  data_sources: {
	    tfrecords_path: "/workspace/tlt-experiments/data/tfrecords/kitti_trainval/kitti_trainval*"
	    image_directory_path: "/workspace/tlt-experiments/data/SIIM-FISABIO-RSNA/formatted"
	  }
	  image_extension: "png"
	  target_class_mapping {
	      key: "opacity"
	      value: "opacity"
	  }
	validation_fold: 0
	}

7. Train the model
------------------
Create a directory for storing model weights (unpruned).

::

	!mkdir -p $USER_EXPERIMENT_DIR/experiment_dir_unpruned

Start training

::

	!tlt-train ssd -e $SPECS_DIR/covid_tlt_train.txt \
               -r $USER_EXPERIMENT_DIR/experiment_dir_unpruned \
               -k $KEY \
               -m $USER_EXPERIMENT_DIR/pretrained_resnet18/tlt_pretrained_object_detection_vresnet18/resnet_18.hdf5 \
               --gpus 4

* :code:`-e`: Specify training specification file
* :code:`-r`: Specify the output directory for unpruned models
* :code:`-k`: Your ngc API key.  
* :code:`-m`: Pretrained model's path.
* :code:`--gpus`: Configure this property if you have multiple GPUs (e.g. GPU accelerators).

Training results after 80 epoch

::

	Epoch 00078: saving model to /workspace/tlt-experiments/ssd/experiment_dir_unpruned/weights/ssd_resnet18_epoch_078.tlt
	Epoch 79/80
	85/85 [==============================] - 18s 210ms/step - loss: 1.7293

	Epoch 00079: saving model to /workspace/tlt-experiments/ssd/experiment_dir_unpruned/weights/ssd_resnet18_epoch_079.tlt
	Epoch 80/80
	85/85 [==============================] - 18s 212ms/step - loss: 1.6824

	Epoch 00080: saving model to /workspace/tlt-experiments/ssd/experiment_dir_unpruned/weights/ssd_resnet18_epoch_080.tlt
	Number of images in the evaluation dataset: 886

	Producing predictions: 100% 56/56 [00:10<00:00,  5.30it/s]
	Start multi-thread per-image matching
	Start to calculate AP for each class
	*******************************
	opacity       AP    0.492
	              mAP   0.492
	*******************************

8. Evaluate the model
---------------------
You might want to evaluate a specific model weight (model weight is saved for each epoch in :file:`$USER_EXPERIMENT_DIR/experiment_dir_unpruned/weights/`)

::

	!tlt-evaluate ssd -e $SPECS_DIR/ssd_train_resnet18_kitti.txt \
                  -m $USER_EXPERIMENT_DIR/experiment_dir_unpruned/weights/ssd_resnet18_epoch_$EPOCH.tlt \
                  -k $KEY

* :code:`-m`: Use specific model weights' path for evaluation.

9. Prune the model
------------------
::

	%set_env EPOCH=080
	!mkdir -p $USER_EXPERIMENT_DIR/experiment_dir_pruned

	!tlt-prune -m $USER_EXPERIMENT_DIR/experiment_dir_unpruned/weights/ssd_resnet18_epoch_$EPOCH.tlt \
	           -o $USER_EXPERIMENT_DIR/experiment_dir_pruned/ssd_resnet18_pruned.tlt \
	           -eq intersection \
	           -pth 0.1 \
	           -k $KEY

* :code:`%set_env EPOCH=080`: Configure this property to prune a specific model output from this epoch in previous training.
* :code:`-pth`: the threshold for trade off between accuracy and model size. A higher value will bring faster inference but lower accuracy, while the lower bring slower faster inference but higher accuracy.

Verify the pruned model::

	!ls -rlt $USER_EXPERIMENT_DIR/experiment_dir_pruned/

10. Prepare retrain specification file
--------------------------------------
:file:`tlt-covid-specs/covid_tlt_retrain.txt`:

::

	%%writefile tlt-covid-specs/covid_tlt_retrain.txt
	random_seed: 42
	ssd_config {
	  aspect_ratios_global: "[1.0, 2.0, 0.5, 3.0, 1.0/3.0]"
	  scales: "[0.05, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85]"
	  two_boxes_for_ar1: true
	  clip_boxes: false
	  loss_loc_weight: 0.8
	  focal_loss_alpha: 0.25
	  focal_loss_gamma: 2.0
	  variances: "[0.1, 0.1, 0.2, 0.2]"
	  arch: "resnet"
	  nlayers: 18
	  freeze_bn: false
	}
	training_config {
	  batch_size_per_gpu: 32
	  num_epochs: 80
	  enable_qat: false
	  learning_rate {
	  soft_start_annealing_schedule {
	    min_learning_rate: 5e-5
	    max_learning_rate: 2e-2
	    soft_start: 0.1
	    annealing: 0.6
	    }
	  }
	  regularizer {
	    type: NO_REG
	    weight: 3e-9
	  }
	}
	eval_config {
	  validation_period_during_training: 10
	  average_precision_mode: SAMPLE
	  batch_size: 32
	  matching_iou_threshold: 0.5
	}
	nms_config {
	  confidence_threshold: 0.01
	  clustering_iou_threshold: 0.6
	  top_k: 200
	}
	augmentation_config {
	  preprocessing {
	    output_image_width: 512
	    output_image_height: 512
	    output_image_channel: 3
	    crop_right: 512
	    crop_bottom: 512
	    min_bbox_width: 1.0
	    min_bbox_height: 1.0
	  }
	  spatial_augmentation {
	    hflip_probability: 0.5
	    vflip_probability: 0.0
	    zoom_min: 0.7
	    zoom_max: 1.8
	    translate_max_x: 8.0
	    translate_max_y: 8.0
	  }
	  color_augmentation {
	    hue_rotation_max: 25.0
	    saturation_shift_max: 0.20000000298
	    contrast_scale_max: 0.10000000149
	    contrast_center: 0.5
	  }
	}
	dataset_config {
	  data_sources: {
	    tfrecords_path: "/workspace/tlt-experiments/data/tfrecords/kitti_trainval/kitti_trainval*"
	    image_directory_path: "/workspace/tlt-experiments/data/SIIM-FISABIO-RSNA/formatted"
	  }
	  image_extension: "png"
	  target_class_mapping {
	      key: "opacity"
	      value: "opacity"
	  }

	validation_fold: 0
	}

11. Retrain the pruned mdoel
----------------------------
Create a directory for storing model weights (pruned).

::

	!mkdir -p $USER_EXPERIMENT_DIR/experiment_dir_retrain

Start training

::

	!tlt-train ssd --gpus 4 \
               -e $SPECS_DIR/covid_tlt_retrain.txt \
               -r $USER_EXPERIMENT_DIR/experiment_dir_retrain \
               -m $USER_EXPERIMENT_DIR/experiment_dir_pruned/ssd_resnet18_pruned.tlt \
               -k $KEY


Training results after 80 epoch::

	Epoch 79/80
	42/42 [==============================] - 13s 312ms/step - loss: 1.1341

	Epoch 00079: saving model to /workspace/tlt-experiments/ssd/experiment_dir_retrain/weights/ssd_resnet18_epoch_079.tlt
	Epoch 80/80
	42/42 [==============================] - 13s 309ms/step - loss: 1.1565

	Epoch 00080: saving model to /workspace/tlt-experiments/ssd/experiment_dir_retrain/weights/ssd_resnet18_epoch_080.tlt
	Number of images in the evaluation dataset: 886

	Producing predictions: 100% 28/28 [00:10<00:00,  2.73it/s]
	Start multi-thread per-image matching
	Start to calculate AP for each class
	*******************************
	opacity       AP    0.485
	              mAP   0.485
	*******************************


12. Evaluate the model
----------------------


...
---