Transfer Learning Toolkit - Specification Files
===============================================
Specification files define how formatted datasets are being converted to TFrecords for training, and properties or parameters to define how models are being trained or re-trained.


Train Dectection models
-----------------------
Typically, you will need to supply three specification files for training the detection models with TLT which are *TFrecords conversion specification file*, *Training specification file*, and *Retrain specification file*


TFrecords conversion specification file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

	kitti_config {
	  root_directory_path: <directory containing image directory and label directory>
	  image_dir_name: <image-dir-name>
	  label_dir_name: <label-dir-name>
	  image_extension: ".png"
	  partition_mode: "random"
	  num_partitions: 2
	  val_split: 14
	  num_shards: 10
	}

* For each label in :file:`label_dir_name`, one must have `Kitti format <https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt>`_.
* For each image and label, they must have one-one correspondence. For example, a label file :file:`<label_dir_name>/000001.txt` describes object details in :file:`<image_dir_name>/000001.png`.

Once your own datasets prepared (e.g. from Kaggle) is being formatted in a way described above, configure :code:`root_directory`, :code:`image_dir_name`, and :code:`label_dir_name`.

The portion validation set is defined by :code:`val_split`.


Training specification file
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sample training specification file :file:`/workspace/examples/ssd/specs/ssd_train_resnet18_kitti.txt`.

::

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
	  # arch: "darknet"
	  nlayers: 18
	  # nlayers: 53
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

* :code:`ssd_config`: properties for SSD backbone defining behaviour during training the pretrained model. 

	* If you want to try different pretrained models available on NGC, configure :code:`arch` and :code:`nlayers`

	* You may want to explore how detection backbones config differ in different training specification files under :file:`/workspace/examples/`

* More often, you pay attention to :code:`dataset_config` as you have your own datasets.

	* :code:`tfrecords_path`: during model training with TLT, tfrecords will be stored to :code:`tfrecords_path`. Configure the path for this property.

	* :code:`image_extension`: make sure all your images share same extension (e.g. png)

	* :code:`target_class_mapping`: say if you have 4 classes of objects, car, cyclist, pedestrain, van, but you only want 3 detection classes at the end (you treat van as car), in such case, you will add 4 :code:`target_class_mapping` in :code:`dataset_config`

	.. code-block::

		target_class_mapping {
		    key: "car"
		    value: "car"
		  }
		target_class_mapping {
		    key: "cyclist"
		    value: "cyclist"
		  }
		target_class_mapping {
		    key: "pedestrian"
		    value: "pedestrian"
		  }
		target_class_mapping {
		    key: "van"
		    value: "car"
		  }


Retrain specification file
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

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
	  # arch: "darknet"
	  nlayers: 18
	  # nlayers: 53
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


Train Classification models
---------------------------