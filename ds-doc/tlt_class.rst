Transfer Learning Toolkit - Classification (TLT2)
=================================================

TLT can be used to train classification model with custom dataset and pretrained models. In this sample training, we retrain a classification model with resnet 18 as backbone using VOC2012 dataset. Likewise to detection models, TLT expects a designated structure for custom dataset which be demonstrated with sample TLT training below. Also, after we re-train the model, we will export it to a TRT inference engine and deploy it on DeepStream.

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
::

	%env USER_EXPERIMENT_DIR=/workspace/tlt-experiments/classification
	%env DATA_DOWNLOAD_DIR=/workspace/tlt-experiments/data
	%env SPECS_DIR=/workspace/examples/classification/specs
	%env KEY=<your-ngcapi-key>

1. Prepare dataset and pretrained model
---------------------------------------

Download VOC2012 dataset::

	!wget https://data.deepai.org/PascalVOC2012.zip -O $DATA_DOWNLOAD_DIR/PascalVOC2012.zip
	!unzip $DATA_DOWNLOAD_DIR/PascalVOC2012.zip -d $DATA_DOWNLOAD_DIR

Verify downloaded VOC2012 dataset::

	!ls -l $DATA_DOWNLOAD_DIR/VOC2012

Format VOC2012 dataset

.. code-block:: python

	from os.path import join as join_path
	import os
	import glob
	import re
	import shutil

	DATA_DIR=os.environ.get('DATA_DOWNLOAD_DIR')
	source_dir = join_path(DATA_DIR, "VOCdevkit/VOC2012")
	target_dir = join_path(DATA_DIR, "formatted")


	suffix = '_trainval.txt'
	classes_dir = join_path(source_dir, "ImageSets", "Main")
	images_dir = join_path(source_dir, "JPEGImages")
	classes_files = glob.glob(classes_dir+"/*"+suffix)
	for file in classes_files:
	    # get the filename and make output class folder
	    classname = os.path.basename(file)
	    if classname.endswith(suffix):
	        classname = classname[:-len(suffix)]
	        target_dir_path = join_path(target_dir, classname)
	        if not os.path.exists(target_dir_path):
	            os.makedirs(target_dir_path)
	    else:
	        continue
	    print(classname)


	    with open(file) as f:
	        content = f.readlines()


	    for line in content:
	        tokens = re.split('\s+', line)
	        if tokens[1] == '1':
	            # copy this image into target dir_path
	            target_file_path = join_path(target_dir_path, tokens[0] + '.jpg')
	            src_file_path = join_path(images_dir, tokens[0] + '.jpg')
	            shutil.copyfile(src_file_path, target_file_path)

.. code-block:: python

	import os
	import glob
	import shutil
	from random import shuffle
	from tqdm import tqdm_notebook as tqdm

	DATA_DIR=os.environ.get('DATA_DOWNLOAD_DIR')
	SOURCE_DIR=join_path(DATA_DIR, 'formatted')
	TARGET_DIR=os.path.join(DATA_DIR,'split')
	# list dir
	dir_list = next(os.walk(SOURCE_DIR))[1]
	# for each dir, create a new dir in split
	for dir_i in tqdm(dir_list):
	        newdir_train = os.path.join(TARGET_DIR, 'train', dir_i)
	        newdir_val = os.path.join(TARGET_DIR, 'val', dir_i)
	        newdir_test = os.path.join(TARGET_DIR, 'test', dir_i)
	        
	        if not os.path.exists(newdir_train):
	                os.makedirs(newdir_train)
	        if not os.path.exists(newdir_val):
	                os.makedirs(newdir_val)
	        if not os.path.exists(newdir_test):
	                os.makedirs(newdir_test)

	        img_list = glob.glob(os.path.join(SOURCE_DIR, dir_i, '*.jpg'))
	        # shuffle data
	        shuffle(img_list)

	        for j in range(int(len(img_list)*0.7)):
	                shutil.copy2(img_list[j], os.path.join(TARGET_DIR, 'train', dir_i))

	        for j in range(int(len(img_list)*0.7), int(len(img_list)*0.8)):
	                shutil.copy2(img_list[j], os.path.join(TARGET_DIR, 'val', dir_i))
	                
	        for j in range(int(len(img_list)*0.8), len(img_list)):
	                shutil.copy2(img_list[j], os.path.join(TARGET_DIR, 'test', dir_i))
	                
	print('Done splitting dataset.')

Next cell, write a function to verify the split dataset

.. code-block:: python

	def verify_class(path):
	    classes = os.listdir(path)
	    total_images = 0
	    for i in classes:
	            current_class = os.listdir(f"{path}/{i}")
	            class_count = len(current_class)
	            total_images += class_count
	            print(f"{class_count} images found in {i} class")
	    print(f"summary: {total_images} images found in all classes")

Verify the split dataset

.. code-block:: python
	
	verify_class("/workspace/tlt-experiments/data/formatted")

::

	670 images found in aeroplane class
	552 images found in bicycle class
	765 images found in bird class
	508 images found in boat class
	706 images found in bottle class
	421 images found in bus class
	1161 images found in car class
	1080 images found in cat class
	1119 images found in chair class
	303 images found in cow class
	538 images found in diningtable class
	1286 images found in dog class
	482 images found in horse class
	526 images found in motorbike class
	4087 images found in person class
	527 images found in pottedplant class
	325 images found in sheep class
	507 images found in sofa class
	544 images found in train class
	575 images found in tvmonitor class
	summary: 16682 images found in all classes

Download pretrained model for classification:

::

	!ngc registry model list nvidia/tlt_pretrained_classification:*

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
	| resne | 77.78 | 80    | 1     | V100  | 576.3 | 576.3 | UPLOA | Apr   |
	| t101  |       |       |       |       |       | 3 MB  | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2020  |
	| resne | 74.38 | 80    | 1     | V100  | 38.3  | 38.31 | UPLOA | Apr   |
	| t10   |       |       |       |       |       | MB    | D_COM | 29,   |
	|       |       |       |       |       |       |       | PLETE | 2021  |
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

Downloda pretrained Resnet18::

	!mkdir -p $USER_EXPERIMENT_DIR/pretrained_resnet18/
	!ngc registry model download-version nvidia/tlt_pretrained_classification:resnet18 --dest $USER_EXPERIMENT_DIR/pretrained_resnet18

Verify pretrained resnet18::

	!ls -l $USER_EXPERIMENT_DIR/pretrained_resnet18/tlt_pretrained_classification_vresnet18

2. Prepare training specification file
--------------------------------------
:code:`$SPECS_DIR/classification_spec.cfg`:
::

	model_config {
	  arch: "resnet",
	  n_layers: 18
	  # Setting these parameters to true to match the template downloaded from NGC.
	  use_batch_norm: true
	  all_projections: true
	  freeze_blocks: 0
	  freeze_blocks: 1
	  input_image_size: "3,224,224"
	}
	train_config {
	  train_dataset_path: "/workspace/tlt-experiments/data/split/train"
	  val_dataset_path: "/workspace/tlt-experiments/data/split/val"
	  pretrained_model_path: "/workspace/tlt-experiments/classification/pretrained_resnet18/tlt_pretrained_classification_vresnet18/resnet_18.hdf5"
	  optimizer: "sgd"
	  batch_size_per_gpu: 64
	  n_epochs: 80
	  n_workers: 16

	  # regularizer
	  reg_config {
	    type: "L2"
	    scope: "Conv2D,Dense"
	    weight_decay: 0.00005
	  }

	  # learning_rate
	  lr_config {
	    scheduler: "step"
	    learning_rate: 0.006
	    #soft_start: 0.056
	    #annealing_points: "0.3, 0.6, 0.8"
	    #annealing_divider: 10
	    step_size: 10
	    gamma: 0.1
	  }
	}
	eval_config {
	  eval_dataset_path: "/workspace/tlt-experiments/data/split/test"
	  model_path: "/workspace/tlt-experiments/classification/output/weights/resnet_080.tlt"
	  top_k: 3
	  batch_size: 256
	  n_workers: 8
	}

For more details about training specification file, see :ref:`tlt_specs_class`



3. Run TLT training
-------------------
