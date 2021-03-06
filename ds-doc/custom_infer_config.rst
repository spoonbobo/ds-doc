Custom Model - Triton Inference Server Configurations
=====================================================

See `nvinferserver <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinferserver.html>`_ for Triton configurations in DeepStream.

App configurations
------------------

Similar to :ref:`ds_config`, we will need a DeepStream app config to run DeepStream. For example, :file:`source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt` in */opt/nvidia/deepstream/deepstream-5.1/samples/configs/deepstream-app-trtis*::

	################################################################################
	# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
	#
	# Permission is hereby granted, free of charge, to any person obtaining a
	# copy of this software and associated documentation files (the "Software"),
	# to deal in the Software without restriction, including without limitation
	# the rights to use, copy, modify, merge, publish, distribute, sublicense,
	# and/or sell copies of the Software, and to permit persons to whom the
	# Software is furnished to do so, subject to the following conditions:
	#
	# The above copyright notice and this permission notice shall be included in
	# all copies or substantial portions of the Software.
	#
	# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
	# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
	# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
	# DEALINGS IN THE SOFTWARE.
	################################################################################

	[application]
	enable-perf-measurement=1
	perf-measurement-interval-sec=5
	#gie-kitti-output-dir=streamscl

	[tiled-display]
	enable=1
	rows=2
	columns=2
	width=1280
	height=720
	gpu-id=0
	#(0): nvbuf-mem-default - Default memory allocated, specific to particular platform
	#(1): nvbuf-mem-cuda-pinned - Allocate Pinned/Host cuda memory, applicable for Tesla
	#(2): nvbuf-mem-cuda-device - Allocate Device cuda memory, applicable for Tesla
	#(3): nvbuf-mem-cuda-unified - Allocate Unified cuda memory, applicable for Tesla
	#(4): nvbuf-mem-surface-array - Allocate Surface Array memory, applicable for Jetson
	nvbuf-memory-type=0

	[source0]
	enable=1
	#Type - 1=CameraV4L2 2=URI 3=MultiURI 4=RTSP
	type=3
	uri=file://../../streams/sample_1080p_h264.mp4
	num-sources=4
	#drop-frame-interval=2
	gpu-id=0
	# (0): memtype_device   - Memory type Device
	# (1): memtype_pinned   - Memory type Host Pinned
	# (2): memtype_unified  - Memory type Unified
	cudadec-memtype=0

	[sink0]
	enable=1
	#Type - 1=FakeSink 2=EglSink 3=File
	type=2
	sync=1
	source-id=0
	gpu-id=0
	nvbuf-memory-type=0

	[sink1]
	enable=0
	#Type - 1=FakeSink 2=EglSink 3=File 4=RTSPStreaming
	type=3
	#1=mp4 2=mkv
	container=1
	#1=h264 2=h265
	codec=1
	#encoder type 0=Hardware 1=Software
	enc-type=0
	sync=0
	#iframeinterval=10
	bitrate=2000000
	#H264 Profile - 0=Baseline 2=Main 4=High
	#H265 Profile - 0=Main 1=Main10
	profile=0
	output-file=out.mp4
	source-id=0

	[sink2]
	enable=0
	#Type - 1=FakeSink 2=EglSink 3=File 4=RTSPStreaming
	type=4
	#1=h264 2=h265
	codec=1
	#encoder type 0=Hardware 1=Software
	enc-type=0
	sync=0
	#iframeinterval=10
	bitrate=400000
	#H264 Profile - 0=Baseline 2=Main 4=High
	#H265 Profile - 0=Main 1=Main10
	profile=0
	# set below properties in case of RTSPStreaming
	rtsp-port=8554
	udp-port=5400

	[osd]
	enable=1
	gpu-id=0
	border-width=1
	text-size=15
	text-color=1;1;1;1;
	text-bg-color=0.3;0.3;0.3;1
	font=Serif
	show-clock=0
	clock-x-offset=800
	clock-y-offset=820
	clock-text-size=12
	clock-color=1;0;0;0
	nvbuf-memory-type=0

	[streammux]
	gpu-id=0
	##Boolean property to inform muxer that sources are live
	live-source=0
	batch-size=4
	##time out in usec, to wait after the first buffer is available
	##to push the batch even if the complete batch is not formed
	batched-push-timeout=40000
	## Set muxer output width and height
	width=1920
	height=1080
	##Enable to maintain aspect ratio wrt source, and allow black borders, works
	##along with width, height properties
	enable-padding=0
	nvbuf-memory-type=0
	## If set to TRUE, system timestamp will be attached as ntp timestamp
	## If set to FALSE, ntp timestamp from rtspsrc, if available, will be attached
	# attach-sys-ts-as-ntp=1

	# config-file property is mandatory for any gie section.
	# Other properties are optional and if set will override the properties set in
	# the infer config file.
	[primary-gie]
	enable=1
	gpu-id=0
	#(0): nvinfer - Default inference plugin based on Tensorrt
	#(1): nvinferserver - inference plugin based on Tensorrt-Inference-Server
	plugin-type=1
	batch-size=4
	#Required by the app for OSD, not a plugin property
	bbox-border-color0=1;0;0;1
	bbox-border-color1=0;1;1;1
	bbox-border-color2=0;0;1;1
	bbox-border-color3=0;1;0;1
	interval=0
	gie-unique-id=1
	nvbuf-memory-type=0
	config-file=config_infer_plan_engine_primary.txt

	[tracker]
	enable=1
	tracker-width=640
	tracker-height=384
	#ll-lib-file=/opt/nvidia/deepstream/deepstream-5.1/lib/libnvds_mot_iou.so
	#ll-lib-file=/opt/nvidia/deepstream/deepstream-5.1/lib/libnvds_nvdcf.so
	ll-lib-file=/opt/nvidia/deepstream/deepstream-5.1/lib/libnvds_mot_klt.so
	#ll-config-file required for DCF/IOU only
	#ll-config-file=../deepstream-app/tracker_config.yml
	#ll-config-file=../deepstream-app/iou_config.txt
	gpu-id=0
	#enable-batch-process applicable to DCF only
	enable-batch-process=1
	display-tracking-id=1

	[secondary-gie0]
	enable=1
	#(0): nvinfer; (1): nvinferserver
	plugin-type=1
	# nvinferserserver's gpu-id can only set from its own config-file
	#gpu-id=0
	batch-size=16
	gie-unique-id=4
	operate-on-gie-id=1
	operate-on-class-ids=0;
	config-file=config_infer_secondary_plan_engine_vehicletypes.txt

	[secondary-gie1]
	enable=1
	plugin-type=1
	batch-size=16
	gie-unique-id=5
	operate-on-gie-id=1
	operate-on-class-ids=0;
	config-file=config_infer_secondary_plan_engine_carcolor.txt

	[secondary-gie2]
	enable=1
	plugin-type=1
	batch-size=16
	gie-unique-id=6
	operate-on-gie-id=1
	operate-on-class-ids=0;
	config-file=config_infer_secondary_plan_engine_carmake.txt

	[tests]
	file-loop=0

As you might spot, the content of this file is similar to :file:`source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt` in *../deepstream-app*. The difference is that *nvinferserver* plugin will be used instead of *nvinfer* with the property :code:`plugin-type` equals to 1 under :code:`primary-gie` and :code:`secondary-gieX`. A :code:`config-file` is needed for each gie. We will inspect the gie config.


Gie configurations
------------------
Sample gie config file :file:`config_infer_plan_engine_primary.txt` located in */opt/nvidia/deepstream/deepstream-5.1/samples/configs/deepstream-app-trtis*

::

	infer_config {
	  unique_id: 1
	  gpu_ids: [0]
	  max_batch_size: 30
	  backend {
	    inputs: [ {
	      name: "input_1"
	    }]
	    outputs: [
	      {name: "conv2d_bbox"},
	      {name: "conv2d_cov/Sigmoid"}
	    ]
	    trt_is {
	      model_name: "Primary_Detector"
	      version: -1
	      model_repo {
	        root: "../../trtis_model_repo"
	        strict_model_config: true
	      }
	    }
	  }

	  preprocess {
	    network_format: MEDIA_FORMAT_NONE
	    tensor_order: TENSOR_ORDER_LINEAR
	    tensor_name: "input_1"
	    maintain_aspect_ratio: 0
	    frame_scaling_hw: FRAME_SCALING_HW_DEFAULT
	    frame_scaling_filter: 1
	    normalize {
	      scale_factor: 0.0039215697906911373
	      channel_offsets: [0, 0, 0]
	    }
	  }

	  postprocess {
	    labelfile_path: "../../models/Primary_Detector/labels.txt"
	    detection {
	      num_detected_classes: 4
	      group_rectangle {
	        confidence_threshold: 0.2
	        group_threshold: 1
	        eps:0.2
	      }
	    }
	  }

	  extra {
	    copy_input_to_host_buffers: false
	    output_buffer_pool_size: 2
	  }
	}
	input_control {
	  process_mode: PROCESS_MODE_FULL_FRAME
	  operate_on_gie_id: -1
	  interval: 0
	}


Inference config
~~~~~~~~~~~~~~~~

Refer to :code:`infer_config` group in gie config.

* :code:`backend`: Include 3 parameters :code:`inputs`, :code:`outputs`, and :code:`trt_is`. See `BackendParams <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinferserver.html#id10>`_ for more details
	
	* :code:`inputs`: 

		* :code:`name`: Input tensor name

	* :code:`outputs`:

		* :code:`name`: Output tensor name

	* :code:`model_name`: The name of model under model repo. See :ref:`model_repo_structure` for more details.

	* :code:`model_repo`:

		* See `TritonModelRepo <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinferserver.html#id15>`_ more details

* :code:`preprocess`: 
	
	* :code:`network_format`: media format

	* :code:`tensor_order`: Select from :code:`TENSOR_ORDER_NONE`, :code:`TENSOR_ORDER_LINEAR`, and :code:`TENSOR_ORDER_NHWC`. 

	* :code:`tensor_name`: Refer to :code:`inputs` under :code:`infer_config`

	* :code:`maintain_aspect_ratio`: set 1 to maintain aspect ratio while performing scaling.

	* :code:`frame_scaling_hw`: Compute hardware to use for scaling frames or cropping the objects to network resolution

	* :code:`frame_scaling_filter`: Refer to :ref:`deepstream_infer_config` to see options.

	* :code:`normalize`: Refer to :ref:`deepstream_infer_config` for more details.

	See `PreProcessParams <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinferserver.html#id16>`_ for more details

* :code:`postprocess`:

	* :code:`detection`:

		* :code:`group_rectangle`: clustering bounding boxes.

		See `DetectionParams <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinferserver.html#id19>`_ 

See `Inference configurations definition <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinferserver.html#id10>`_ to see full details of infer config group.


Input control config
~~~~~~~~~~~~~~~~~~~~

* :code:`process_mode`: By default, select :code:`PROCESS_MODE_FULL_FRAME` for PGIE, or :code:`PROCESS_MODE_CLIP_OBJECTS` for SGIE.


See `Input control definition <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinferserver.html#id3>`_ to see full details of input control config.


.. _model_repo_structure:

Triton model repo and config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Triton model repository
***********************

Sample :file:`trtis_model_repo` located in */opt/nvidia/deepstream/deepstream-5.1/samples/*::

	.
	|-- Primary_Detector
	|   |-- 1
	|   |   `-- resnet10.caffemodel_b30_gpu0_int8.engine
	|   `-- config.pbtxt
	|-- Secondary_CarColor
	|   |-- 1
	|   |   `-- resnet18.caffemodel_b16_gpu0_int8.engine
	|   `-- config.pbtxt
	|-- Secondary_CarMake
	|   |-- 1
	|   |   `-- resnet18.caffemodel_b16_gpu0_int8.engine
	|   `-- config.pbtxt
	|-- Secondary_VehicleTypes
	|   |-- 1
	|   |   `-- resnet18.caffemodel_b16_gpu0_int8.engine
	|   `-- config.pbtxt
	|-- Segmentation_Industrial
	|   |-- 1
	|   |   `-- unet_output_graph.uff_b1_gpu0_fp32.engine
	|   `-- config.pbtxt
	|-- Segmentation_Semantic
	|   |-- 1
	|   |   `-- unetres18_v4_pruned0.65_800_data.uff_b1_gpu0_fp32.engine
	|   `-- config.pbtxt
	|-- densenet_onnx
	|   |-- 1
	|   |   `-- model.onnx
	|   |-- config.pbtxt
	|   `-- densenet_labels.txt
	|-- inception_graphdef
	|   |-- 1
	|   |   `-- model.graphdef
	|   |-- config.pbtxt
	|   `-- inception_labels.txt
	|-- mobilenet_v1
	|   |-- 1
	|   |   `-- model.graphdef
	|   |-- config.pbtxt
	|   `-- labels.txt
	|-- ssd_inception_v2_coco_2018_01_28
	|   |-- 1
	|   |   `-- model.graphdef
	|   |-- config.pbtxt
	|   `-- labels.txt
	`-- ssd_mobilenet_v1_coco_2018_01_28
	    |-- 1
	    |   `-- frozen_inference_graph.pb
	    |-- config.pbtxt
	    `-- labels.txt

Triton models are stored in model repositories which follows the follwing structure::

	<model-repository-path>/
	    <model-name>/
	      [config.pbtxt]
	      [<output-labels-file> ...]
	      <version>/
	        <model-definition-file>
	      <version>/
	        <model-definition-file>
	      ...
	    <model-name>/
	      [config.pbtxt]
	      [<output-labels-file> ...]
	      <version>/
	        <model-definition-file>
	      <version>/
	        <model-definition-file>
	      ...
	    ...

**Minimal model repository for models:**

*TensorRT model*
::

	<model-repository-path>/
	  <model-name>/
	    config.pbtxt
	    1/
	      model.plan

*ONNX model*
::

	<model-repository-path>/
      <model-name>/
        config.pbtxt
        1/
          model.onnx

*TorchScript model*
::

	<model-repository-path>/
      <model-name>/
        config.pbtxt
        1/
          model.pt


*Tensorflow model*
::

	<model-repository-path>/
      <model-name>/
        config.pbtxt
        1/
          model.graphdef

::

	<model-repository-path>/
      <model-name>/
        config.pbtxt
        1/
          model.savedmodel/
             <saved-model files>


*Python model*
::

	<model-repository-path>/
      <model-name>/
        config.pbtxt
        1/
          model.py


*DALI model* 
::

	<model-repository-path>/
      <model-name>/
        config.pbtxt
        1/
          model.dali


See `Model repository <https://github.com/triton-inference-server/server/blob/r20.12/docs/model_repository.md>`_ Model repository for Triton models

Triton model configuration
**************************
Sample model config file :file:`config.pbtxt` located in */opt/nvidia/deepstream/deepstream-5.1/samples/trtis_model_repo/Primary_Detector*.

::

	name: "Primary_Detector"
	platform: "tensorrt_plan"
	max_batch_size: 30
	default_model_filename: "resnet10.caffemodel_b30_gpu0_int8.engine"
	input [
	  {
	    name: "input_1"
	    data_type: TYPE_FP32
	    format: FORMAT_NCHW
	    dims: [ 3, 368, 640 ]
	  }
	]
	output [
	  {
	    name: "conv2d_bbox"
	    data_type: TYPE_FP32
	    dims: [ 16, 23, 40 ]
	  },

	  {
	    name: "conv2d_cov/Sigmoid"
	    data_type: TYPE_FP32
	    dims: [ 4, 23, 40 ]
	  }
	]
	instance_group [
	  {
	    kind: KIND_GPU
	    count: 1
	    gpus: 0
	  }
	]

Major parameters of Triton model config

* platform and/or backend properties

	See `Triton Inference Server Backend <https://github.com/triton-inference-server/backend/blob/main/README.md#backends>`_ for more details.

* :code:`max_batch_size`

* :code:`input`

* :code:`output`

Optional parameters of Triton Model config

* :code:`version_policy`
	
	::

		version_policy: { all { }}

* :code:`instance_group`

	::

		instance_group [
	  {
	    kind: KIND_GPU
	    count: 1
	    gpus: 0
	  }

* :code:`dynamic_batching`

	::

		  dynamic_batching {
		    preferred_batch_size: [ 4, 8 ]
		    max_queue_delay_microseconds: 100
		  }

See `Model configurations <https://github.com/triton-inference-server/server/blob/r20.12/docs/model_configuration.md>`_ Triton model config.
