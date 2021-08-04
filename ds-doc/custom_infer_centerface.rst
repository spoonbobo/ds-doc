Custom Model - Centerface (Triton)
==================================

Sample deployment of Centerface network in Triton Inference Server on AGX Xavier. For more details, see `Centerface repo <https://github.com/NVIDIA-AI-IOT/deepstream_triton_model_deploy/tree/master/centerface>`_ for more details.

Prerequisites
-------------

* Jetson AGX Xavier
* DeepStream SDK 5.1

*Follow this sample in DeepStream NGC Container or on Nvidia AGX Xavier.*


1. Clone the repo and download models
-------------------------------------

Let's download onnx package and the `Centernet project repo <https://github.com/NVIDIA-AI-IOT/deepstream_triton_model_deploy>`_ under */opt/nvidia/deepstream/deepstream-5.1/samples*::

	pip install onnx
	cd /opt/nvidia/deepstream/deepstream-5.1/samples/
	git clone https://github.com/NVIDIA-AI-IOT/deepstream_triton_model_deploy
	cd deepstream_triton_model_deploy/centerface/centerface/1/
	./run.sh

2. Inspect DeepStream app config
--------------------------------

app config :file:`source1_primary_detector.txt` is in */opt/nvidia/deepstream/deepstream-5.1/samples/deepstream_triton_model_deploy/centerface/config*

::

	[application]
	enable-perf-measurement=1
	perf-measurement-interval-sec=5
	#gie-kitti-output-dir=kitti-trtis

	[tiled-display]
	enable=1
	rows=1
	columns=1
	width=1280
	height=720
	gpu-id=0
	#(0): nvbuf-mem-default - Default memory allocated, specific to particular platform
	#(1): nvbuf-mem-cuda-pinned - Allocate Pinned/Host cuda memory applicable for Tesla
	#(2): nvbuf-mem-cuda-device - Allocate Device cuda memory applicable for Tesla
	#(3): nvbuf-mem-cuda-unified - Allocate Unified cuda memory applicable for Tesla
	#(4): nvbuf-mem-surface-array - Allocate Surface Array memory, applicable for Jetson
	nvbuf-memory-type=0

	[source0]
	enable=1
	#Type - 1=CameraV4L2 2=URI 3=MultiURI 4=RTSP
	type=2
	#uri=file://../../../samples/configs/tlt_pretrained_models/Redaction-A_1.mp4
	uri=file:///opt/nvidia/deepstream/deepstream-5.1/samples/streams/sample_1080p_h264.mp4
	num-sources=1
	#drop-frame-interval=2
	gpu-id=0
	# (0): memtype_device   - Memory type Device
	# (1): memtype_pinned   - Memory type Host Pinned
	# (2): memtype_unified  - Memory type Unified
	cudadec-memtype=0

	[sink0]
	enable=0
	#Type - 1=FakeSink 2=EglSink 3=File
	type=1
	sync=0
	source-id=0
	gpu-id=0
	nvbuf-memory-type=0

	[sink1]
	enable=1
	type=3
	#1=mp4 2=mkv
	container=1
	#1=h264 2=h265
	codec=1
	sync=0
	#iframeinterval=10
	bitrate=2000000
	output-file=out.mp4
	source-id=0

	[sink2]
	enable=0
	#Type - 1=FakeSink 2=EglSink 3=File 4=RTSPStreaming
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
	batch-size=1
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

	# config-file property is mandatory for any gie section.
	# Other properties are optional and if set will override the properties set in
	# the infer config file.
	[primary-gie]
	enable=1
	#(0): nvinfer; (1): nvinferserver
	plugin-type=1
	#infer-raw-output-dir=trtis-output
	batch-size=1
	interval=0
	gie-unique-id=1
	bbox-border-color0=1;0;0;1
	bbox-border-color1=0;1;1;1
	#bbox-border-color2=0;0;1;1
	#bbox-border-color3=0;1;0;1
	config-file=centerface.txt

	[tests]
	file-loop=0

3. Inspect gie config file
--------------------------

gie config :file:`centerface.txt` is in the same directory with :file:`source1_primary_detector.txt`.

::

	infer_config {
	  unique_id: 1
	  gpu_ids: 0
	  max_batch_size: 1
	  backend {
	    inputs [
	      {
	        name: "input.1"
	        dims: [3, 480, 640]
	      }
	    ]
	    trt_is {
	      model_name: "centerface"
	      version: -1
	      model_repo {
	        root: "../"
	        log_level: 1
	        tf_gpu_memory_fraction: 0.2
	        tf_disable_soft_placement: 0
	      }
	    }
	  }

	  preprocess {
	    network_format: IMAGE_FORMAT_RGB
	    tensor_order: TENSOR_ORDER_LINEAR
	    maintain_aspect_ratio: 0
	    normalize {
	      scale_factor: 1.0
	      channel_offsets: [0, 0, 0]
	    }
	  }

	  postprocess {
	    labelfile_path: "../centerface/centerface_labels.txt"
	    detection {
	      num_detected_classes: 1
	      custom_parse_bbox_func: "NvDsInferParseCustomCenterNetFace"
	      simple_cluster {
	        threshold: 0.3
	      }
	    }
	  }

	  custom_lib {
	    path: "../customparser/libnvds_infercustomparser_centernet.so"
	  }

	  extra {
	    copy_input_to_host_buffers: false
	  }
	}
	input_control {
	  process_mode: PROCESS_MODE_FULL_FRAME
	  interval: 0
	}


4. Inspect customparser
-----------------------

See :file:`customparserbbox_centernet.cpp` in */opt/nvidia/deepstream/deepstream-5.1/samples/deepstream_triton_model_deploy/centerface/customparser* for SSD parser details.


5. Inspect model repo
---------------------

The model repo :file:`centerface` in */opt/nvidia/deepstream/deepstream-5.1/samples/deepstream_triton_model_deploy/centerface/centerface* follows the structure of deployment of ONNX model in Triton inference server.
::

	.
	|-- 1
	|   |-- change_dim.py
	|   |-- model.onnx
	|   `-- run.sh
	|-- centerface_labels.txt
	`-- config.pbtxt

For more details, see :ref:`model_repo_structure`.

6. Inspect model config
-----------------------

The model config :file:`config.pbtxt` is in the model repo.

::

	name: "centerface"
	platform: "onnxruntime_onnx"
	max_batch_size: 0
	input [
	  {
	    name: "input.1"
	    data_type: TYPE_FP32
	#    format: FORMAT_NCHW
	    dims: [ -1, 3, 480, 640]
	#    reshape { shape: [ 1, 3, 480, 640 ] }
	  }
	]

	output [
	  {
	    name: "537"
	    data_type: TYPE_FP32
	    dims: [ -1, 1, -1, -1 ]
	   # reshape { shape: [ 1, 1, 1, 1 ] }
	    label_filename: "centerface_labels.txt"
	  },
	  {
	    name: "538"
	    data_type: TYPE_FP32
	    dims: [ -1, 2, -1, -1]
	    label_filename: "centerface_labels.txt"
	  },

	  {
	    name: "539"
	    data_type: TYPE_FP32
	    dims: [-1,  2, -1, -1]
	    label_filename: "centerface_labels.txt"
	  },
	  {
	    name: "540"
	    data_type: TYPE_FP32
	    dims: [-1, 10 , -1, -1]
	    label_filename: "centerface_labels.txt"
	  }
	]

	instance_group {
	  count: 1
	  gpus: 0
	  kind: KIND_GPU
	}

	# Enable TensorRT acceleration running in gpu instance. It might take several
	# minutes during intialization to generate tensorrt online caches.

	#optimization { execution_accelerators {
	 # gpu_execution_accelerator : [ { name : "tensorrt" } ]
	#		}}


For more details, see :ref:`model_repo_structure`

7. Run DeepStream
-----------------

::

	deepstream-app -c /opt/nvidia/deepstream/deepstream-5.1/samples/deepstream_triton_model_deploy/centerface/config/source1_primary_detector.txt

Note that the default :file:`[sink0]` is disable and fakesink. Enable the sink and set the :code:`type` property to 2 to see DeepStream output in GUI.