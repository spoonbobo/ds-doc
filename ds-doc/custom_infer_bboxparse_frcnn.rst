Custom Model - Sample Custom Parser - FasterRCNN
================================================

For the source code of FasterRCNN, see `TensorRT sample - FasterRCNN <https://github.com/NVIDIA/TensorRT/blob/master/samples/sampleFasterRCNN/sampleFasterRCNN.cpp>`_

Custom parser implementation
----------------------------

the parser file :file:`nvdsparsebbox_fasterRCNN.cpp` is in */opt/nvidia/deepstream/deepstream-5.1/sources/objectDetector_FasterRCNN/nvdsinfer_custom_impl_fasterRCNN*.

.. code-block:: cpp

	#include <cmath>
	#include <cstring>
	#include <iostream>
	#include "nvdsinfer_custom_impl.h"
	#include "nvdssample_fasterRCNN_common.h"

	#define MIN(a,b) ((a) < (b) ? (a) : (b))
	#define MAX(a,b) ((a) > (b) ? (a) : (b))
	#define CLIP(a,min,max) (MAX(MIN(a, max), min))

	/* This is a sample bounding box parsing function for the sample FasterRCNN
	 * detector model provided with the TensorRT samples. */

	extern "C"
	bool NvDsInferParseCustomFasterRCNN (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
	        NvDsInferNetworkInfo  const &networkInfo,
	        NvDsInferParseDetectionParams const &detectionParams,
	        std::vector<NvDsInferObjectDetectionInfo> &objectList);

	/* C-linkage to prevent name-mangling */
	extern "C"
	bool NvDsInferParseCustomFasterRCNN (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
	        NvDsInferNetworkInfo  const &networkInfo,
	        NvDsInferParseDetectionParams const &detectionParams,
	        std::vector<NvDsInferObjectDetectionInfo> &objectList)
	{
	  static int bboxPredLayerIndex = -1;
	  static int clsProbLayerIndex = -1;
	  static int roisLayerIndex = -1;
	  static const int NUM_CLASSES_FASTER_RCNN = 21;
	  static bool classMismatchWarn = false;
	  int numClassesToParse;

	  if (bboxPredLayerIndex == -1) {
	    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
	      if (strcmp(outputLayersInfo[i].layerName, "bbox_pred") == 0) {
	        bboxPredLayerIndex = i;
	        break;
	      }
	    }
	    if (bboxPredLayerIndex == -1) {
	    std::cerr << "Could not find bbox_pred layer buffer while parsing" << std::endl;
	    return false;
	    }
	  }

	  if (clsProbLayerIndex == -1) {
	    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
	      if (strcmp(outputLayersInfo[i].layerName, "cls_prob") == 0) {
	        clsProbLayerIndex = i;
	        break;
	      }
	    }
	    if (clsProbLayerIndex == -1) {
	    std::cerr << "Could not find cls_prob layer buffer while parsing" << std::endl;
	    return false;
	    }
	  }

	  if (roisLayerIndex == -1) {
	    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
	      if (strcmp(outputLayersInfo[i].layerName, "rois") == 0) {
	        roisLayerIndex = i;
	        break;
	      }
	    }
	    if (roisLayerIndex == -1) {
	    std::cerr << "Could not find rois layer buffer while parsing" << std::endl;
	    return false;
	    }
	  }

	  if (!classMismatchWarn) {
	    if (NUM_CLASSES_FASTER_RCNN !=
	        detectionParams.numClassesConfigured) {
	      std::cerr << "WARNING: Num classes mismatch. Configured:" <<
	        detectionParams.numClassesConfigured << ", detected by network: " <<
	        NUM_CLASSES_FASTER_RCNN << std::endl;
	    }
	    classMismatchWarn = true;
	  }

	  numClassesToParse = MIN (NUM_CLASSES_FASTER_RCNN,
	      detectionParams.numClassesConfigured);

	  float *rois = (float *) outputLayersInfo[roisLayerIndex].buffer;
	  float *deltas = (float *) outputLayersInfo[bboxPredLayerIndex].buffer;
	  float *scores = (float *) outputLayersInfo[clsProbLayerIndex].buffer;

	  for (int i = 0; i < nmsMaxOut; ++i)
	  {
	    float width = rois[i * 4 + 2] - rois[i * 4] + 1;
	    float height = rois[i * 4 + 3] - rois[i * 4 + 1] + 1;
	    float ctr_x = rois[i * 4] + 0.5f * width;
	    float ctr_y = rois[i * 4 + 1] + 0.5f * height;
	    float *deltas_offset = deltas + i * NUM_CLASSES_FASTER_RCNN * 4;
	    for (int j = 0; j < numClassesToParse; ++j)
	    {
	      float confidence = scores[i * NUM_CLASSES_FASTER_RCNN + j];
	      if (confidence < detectionParams.perClassPreclusterThreshold[j])
	        continue;
	      NvDsInferObjectDetectionInfo object;

	      float dx = deltas_offset[j * 4];
	      float dy = deltas_offset[j * 4 + 1];
	      float dw = deltas_offset[j * 4 + 2];
	      float dh = deltas_offset[j * 4 + 3];
	      float pred_ctr_x = dx * width + ctr_x;
	      float pred_ctr_y = dy * height + ctr_y;
	      float pred_w = exp(dw) * width;
	      float pred_h = exp(dh) * height;
	      float rectx1 = MIN (pred_ctr_x - 0.5f * pred_w, networkInfo.width - 1.f);
	      float recty1 = MIN (pred_ctr_y - 0.5f * pred_h, networkInfo.height - 1.f);
	      float rectx2 = MIN (pred_ctr_x + 0.5f * pred_w, networkInfo.width - 1.f);
	      float recty2 = MIN (pred_ctr_y + 0.5f * pred_h, networkInfo.height - 1.f);


	      object.classId = j;
	      object.detectionConfidence = confidence;

	      /* Clip object box co-ordinates to network resolution */
	      object.left = CLIP(rectx1, 0, networkInfo.width - 1);
	      object.top = CLIP(recty1, 0, networkInfo.height - 1);
	      object.width = CLIP(rectx2, 0, networkInfo.width - 1) - object.left + 1;
	      object.height = CLIP(recty2, 0, networkInfo.height - 1) - object.top + 1;

	      objectList.push_back(object);
	    }
	  }
	  return true;
	}

	/* Check that the custom function has been defined correctly */
	CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomFasterRCNN);

In the same directory with the parser file, we can find :file:`nvdssample_fasterRCNN_common.h`

.. code-block:: cpp

	#ifndef __NVDS_SAMPLE_FASTERRCNN_COMMON_H__
	#define __NVDS_SAMPLE_FASTERRCNN_COMMON_H__

	const int nmsMaxOut = 300;
	const int poolingH = 7;
	const int poolingW = 7;
	const int featureStride = 16;
	const int preNmsTop = 6000;
	const int anchorsRatioCount = 3;
	const int anchorsScaleCount = 3;
	const float iouThreshold = 0.7f;
	const float minBoxSize = 16;
	const float spatialScale = 0.0625f;
	const float anchorsRatios[anchorsRatioCount] = {0.5f, 1.0f, 2.0f};
	const float anchorsScales[anchorsScaleCount] = {8.0f, 16.0f, 32.0f};

	#endif
