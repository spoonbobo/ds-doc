Custom Model - Sample Custom Parser - SSD
=========================================

For the source code of SSD, see `TensorRT sample - SSD <https://github.com/NVIDIA/TensorRT/blob/master/samples/sampleSSD/sampleSSD.cpp>`_

Custom parser implementation
----------------------------

the parser file :file:`nvdsparsebbox_ssd.cpp` is in */opt/nvidia/deepstream/deepstream-5.1/sources/objectDetector_SSD/nvdsinfer_custom_impl_ssd*.

.. code-block:: cpp

	#include <cstring>
	#include <iostream>
	#include "nvdsinfer_custom_impl.h"

	#define MIN(a,b) ((a) < (b) ? (a) : (b))
	#define MAX(a,b) ((a) > (b) ? (a) : (b))
	#define CLIP(a,min,max) (MAX(MIN(a, max), min))

	/* This is a sample bounding box parsing function for the sample SSD UFF
	 * detector model provided with the TensorRT samples. */

	extern "C"
	bool NvDsInferParseCustomSSD (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
	        NvDsInferNetworkInfo  const &networkInfo,
	        NvDsInferParseDetectionParams const &detectionParams,
	        std::vector<NvDsInferObjectDetectionInfo> &objectList);

	/* C-linkage to prevent name-mangling */
	extern "C"
	bool NvDsInferParseCustomSSD (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
	        NvDsInferNetworkInfo  const &networkInfo,
	        NvDsInferParseDetectionParams const &detectionParams,
	        std::vector<NvDsInferObjectDetectionInfo> &objectList)
	{
	  static int nmsLayerIndex = -1;
	  static int nms1LayerIndex = -1;
	  static bool classMismatchWarn = false;
	  int numClassesToParse;
	  static const int NUM_CLASSES_SSD = 91;

	  if (nmsLayerIndex == -1) {
	    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
	      if (strcmp(outputLayersInfo[i].layerName, "NMS") == 0) {
	        nmsLayerIndex = i;
	        break;
	      }
	    }
	    if (nmsLayerIndex == -1) {
	    std::cerr << "Could not find NMS layer buffer while parsing" << std::endl;
	    return false;
	    }
	  }

	  if (nms1LayerIndex == -1) {
	    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
	      if (strcmp(outputLayersInfo[i].layerName, "NMS_1") == 0) {
	        nms1LayerIndex = i;
	        break;
	      }
	    }
	    if (nms1LayerIndex == -1) {
	    std::cerr << "Could not find NMS_1 layer buffer while parsing" << std::endl;
	    return false;
	    }
	  }

	  if (!classMismatchWarn) {
	    if (NUM_CLASSES_SSD !=
	        detectionParams.numClassesConfigured) {
	      std::cerr << "WARNING: Num classes mismatch. Configured:" <<
	        detectionParams.numClassesConfigured << ", detected by network: " <<
	        NUM_CLASSES_SSD << std::endl;
	    }
	    classMismatchWarn = true;
	  }

	  numClassesToParse = MIN (NUM_CLASSES_SSD,
	      detectionParams.numClassesConfigured);

	  int keepCount = *((int *) outputLayersInfo[nms1LayerIndex].buffer);
	  float *detectionOut = (float *) outputLayersInfo[nmsLayerIndex].buffer;

	  for (int i = 0; i < keepCount; ++i)
	  {
	    float* det = detectionOut + i * 7;
	    int classId = det[1];

	    if (classId >= numClassesToParse)
	      continue;

	    float threshold = detectionParams.perClassPreclusterThreshold[classId];

	    if (det[2] < threshold)
	      continue;

	    unsigned int rectx1, recty1, rectx2, recty2;
	    NvDsInferObjectDetectionInfo object;

	    rectx1 = det[3] * networkInfo.width;
	    recty1 = det[4] * networkInfo.height;
	    rectx2 = det[5] * networkInfo.width;
	    recty2 = det[6] * networkInfo.height;

	    object.classId = classId;
	    object.detectionConfidence = det[2];

	    /* Clip object box co-ordinates to network resolution */
	    object.left = CLIP(rectx1, 0, networkInfo.width - 1);
	    object.top = CLIP(recty1, 0, networkInfo.height - 1);
	    object.width = CLIP(rectx2, 0, networkInfo.width - 1) -
	      object.left + 1;
	    object.height = CLIP(recty2, 0, networkInfo.height - 1) -
	      object.top + 1;

	    objectList.push_back(object);
	  }

	  return true;
	}

	/* Check that the custom function has been defined correctly */
	CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomSSD);
