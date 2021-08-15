.. _custom_parser_def:

Custom Model - Custom Parser
============================

To deploy custom models in DeepStream, it is a must to write custom library which can parse bounding box coordinates and the object class from the output layers. This section describes how the custom parser is implemented.

Definition
----------

The definition of custom parser implementation :file:`nvdsinfer_custom_impl.h` is in */opt/nvidia/deepstream/deepstream-5.1/sources/includes*. You must include this header file when making your library.

.. code-block:: cpp

	#ifndef _NVDSINFER_CUSTOM_IMPL_H_
	#define _NVDSINFER_CUSTOM_IMPL_H_

	#include <string>
	#include <vector>

	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
	#include "NvCaffeParser.h"
	#include "NvUffParser.h"
	#pragma GCC diagnostic pop

	#include "nvdsinfer.h"

	/*
	 * C++ interfaces
	 */
	#ifdef __cplusplus

	/**
	 * A model parser interface to translate user-defined model to a TensorRT network.
	 *
	 * Users can parse any custom model derived from this inferface. Instance would
	 * be created by a call to @fn NvDsInferCreateModelParser.
	 *
	 * Implementations should make sure that all member functions are overriden.
	 * This parser will be deleted after the engine (nvinfer1::ICudaEngine) is built.
	 */
	class IModelParser
	{
	public:
	    IModelParser() = default;
	    /**
	     * Destructor, make sure all external resource would be released here. */
	    virtual ~IModelParser() = default;

	    /**
	     * Function interface for parsing custom model and building tensorrt
	     * network.
	     *
	     * @param[in, out] network NvDsInfer will create the @a network and
	     *                 implementation can setup this network layer by layer.
	     * @return NvDsInferStatus indicating if model parsing was sucessful.
	     */
	    virtual NvDsInferStatus parseModel(
	        nvinfer1::INetworkDefinition& network) = 0;

	    /**
	     * Function interface to check if parser can support full-dimensions.
	     */
	    virtual bool hasFullDimsSupported() const = 0;

	    /**
	     * Function interface to get the new model name which is to be used for
	     * constructing the serialized engine file path.
	     */
	    virtual const char* getModelName() const = 0;
	};
	#endif

	/*
	 * C interfaces
	 */

	#ifdef __cplusplus
	extern "C"
	{
	#endif

	/**
	 * Holds the detection parameters required for parsing objects.
	 */
	typedef struct
	{
	  /** Holds the number of classes requested to be parsed, starting with
	   class ID 0. Parsing functions may only output objects with
	   class ID less than this value. */
	  unsigned int numClassesConfigured;
	  /** Holds a per-class vector of detection confidence thresholds
	   to be applied prior to clustering operation.
	   Parsing functions may only output an object with detection confidence
	   greater than or equal to the vector element indexed by the object's
	   class ID. */
	  std::vector<float> perClassPreclusterThreshold;
	  /* Per class threshold to be applied post clustering operation */
	  std::vector<float> perClassPostclusterThreshold;

	  /** Deprecated. Use perClassPreclusterThreshold instead. Reference to
	   * maintain backward compatibility. */
	  std::vector<float> &perClassThreshold = perClassPreclusterThreshold;
	} NvDsInferParseDetectionParams;

	/**
	 * Type definition for the custom bounding box parsing function.
	 *
	 * @param[in]  outputLayersInfo A vector containing information on the output
	 *                              layers of the model.
	 * @param[in]  networkInfo      Network information.
	 * @param[in]  detectionParams  Detection parameters required for parsing
	 *                              objects.
	 * @param[out] objectList       A reference to a vector in which the function
	 *                              is to add parsed objects.
	 */
	typedef bool (* NvDsInferParseCustomFunc) (
	        std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
	        NvDsInferNetworkInfo  const &networkInfo,
	        NvDsInferParseDetectionParams const &detectionParams,
	        std::vector<NvDsInferObjectDetectionInfo> &objectList);

	/**
	 * Validates a custom parser function definition. Must be called
	 * after defining the function.
	 */
	#define CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(customParseFunc) \
	    static void checkFunc_ ## customParseFunc (NvDsInferParseCustomFunc func = customParseFunc) \
	        { checkFunc_ ## customParseFunc (); }; \
	    extern "C" bool customParseFunc (std::vector<NvDsInferLayerInfo> const &outputLayersInfo, \
	           NvDsInferNetworkInfo  const &networkInfo, \
	           NvDsInferParseDetectionParams const &detectionParams, \
	           std::vector<NvDsInferObjectDetectionInfo> &objectList);

	/**
	 * Type definition for the custom bounding box and instance mask parsing function.
	 *
	 * @param[in]  outputLayersInfo A vector containing information on the output
	 *                              layers of the model.
	 * @param[in]  networkInfo      Network information.
	 * @param[in]  detectionParams  Detection parameters required for parsing
	 *                              objects.
	 * @param[out] objectList       A reference to a vector in which the function
	 *                              is to add parsed objects and instance mask.
	 */
	typedef bool (* NvDsInferInstanceMaskParseCustomFunc) (
	        std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
	        NvDsInferNetworkInfo  const &networkInfo,
	        NvDsInferParseDetectionParams const &detectionParams,
	        std::vector<NvDsInferInstanceMaskInfo> &objectList);

	/**
	 * Validates a custom parser function definition. Must be called
	 * after defining the function.
	 */
	#define CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(customParseFunc) \
	    static void checkFunc_ ## customParseFunc (NvDsInferInstanceMaskParseCustomFunc func = customParseFunc) \
	        { checkFunc_ ## customParseFunc (); }; \
	    extern "C" bool customParseFunc (std::vector<NvDsInferLayerInfo> const &outputLayersInfo, \
	           NvDsInferNetworkInfo  const &networkInfo, \
	           NvDsInferParseDetectionParams const &detectionParams, \
	           std::vector<NvDsInferInstanceMaskInfo> &objectList);

	/**
	 * Type definition for the custom classifier output parsing function.
	 *
	 * @param[in]  outputLayersInfo  A vector containing information on the
	 *                               output layers of the model.
	 * @param[in]  networkInfo       Network information.
	 * @param[in]  classifierThreshold
	                                 Classification confidence threshold.
	 * @param[out] attrList          A reference to a vector in which the function
	 *                               is to add the parsed attributes.
	 * @param[out] descString        A reference to a string object in which the
	 *                               function may place a description string.
	 */
	typedef bool (* NvDsInferClassiferParseCustomFunc) (
	        std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
	        NvDsInferNetworkInfo  const &networkInfo,
	        float classifierThreshold,
	        std::vector<NvDsInferAttribute> &attrList,
	        std::string &descString);

	/**
	 * Validates the classifier custom parser function definition. Must be called
	 * after defining the function.
	 */
	#define CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(customParseFunc) \
	    static void checkFunc_ ## customParseFunc (NvDsInferClassiferParseCustomFunc func = customParseFunc) \
	        { checkFunc_ ## customParseFunc (); }; \
	    extern "C" bool customParseFunc (std::vector<NvDsInferLayerInfo> const &outputLayersInfo, \
	           NvDsInferNetworkInfo  const &networkInfo, \
	           float classifierThreshold, \
	           std::vector<NvDsInferAttribute> &attrList, \
	           std::string &descString);

	typedef struct _NvDsInferContextInitParams NvDsInferContextInitParams;

	/**
	 * Type definition for functions that build and return a @c CudaEngine for
	 * custom models.
	 *
	 * @deprecated  The NvDsInferCudaEngineGet interface is replaced by
	 * NvDsInferEngineCreateCustomFunc().
	 *
	 * The implementation of this interface must build the
	 * nvinfer1::ICudaEngine instance using nvinfer1::IBuilder instance
	 * @a builder. The builder instance is managed by the caller;
	 * the implementation must not destroy it.
	 *
	 * Properties like @a MaxBatchSize, @a MaxWorkspaceSize, INT8/FP16
	 * precision parameters, and DLA parameters (if applicable) are set on the
	 * builder before it is passed to the interface. The corresponding Get
	 * functions of the nvinfer1::IBuilder interface can be used to get
	 * the property values.
	 *
	 * The implementation must make sure not to reduce the @a MaxBatchSize of the
	 * returned @c CudaEngine.
	 *
	 * @param[in]  builder      An nvinfer1::IBuilder instance.
	 * @param[in]  initParams   A pointer to the structure to be used for
	 *                          initializing the NvDsInferContext instance.
	 * @param[in]  dataType     Data precision.
	 * @param[out] cudaEngine   A pointer to a location where the function is to
	 *                          store a reference to the nvinfer1::ICudaEngine
	 *                          instance it has built.
	 * @return  True if the engine build was successful, or false otherwise. TBD Shaunak asked to have the original "deprecated" description restored. That would be redundant; there's a @deprecated command near the top of the comment.
	 */
	typedef bool (* NvDsInferEngineCreateCustomFunc) (
	        nvinfer1::IBuilder * const builder,
	        const NvDsInferContextInitParams * const initParams,
	        nvinfer1::DataType dataType,
	        nvinfer1::ICudaEngine *& cudaEngine);

	/**
	 * A macro that validates a custom engine creator function definition.
	 * Call this macro after the function is defined.
	*/
	#define CHECK_CUSTOM_ENGINE_CREATE_FUNC_PROTOTYPE(customEngineCreateFunc) \
	    static void checkFunc_ ## customEngineCreateFunc (NvDsInferEngineCreateCustomFunc = customEngineCreateFunc) \
	        { checkFunc_ ## customEngineCreateFunc(); }; \
	    extern "C" bool customEngineCreateFunc (  \
	        nvinfer1::IBuilder * const builder,  \
	        const NvDsInferContextInitParams const *initParams, \
	        nvinfer1::DataType dataType, \
	        nvinfer1::ICudaEngine *& cudaEngine);

	/**
	 * Specifies the type of the Plugin Factory.
	 */
	typedef enum
	{
	  /** Specifies nvcaffeparser1::IPluginFactory or
	   nvuffparser::IPluginFactory. */
	  PLUGIN_FACTORY,
	  /** Specifies nvcaffeparser1::IPluginFactoryExt or
	   nvuffparser::IPluginFactoryExt. */
	  PLUGIN_FACTORY_EXT,
	  /** Specifies nvcaffeparser1::IPluginFactoryV2. Used only for Caffe models. */
	  PLUGIN_FACTORY_V2
	} NvDsInferPluginFactoryType;

	/**
	 * Holds a pointer to a heap-allocated Plugin Factory object required during
	 * Caffe model parsing.
	 */
	typedef union
	{
	  nvcaffeparser1::IPluginFactory *pluginFactory;
	  nvcaffeparser1::IPluginFactoryExt *pluginFactoryExt;
	  nvcaffeparser1::IPluginFactoryV2 *pluginFactoryV2;
	} NvDsInferPluginFactoryCaffe;

	/**
	 * Holds a  pointer to a heap-allocated Plugin Factory object required during
	 * UFF model parsing.
	 */
	typedef union
	{
	  nvuffparser::IPluginFactory *pluginFactory;
	  nvuffparser::IPluginFactoryExt *pluginFactoryExt;
	} NvDsInferPluginFactoryUff;

	/**
	 * Gets a new instance of a Plugin Factory interface to be used
	 * during parsing of Caffe models. The function must set the correct @a type and
	 * the correct field in the @a pluginFactory union, based on the type of the
	 * Plugin Factory, (i.e. one of @a pluginFactory, @a pluginFactoryExt, or
	 * @a pluginFactoryV2).
	 *
	 * @param[out] pluginFactory    A reference to the union that contains
	 *                              a pointer to the Plugin Factory object.
	 * @param[out] type             Specifies the type of @a pluginFactory, i.e.
	 *                              which member the @a pluginFactory union
	 *                              is valid.
	 * @return  True if the Plugin Factory was created successfully, or false
	 *  otherwise.
	 */
	bool NvDsInferPluginFactoryCaffeGet (NvDsInferPluginFactoryCaffe &pluginFactory,
	    NvDsInferPluginFactoryType &type);

	/**
	 * Destroys a Plugin Factory instance created by
	 * NvDsInferPluginFactoryCaffeGet().
	 *
	 * @param[in] pluginFactory A reference to the union that contains a
	 *                          pointer to the Plugin Factory instance returned
	 *                          by NvDsInferPluginFactoryCaffeGet().
	 */
	void NvDsInferPluginFactoryCaffeDestroy (NvDsInferPluginFactoryCaffe &pluginFactory);

	/**
	 * Returns a new instance of a Plugin Factory interface to be used
	 * during parsing of UFF models. The function must set the correct @a type and
	 * the correct field in the @a pluginFactory union, based on the type of the
	 * Plugin Factory (i.e. @a pluginFactory or @a pluginFactoryExt).
	 *
	 * @param[out] pluginFactory    A reference to a union that contains a pointer
	 *                              to the Plugin Factory object.
	 * @param[out] type             Specifies the type of @a pluginFactory, i.e.
	 *                              which member of the @a pluginFactory union
	 *                              is valid.
	 * @return  True if the Plugin Factory was created successfully, or false
	 *  otherwise.
	 */
	bool NvDsInferPluginFactoryUffGet (NvDsInferPluginFactoryUff &pluginFactory,
	    NvDsInferPluginFactoryType &type);

	/**
	 * Destroys a Plugin Factory instance created by NvDsInferPluginFactoryUffGet().
	 *
	 * @param[in] pluginFactory     A reference to the union that contains a
	 *                              pointer to the Plugin Factory instance returned
	 *                              by NvDsInferPluginFactoryUffGet().
	 */
	void NvDsInferPluginFactoryUffDestroy (NvDsInferPluginFactoryUff &pluginFactory);

	/**
	 * Returns a new instance of a Plugin Factory interface to be used
	 * during parsing deserialization of CUDA engines.
	 *
	 * @param[out] pluginFactory    A reference to nvinfer1::IPluginFactory*
	 *                              in which the function is to place a pointer to
	 *                              the instance.
	 * @return  True if the Plugin Factory was created successfully, or false
	 *  otherwise.
	 */
	bool NvDsInferPluginFactoryRuntimeGet (nvinfer1::IPluginFactory *& pluginFactory);

	/**
	 * Destroys a Plugin Factory instance created by
	 * NvDsInferPluginFactoryRuntimeGet().
	 *
	 * @param[in] pluginFactory     A pointer to the Plugin Factory instance
	 *                              returned by NvDsInferPluginFactoryRuntimeGet().
	 */
	void NvDsInferPluginFactoryRuntimeDestroy (nvinfer1::IPluginFactory * pluginFactory);

	/**
	 * Initializes the input layers for inference. This function is called only once
	 * during before the first inference call.
	 *
	 * @param[in] inputLayersInfo   A reference to a vector containing information
	 *                              on the input layers of the model. This does not
	 *                              contain the NvDsInferLayerInfo structure for
	 *                              the layer for video frame input.
	 * @param[in] networkInfo       A reference to anetwork information structure.
	 * @param[in] maxBatchSize      The maximum batch size for inference.
	 *                              The input layer buffers are allocated
	 *                              for this batch size.
	 * @return  True if input layers are initialized successfully, or false
	 *  otherwise.
	 */
	bool NvDsInferInitializeInputLayers (std::vector<NvDsInferLayerInfo> const &inputLayersInfo,
	        NvDsInferNetworkInfo const &networkInfo,
	        unsigned int maxBatchSize);
	/**
	 * The NvDsInferCudaEngineGet interface has been deprecated and has been
	 * replaced by NvDsInferEngineCreateCustomFunc function.
	 */
	bool NvDsInferCudaEngineGet(nvinfer1::IBuilder *builder,
	        NvDsInferContextInitParams *initParams,
	        nvinfer1::DataType dataType,
	        nvinfer1::ICudaEngine *& cudaEngine)
	        __attribute__((deprecated("Use 'engine-create-func-name' config parameter instead")));

	/**
	 * Create a customized neural network parser for user-defined models.
	 *
	 * User need to implement a new IModelParser class with @a initParams
	 * referring to any model path and/or customNetworkConfigFilePath.
	 *
	 * @param[in] initParams with model paths or config files.
	 * @return Instance of IModelParser implementation.
	 */
	IModelParser* NvDsInferCreateModelParser(
	    const NvDsInferContextInitParams* initParams);

	#ifdef __cplusplus
	}
	#endif

	#endif

the file for also references from :code:`nvdsinfer.h` in the same directory

.. code-block:: cpp

	/*
	 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
	 *
	 * NVIDIA Corporation and its licensors retain all intellectual property
	 * and proprietary rights in and to this software, related documentation
	 * and any modifications thereto.  Any use, reproduction, disclosure or
	 * distribution of this software and related documentation without an express
	 * license agreement from NVIDIA Corporation is strictly prohibited.
	 *
	 */

	/**
	 * @file
	 * <b>NVIDIA DeepStream inference specifications </b>
	 *
	 * @b Description: This file defines common elements used in the API
	 * exposed by the Gst-nvinfer plugin.
	 */

	/**
	 * @defgroup  ee_nvinf  Gst-infer API Common Elements
	 *
	 * Defines common elements used in the API exposed by the Gst-inference plugin.
	 * @ingroup NvDsInferApi
	 * @{
	 */

	#ifndef _NVDSINFER_H_
	#define _NVDSINFER_H_

	#include <stdint.h>

	#ifdef __cplusplus
	extern "C"
	{
	#endif

	#define NVDSINFER_MAX_DIMS 8

	#define _DS_DEPRECATED_(STR) __attribute__ ((deprecated (STR)))

	/**
	 * Holds the dimensions of a layer.
	 */
	typedef struct
	{
	  /** Holds the number of dimesions in the layer.*/
	  unsigned int numDims;
	  /** Holds the size of the layer in each dimension. */
	  unsigned int d[NVDSINFER_MAX_DIMS];
	  /** Holds the number of elements in the layer, including all dimensions.*/
	  unsigned int numElements;
	} NvDsInferDims;

	/**
	 * Holds the dimensions of a three-dimensional layer.
	 */
	typedef struct
	{
	  /** Holds the channel count of the layer.*/
	  unsigned int c;
	  /** Holds the height of the layer.*/
	  unsigned int h;
	  /** Holds the width of the layer.*/
	  unsigned int w;
	} NvDsInferDimsCHW;

	/**
	 * Specifies the data type of a layer.
	 */
	typedef enum
	{
	  /** Specifies FP32 format. */
	  FLOAT = 0,
	  /** Specifies FP16 format. */
	  HALF = 1,
	  /** Specifies INT8 format. */
	  INT8 = 2,
	  /** Specifies INT32 format. */
	  INT32 = 3
	} NvDsInferDataType;

	/**
	 * Holds information about one layer in the model.
	 */
	typedef struct
	{
	  /** Holds the data type of the layer. */
	  NvDsInferDataType dataType;
	  /** Holds the dimensions of the layer. */
	  union {
	      NvDsInferDims inferDims;
	      NvDsInferDims dims _DS_DEPRECATED_("dims is deprecated. Use inferDims instead");
	  };
	  /** Holds the TensorRT binding index of the layer. */
	  int bindingIndex;
	  /** Holds the name of the layer. */
	  const char* layerName;
	  /** Holds a pointer to the buffer for the layer data. */
	  void *buffer;
	  /** Holds a Boolean; true if the layer is an input layer,
	   or false if an output layer. */
	  int isInput;
	} NvDsInferLayerInfo;

	/**
	 * Holds information about the model network.
	 */
	typedef struct
	{
	  /** Holds the input width for the model. */
	  unsigned int width;
	  /** Holds the input height for the model. */
	  unsigned int height;
	  /** Holds the number of input channels for the model. */
	  unsigned int channels;
	} NvDsInferNetworkInfo;

	/**
	 * Sets values on a @ref NvDsInferDimsCHW structure from a @ref NvDsInferDims
	 * structure.
	 */
	#define getDimsCHWFromDims(dimsCHW,dims) \
	  do { \
	    (dimsCHW).c = (dims).d[0]; \
	    (dimsCHW).h = (dims).d[1]; \
	    (dimsCHW).w = (dims).d[2]; \
	  } while (0)

	#define getDimsHWCFromDims(dimsCHW,dims) \
	  do { \
	    (dimsCHW).h = (dims).d[0]; \
	    (dimsCHW).w = (dims).d[1]; \
	    (dimsCHW).c = (dims).d[2]; \
	  } while (0)

	/**
	 * Holds information about one parsed object from a detector's output.
	 */
	typedef struct
	{
	  /** Holds the ID of the class to which the object belongs. */
	  unsigned int classId;

	  /** Holds the horizontal offset of the bounding box shape for the object. */
	  float left;
	  /** Holds the vertical offset of the object's bounding box. */
	  float top;
	  /** Holds the width of the object's bounding box. */
	  float width;
	  /** Holds the height of the object's bounding box. */
	  float height;

	  /** Holds the object detection confidence level; must in the range
	   [0.0,1.0]. */
	  float detectionConfidence;
	} NvDsInferObjectDetectionInfo;

	/**
	 * A typedef defined to maintain backward compatibility.
	 */
	typedef NvDsInferObjectDetectionInfo NvDsInferParseObjectInfo;

	/**
	 * Holds information about one parsed object and instance mask from a detector's output.
	 */
	typedef struct
	{
	  /** Holds the ID of the class to which the object belongs. */
	  unsigned int classId;

	  /** Holds the horizontal offset of the bounding box shape for the object. */
	  float left;
	  /** Holds the vertical offset of the object's bounding box. */
	  float top;
	  /** Holds the width of the object's bounding box. */
	  float width;
	  /** Holds the height of the object's bounding box. */
	  float height;

	  /** Holds the object detection confidence level; must in the range
	   [0.0,1.0]. */
	  float detectionConfidence;

	  /** Holds object segment mask */
	  float *mask;
	  /** Holds width of mask */
	  unsigned int mask_width;
	  /** Holds height of mask */
	  unsigned int mask_height;
	  /** Holds size of mask in bytes*/
	  unsigned int mask_size;
	} NvDsInferInstanceMaskInfo;

	/**
	 * Holds information about one classified attribute.
	 */
	typedef struct
	{
	    /** Holds the index of the attribute's label. This index corresponds to
	     the order of output layers specified in the @a outputCoverageLayerNames
	     vector during initialization. */
	    unsigned int attributeIndex;
	    /** Holds the the attribute's output value. */
	    unsigned int attributeValue;
	    /** Holds the attribute's confidence level. */
	    float attributeConfidence;
	    /** Holds a pointer to a string containing the attribute's label.
	     Memory for the string must not be freed. Custom parsing functions must
	     allocate strings on heap using strdup or equivalent. */
	    char *attributeLabel;
	} NvDsInferAttribute;

	/**
	 * Enum for the status codes returned by NvDsInferContext.
	 */
	typedef enum {
	    /** NvDsInferContext operation succeeded. */
	    NVDSINFER_SUCCESS = 0,
	    /** Failed to configure the NvDsInferContext instance possibly due to an
	     *  erroneous initialization property. */
	    NVDSINFER_CONFIG_FAILED,
	    /** Custom Library interface implementation failed. */
	    NVDSINFER_CUSTOM_LIB_FAILED,
	    /** Invalid parameters were supplied. */
	    NVDSINFER_INVALID_PARAMS,
	    /** Output parsing failed. */
	    NVDSINFER_OUTPUT_PARSING_FAILED,
	    /** CUDA error was encountered. */
	    NVDSINFER_CUDA_ERROR,
	    /** TensorRT interface failed. */
	    NVDSINFER_TENSORRT_ERROR,
	    /** Resource error was encountered. */
	    NVDSINFER_RESOURCE_ERROR,
	    /** TRT-IS error was encountered. */
	    NVDSINFER_TRTIS_ERROR,
	    /** Unknown error was encountered. */
	    NVDSINFER_UNKNOWN_ERROR
	} NvDsInferStatus;

	/**
	 * Enum for the log levels of NvDsInferContext.
	 */
	typedef enum {
	    NVDSINFER_LOG_ERROR = 0,
	    NVDSINFER_LOG_WARNING,
	    NVDSINFER_LOG_INFO,
	    NVDSINFER_LOG_DEBUG,
	} NvDsInferLogLevel;

	/**
	 * Get the string name for the status.
	 *
	 * @param[in] status An NvDsInferStatus value.
	 * @return String name for the status. Memory is owned by the function. Callers
	 *         should not free the pointer.
	 */
	const char* NvDsInferStatus2Str(NvDsInferStatus status);

	#ifdef __cplusplus
	}
	#endif

	/* C++ data types */
	#ifdef __cplusplus

	/**
	 * Enum for selecting between minimum/optimal/maximum dimensions of a layer
	 * in case of dynamic shape network.
	 */
	typedef enum
	{
	    kSELECTOR_MIN = 0,
	    kSELECTOR_OPT,
	    kSELECTOR_MAX,
	    kSELECTOR_SIZE
	} NvDsInferProfileSelector;

	/**
	 * Holds full dimensions (including batch size) for a layer.
	 */
	typedef struct
	{
	    int batchSize = 0;
	    NvDsInferDims dims = {0};
	} NvDsInferBatchDims;

	/**
	 * Extended structure for bound layer information which additionally includes
	 * min/optimal/max full dimensions of a layer in case of dynamic shape.
	 */
	struct NvDsInferBatchDimsLayerInfo : NvDsInferLayerInfo
	{
	    NvDsInferBatchDims profileDims[kSELECTOR_SIZE];
	};

	#endif

	#endif

	/** @} */
