.. _deepstream_infer_config:

DeepStream - Inference Configurations
=====================================

See `Gst-nvinfer <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html>`_ for all configurable properties in inference config file.


net-scale-factor
----------------
For most images, they have pixel values ranged [0, 255]. The :code:`net-scale-factor` is applied to convert pixel value to a scale using the formula::

	y = net-scale-factor * (x - mean)

For example, if we set the factor value as 0.003921 and null for mean or offsets, the pixel values will be normalised to a range [0, 1].


offsets
-------
:code:`offsets` is an array of mean values of color components to be subtracted from each pixel for pixel normalization. For example, a pixel have three color components *r*, *g*, *b*. With :code:`net-scale-factor` set to 1, and offsets set to *x;y;z*, the normalised pixel values become *r-x*, *g-y* and *b-z*.


scaling filter
--------------
:code:`scaling-filter` defines the interpolation method for image scaling. Value for this filter can be found at :dfn:`<deepstream-root>/sources/includes/nvbufsurftransform.h`.

::

	/**
	 * Specifies video interpolation methods.
	 */
	typedef enum
	{
	  /** Specifies Nearest Interpolation Method interpolation. */
	  NvBufSurfTransformInter_Nearest = 0,
	  /** Specifies Bilinear Interpolation Method interpolation. */
	  NvBufSurfTransformInter_Bilinear,
	  /** Specifies GPU-Cubic, VIC-5 Tap interpolation. */
	  NvBufSurfTransformInter_Algo1,
	  /** Specifies GPU-Super, VIC-10 Tap interpolation. */
	  NvBufSurfTransformInter_Algo2,
	  /** Specifies GPU-Lanzos, VIC-Smart interpolation. */
	  NvBufSurfTransformInter_Algo3,
	  /** Specifies GPU-Ignored, VIC-Nicest interpolation. */
	  NvBufSurfTransformInter_Algo4,
	  /** Specifies GPU-Nearest, VIC-Nearest interpolation. */
	  NvBufSurfTransformInter_Default
	} NvBufSurfTransform_Inter;