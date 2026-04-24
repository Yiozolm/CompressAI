compressai.layers
=================

.. currentmodule:: compressai.layers

The :mod:`compressai.layers` package collects reusable building blocks for
learned image compression models. Helpers and lightweight modules sit at the
top level, while specialized blocks are grouped into sub-packages by topic:

.. list-table::
   :widths: 1 4
   :header-rows: 1

   * - Sub-package
     - Scope
   * - :mod:`compressai.layers.attn`
     - Window/Swin-style attention modules used by STF, WACNN, TCM, and
       related transformer-based codecs.
   * - :mod:`compressai.layers.graph`
     - Graph attention and feature-aggregation layers used by GLIC.
   * - :mod:`compressai.layers.ssm`
     - Selective state-space (Mamba/VMamba) layers and the underlying
       ``selective_scan`` kernels.
   * - :mod:`compressai.layers.wave`
     - Wavelet-domain transforms (DWT/IDWT) and ``WLS`` / ``iWLS`` analysis
       and synthesis blocks used by GLIC. Requires the optional
       ``pytorch_wavelets`` dependency.
   * - :mod:`compressai.layers.lic`
     - Miscellaneous LIC building blocks (gated FFNs, dictionary
       cross-attention, invertible coupling, MLIC analysis/synthesis, ...).


Convolution helpers
-------------------

.. autofunction:: conv
.. autofunction:: deconv
.. autofunction:: conv1x1
.. autofunction:: conv3x3
.. autofunction:: subpel_conv3x3


MaskedConv2d
------------
.. autoclass:: MaskedConv2d


CheckerboardMaskedConv2d
------------------------
.. autoclass:: CheckerboardMaskedConv2d


GDN
---
.. autoclass:: GDN


GDN1
----
.. autoclass:: GDN1


ResidualBlock
-------------
.. autoclass:: ResidualBlock


ResidualBlockWithStride
-----------------------
.. autoclass:: ResidualBlockWithStride


ResidualBlockUpsample
---------------------
.. autoclass:: ResidualBlockUpsample


AttentionBlock
--------------
.. autoclass:: AttentionBlock


QReLU
-----
.. autoclass:: QReLU


compressai.layers.attn
----------------------

.. currentmodule:: compressai.layers.attn

.. autoclass:: WMSA
.. autoclass:: SwinBlock
.. autoclass:: ConvTransBlock
.. autoclass:: SWAtten
.. autoclass:: WinNoShiftAttention
.. autoclass:: PatchMerging
.. autoclass:: PatchSplit
.. autoclass:: WindowAttention
.. autofunction:: window_partition
.. autofunction:: window_reverse
.. autofunction:: build_window_attention_mask
.. autofunction:: infer_swatten_window_size
.. autofunction:: infer_swatten_head_dim
.. autofunction:: infer_swatten_attention_dim


compressai.layers.graph
-----------------------

.. currentmodule:: compressai.layers.graph

.. autoclass:: GraphAggregator
.. autoclass:: GraphAttentionLayer
.. autoclass:: GraphDepthwiseFeedForward
.. autoclass:: GraphLayerStack
.. autoclass:: GFA
.. autoclass:: MGB
.. autoclass:: FeatureReshape
.. autoclass:: FeatureRestore


compressai.layers.ssm
---------------------

.. currentmodule:: compressai.layers.ssm

.. autoclass:: SS2D
.. autoclass:: VSSBlock
.. autofunction:: build_vss_backbone
.. autofunction:: build_vss_context_stage
.. autofunction:: infer_vss_depths
.. autofunction:: infer_vss_block_kwargs
.. autofunction:: selective_scan
.. autofunction:: selective_scan_ref
.. autofunction:: get_selective_scan_backend
.. autofunction:: is_mamba_ssm_available
.. autofunction:: is_selective_scan_cuda_available


compressai.layers.wave
----------------------

.. currentmodule:: compressai.layers.wave

.. autoclass:: DWT2D
.. autoclass:: IDWT2D
.. autoclass:: WLS
.. autoclass:: iWLS
.. autoclass:: WaveletResidualBlockWithStride
.. autoclass:: WaveletResidualBlockUpsample
.. autoclass:: WeConveneAnalysisTransform
.. autoclass:: WeConveneSynthesisTransform
.. autoclass:: WeConveneHyperAnalysisTransform
.. autoclass:: WeConveneHyperSynthesisTransform
.. autofunction:: is_pytorch_wavelets_available


compressai.layers.lic
---------------------

.. currentmodule:: compressai.layers.lic

.. autoclass:: OLP
.. autoclass:: LayerNorm2d
.. autoclass:: GatedFFN
.. autoclass:: GatedTransformCNN
.. autoclass:: DepthwiseConv5x5
.. autoclass:: ResidualBottleneckBlock
.. autoclass:: ResidualBottleneckBlockWithStride
.. autoclass:: ResidualBottleneckBlockWithUpsample
.. autoclass:: ConvolutionalGLU
.. autoclass:: MutiScaleDictionaryCrossAttentionGLU
.. autoclass:: SwinBlockWithConvMulti
.. autoclass:: SpatialAttentionBlock
.. autoclass:: SpatialAttentionLayer
.. autoclass:: AdaptiveFrequencyBlock
.. autoclass:: InverseAdaptiveFrequencyBlock
.. autoclass:: DenoisingAsRegularizer
.. autoclass:: CrossSparseWindowAttention
.. autoclass:: CouplingLayer
.. autoclass:: InvertibleConv1x1
.. autoclass:: SqueezeLayer
.. autoclass:: AnalysisTransform
.. autoclass:: SynthesisTransform
.. autoclass:: HyperAnalysis
.. autoclass:: HyperSynthesis
.. autoclass:: ChannelContext
.. autoclass:: LocalContext
.. autoclass:: LinearGlobalIntraContext
.. autoclass:: LinearGlobalInterContext
.. autoclass:: LatentResidualPrediction
.. autoclass:: EntropyParameters
.. autoclass:: CMICStage
.. autoclass:: CMICChannelContextBlock
.. autoclass:: CMICSpatialContextBlock
.. autoclass:: CMICAnalysisTransform
.. autoclass:: CMICSynthesisTransform
.. autoclass:: FATBlock
.. autoclass:: SwinFDWA
.. autoclass:: BranchWindowAttention
.. autoclass:: WindowFrequencyModulation
.. autoclass:: FTICAnalysisTransform
.. autoclass:: FTICSynthesisTransform
.. autoclass:: FTICHyperAnalysisTransform
.. autoclass:: FTICHyperSynthesisTransform
.. autoclass:: TCA
.. autoclass:: TCABlock
.. autoclass:: TCAEntropyModel
.. autoclass:: MaskedSliceChannelAttention
.. autoclass:: ConvPositionalEncoding
.. autofunction:: is_freia_available
