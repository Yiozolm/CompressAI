compressai.models
=================

.. currentmodule:: compressai.models


CompressionModel
----------------
.. autoclass:: CompressionModel
    :members:


SimpleVAECompressionModel
-------------------------
.. autoclass:: SimpleVAECompressionModel


Slice / Dictionary entropy base classes
---------------------------------------

Abstract :class:`CompressionModel` subclasses shared by candidate LIC models.
``SliceEntropyCompressionModel`` is inherited by :class:`WACNN`,
:class:`SymmetricalTransFormer`, and :class:`MambaVC`;
``DictionaryEntropyCompressionModel`` is inherited by :class:`DCAE` and
:class:`SAAF`. Subclasses populate ``g_a`` / ``g_s`` / ``h_a`` / ``h_*_s``
and call the matching ``_init_*_entropy`` method to wire up the per-slice
entropy modules and the ``z`` :class:`EntropyBottleneck`.

.. currentmodule:: compressai.models._bases

.. autoclass:: SliceEntropyCompressionModel
    :members: _init_slice_entropy
.. autoclass:: DictionaryEntropyCompressionModel
    :members: _init_dictionary_entropy
.. autofunction:: slice_support_channels
.. autofunction:: lrp_support_channels
.. autofunction:: make_entropy_transform
.. autofunction:: infer_num_slices
.. autofunction:: infer_max_support_slices
.. autofunction:: infer_dictionary_num_slices
.. autofunction:: infer_dictionary_max_support_slices
.. autofunction:: infer_stage_block_num
.. autofunction:: infer_attention_head_dim
.. autofunction:: infer_window_size

.. currentmodule:: compressai.models


FactorizedPrior
----------------
.. autoclass:: FactorizedPrior


ScaleHyperprior
---------------
.. autoclass:: ScaleHyperprior


MeanScaleHyperprior
-------------------
.. autoclass:: MeanScaleHyperprior


JointAutoregressiveHierarchicalPriors
-------------------------------------
.. autoclass:: JointAutoregressiveHierarchicalPriors


Cheng2020Anchor
---------------
.. autoclass:: Cheng2020Anchor


Cheng2020Attention
------------------
.. autoclass:: Cheng2020Attention


Cheng2020AnchorCheckerboard
---------------------------
.. autoclass:: Cheng2020AnchorCheckerboard


Elic2022Official
----------------
.. autoclass:: Elic2022Official


Elic2022Chandelier
------------------
.. autoclass:: Elic2022Chandelier


ScaleHyperpriorVbr
------------------
.. autoclass:: ScaleHyperpriorVbr


MeanScaleHyperpriorVbr
----------------------
.. autoclass:: MeanScaleHyperpriorVbr


JointAutoregressiveHierarchicalPriorsVbr
----------------------------------------
.. autoclass:: JointAutoregressiveHierarchicalPriorsVbr


.. currentmodule:: compressai.models.video

ScaleSpaceFlow
--------------
.. autoclass:: ScaleSpaceFlow


.. currentmodule:: compressai.models.pointcloud

DensityPreservingReconstructionPccModel
---------------------------------------
.. autoclass:: DensityPreservingReconstructionPccModel


PointNetReconstructionPccModel
------------------------------
.. autoclass:: PointNetReconstructionPccModel


PointNet2SsgReconstructionPccModel
----------------------------------
.. autoclass:: PointNet2SsgReconstructionPccModel

