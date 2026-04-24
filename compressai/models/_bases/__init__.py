"""Abstract base classes shared by multiple slice-based LIC models.

These were historically hidden behind ``stf_support`` / ``dcae_support`` file
names which obscured the fact that they're real abstract :class:`CompressionModel`
subclasses inherited by 3-4 models each.
"""
from .dictionary_entropy import (
    DictionaryEntropyCompressionModel,
    infer_attention_head_dim,
    infer_dictionary_max_support_slices,
    infer_dictionary_num_slices,
    infer_stage_block_num,
    infer_window_size,
)
from .slice_entropy import (
    SliceEntropyCompressionModel,
    infer_max_support_slices,
    infer_num_slices,
    lrp_support_channels,
    make_entropy_transform,
    slice_support_channels,
)

__all__ = [
    "DictionaryEntropyCompressionModel",
    "SliceEntropyCompressionModel",
    "infer_attention_head_dim",
    "infer_dictionary_max_support_slices",
    "infer_dictionary_num_slices",
    "infer_max_support_slices",
    "infer_num_slices",
    "infer_stage_block_num",
    "infer_window_size",
    "lrp_support_channels",
    "make_entropy_transform",
    "slice_support_channels",
]
