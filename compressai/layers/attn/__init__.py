from .cam import CausalAttentionModule
from .contextformer import ContextFormerBlock, ContextFormerContextModel
from .cross import CrossAttention, CrossAttentionBlock
from .cross_window import WindowedCrossAttention
from .entroformer import (
    EntroformerAttention,
    EntroformerAttentionBlock,
    EntroformerBlock,
    EntroformerConfig,
    EntroformerFeedForward,
    EntroformerPreNorm,
    EntroformerUpPixelShuffle,
    TransDecoder,
    TransDecoder2,
    TransHyperScale,
    entroformer_clones,
)
from .inference import (
    infer_swatten_attention_dim,
    infer_swatten_head_dim,
    infer_swatten_window_size,
)
from .nsa import (
    NSABlock,
    ResViTBlock,
)
from .rstb import (
    PatchEmbed1D,
    PatchUnEmbed1D,
    RSTB,
    SwinTransformerBlock,
)
from .swin import (
    ConvTransBlock,
    PatchMerging,
    PatchSplit,
    SWAtten,
    SwinBlock,
    WMSA,
    WinNoShiftAttention,
)
from .swin_attention import (
    WindowAttention,
    build_window_attention_mask,
    pad_to_window_multiple,
    window_partition,
    window_reverse,
)

__all__ = [
    "CausalAttentionModule",
    "ConvTransBlock",
    "ContextFormerBlock",
    "ContextFormerContextModel",
    "CrossAttention",
    "CrossAttentionBlock",
    "EntroformerAttention",
    "EntroformerAttentionBlock",
    "EntroformerBlock",
    "EntroformerConfig",
    "EntroformerFeedForward",
    "EntroformerPreNorm",
    "EntroformerUpPixelShuffle",
    "NSABlock",
    "PatchEmbed1D",
    "PatchMerging",
    "PatchSplit",
    "PatchUnEmbed1D",
    "RSTB",
    "ResViTBlock",
    "SWAtten",
    "SwinBlock",
    "SwinTransformerBlock",
    "TransDecoder",
    "TransDecoder2",
    "TransHyperScale",
    "WMSA",
    "WinNoShiftAttention",
    "WindowAttention",
    "WindowedCrossAttention",
    "build_window_attention_mask",
    "entroformer_clones",
    "infer_swatten_attention_dim",
    "infer_swatten_head_dim",
    "infer_swatten_window_size",
    "pad_to_window_multiple",
    "window_partition",
    "window_reverse",
]


def __getattr__(name):
    if name == "Win_noShift_Attention":
        from .swin import Win_noShift_Attention as _alias

        return _alias
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
