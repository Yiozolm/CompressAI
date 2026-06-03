# Task Plan: TinyLIC + ShiftLIC 集成

## Goal

把 `candidate/TinyLIC/` 与 `candidate/ShiftLIC/` 一起迁入 `compressai/`。**TinyLIC 是 base**，ShiftLIC large 是把 TinyLIC 的 transform 全部替换成 shift block 后的衍生型号，两者**共享熵模型**。集成顺序：先 TinyLIC（含可复用熵模型 codec），再 ShiftLIC 复用。

## 关键发现：TinyLIC ↔ ShiftLIC 的关系

直接逐字 diff `candidate/TinyLIC/compressai/models/tinylic.py` 与 `candidate/ShiftLIC/shiftlic/large/train.py`：

| 部分 | TinyLIC | ShiftLIC large | ShiftLIC small/middle |
|---|---|---|---|
| `g_a / g_s` | `conv5/3 + ResViTBlock`（NSA + Mlp） | `PixelUnshuffle/Shuffle + 1×1 + ResidualBlockShift × 3` per scale | 同 large；middle/large 在 192/256 尺度后插 `CheapCS1` |
| `h_a / h_s` | `conv3 + ResViTBlock` | `ResidualBlockShift + LeakyReLU + PixelUnshuffle/Shuffle` | 同 large，但末层输出 N→M（scale-only），而 large 输出 N→2M |
| `MultistageMaskedConv2d` | mask A/B/C 三型 | **完全相同** | 不用 |
| `sc_transform_{1..5}` 配置 | A 3×3, B 5×5, C 5×5, B 5×5, B 5×5 | **完全相同** | 不用 |
| `entropy_parameters_{1..4}` 通道收缩 | 12/3 → 10/3 → 8/3 → 6/3 | **完全相同** | 不用 |
| `cc_transforms[4]` | `conv5 + GELU + conv5 + GELU + conv3` | `ResidualBlockShift × 4 + GELU` | 不用 |
| `gamma_func` | `cosine` | `linear` | 不用 |
| `Demultiplexer/v2`、`Multiplexer/v2` | 一致 | 一致 | 不用 |
| `forward / compress / decompress` 主循环 | TinyLIC 实现 | ShiftLIC 是**逐字翻版** | scale-only `ScaleHyperprior` |
| `_init_weights / aux_loss / update / load_state_dict` | TinyLIC 实现 | ShiftLIC 完全复用 | 同上简化 |

**结论**：抽一个通用的 `MultistageCheckerboardLatentCodec` 一份代码同时服务 TinyLIC 与 ShiftLIC large；`gamma_mode` 与 `make_cc_transform` 由调用方注入。

## 上游元数据

| 项 | TinyLIC | ShiftLIC |
|---|---|---|
| Paper | Lu & Ma, 2022 (arXiv 2204.11448) | Bao et al., TCSVT 2025 (arXiv 2503.23052) |
| 上游 commit | `b9081f7` (2023-12-04) | `21e52e1` (2025-04-17) |
| License | **Apache 2.0** | 未声明（按既定政策不阻塞） |
| 预训练权重 | NJU Box 公开（Q1–Q8） | 上游 README "will be made available soon"，候选目录无 `.pth.tar` |
| 数值等价校验 | 可做（有公开权重） | **不可做**（无权重），仅 forward smoke |

## TinyLIC：依赖与共享层盘点

- `compressai/layers/natten/`（候选 vendor 的 NATTEN）：
  - `nattencuda.py` —— Linux+CUDA 专用的 C++ 扩展（`nattenav_cuda` / `nattenqkrpb_cuda`）
  - `nattentorch.py` —— **纯 PyTorch 退路**（macOS / CPU 可用，性能差但能跑通）
  - 本仓策略：**严格可选依赖**`natten`（PyPI 包，NVIDIA SHI Lab 维护，仅 Linux+CUDA），缺失时落到 vendor 的 nattentorch fallback；不提供 CUDA 编译，仅 lazy import + import-on-use
- `Mlp` / `NSABlock` / `BasicViTLayer` / `ResViTBlock` —— 现仓无，新建
- `MultistageMaskedConv2d`（mask A/B/C）—— 现仓无，新建（顶层 `compressai/layers/layers.py`）
- `Demultiplexer/v2` / `Multiplexer/v2` / `Space2Depth` / `Depth2Space` —— 现仓无，新建
- `quantize_ste` —— 现仓 `compressai/ops/` 已有 `ste_round` / `quantize_ste`，复用
- `update_registered_buffers` / `conv` / `deconv` —— 现仓 `compressai/models/utils.py` 已有

## ShiftLIC：增量盘点

- `Shift4` / `ResidualBlockShift` / `CheapChannelV1` / `CheapCS1` / `channel_shuffle` —— 现仓无，新建
- 熵模型部分**全部复用** TinyLIC 落地的 `MultistageCheckerboardLatentCodec`（仅传 `gamma_mode="linear"` 与 `make_cc_transform=ResidualShiftStack`）
- small/middle 走 `ScaleHyperprior` 风格（无 mean，输出 N→M），不用 codec

## Phases

> 顺序很重要：Phase 0–4 完成 TinyLIC，Phase 5–8 完成 ShiftLIC，Phase 5 直接复用 Phase 1 的 codec。

### TinyLIC

- [X] **Phase 0：共享层（TinyLIC 必需）**
  - [X] `compressai/layers/layers.py` 追加 `MultistageMaskedConv2d`（mask A/B/C），与 `MaskedConv2d` / `CheckerboardMaskedConv2d` 同位
  - [X] `compressai/layers/attn/nsa.py`（新建）：`NSABlock`、`BasicViTLayer`、`ResViTBlock`；底层 `NeighborhoodAttention` 通过 `compressai.layers.natten` 适配层引入
  - [X] `compressai/layers/natten/__init__.py`（新建）：lazy 路由（`is_natten_available()` 探测；`NeighborhoodAttention` 始终用 vendor torch fallback 以保 state_dict 1:1，详见 Errors §1）
  - [X] `compressai/layers/natten/_torch_impl.py`（新建）：vendor 的 `NeighborhoodAttentionTorch` 纯 PyTorch 实现
  - [X] `compressai/ops/multiplex.py`（新建）：`space2depth(x, r=2)` / `depth2space(x, r=2)` / `demultiplex(x)` / `multiplex(a, na)` / `demultiplex_v2(x)` / `multiplex_v2(y1, y2, y3, y4)` —— 纯函数，TinyLIC 与 ShiftLIC large compress/decompress 共用
  - [X] `compressai/layers/__init__.py` 通过 `from .layers/.attn import *` 自动 re-export `MultistageMaskedConv2d` / `ResViTBlock` / `NSABlock` / `BasicViTLayer`
  - [X] `pyproject.toml`：`[project.optional-dependencies]` + `[dependency-groups]` 增 `tinylic = ["natten; platform_system == 'Linux'"]`
  - [X] `tests/test_layers.py` 加 `MultistageMaskedConv2d`（mask A/B/C 形状与零位置）+ `multiplex` / `multiplex_v2` 圆 trip smoke

- [X] **Phase 1：通用 staged latent codec**
  - [X] `compressai/latent_codecs/multistage_checkerboard.py::MultistageCheckerboardLatentCodec`（新建），构造参数与 plan 一致，但 **`make_cc_transform` 签名简化为 `(in_ch, out_ch) -> nn.Module`**（去掉 `last` flag——upstream 所有 iter 同结构，仅 in/out 不同）
  - [X] `compressai/latent_codecs/__init__.py` re-export
  - [X] 默认 `_default_make_cc_transform`（TinyLIC 风格）已修正 inner channel = `out_ch // 2`（详见 Errors §2）

- [X] **Phase 2：TinyLIC 模型主体**
  - [X] `compressai/models/tinylic.py::TinyLIC(CompressionModel)`，顶层属性沿用候选命名 `g_a0..g_a7` / `g_s0..g_s7` / `h_a0..h_a3` / `h_s0..h_s3`
  - [X] `forward` / `compress` / `decompress` 委托给 `MultistageCheckerboardLatentCodec`；`load_state_dict` 自动把上游顶层 `entropy_parameters_*` / `cc_transforms.*` / `sc_transform_*` / `gaussian_conditional.*` 重定向到 `latent_codec.*`
  - [X] `from_state_dict`：从 `g_a0.weight` 推 N，从 `g_a6.weight` 推 M
  - [X] `compressai/models/__init__.py` 通过 `from .tinylic import *` re-export `TinyLIC`

- [X] **Phase 3：TinyLIC zoo + checkpoint convert**
  - [X] `compressai/zoo/image.py`：`candidate_model_architectures["tinylic"] = TinyLIC`，`tinylic` factory；URL 留空
  - [X] `compressai/zoo/__init__.py`：import + 写入 `image_models["tinylic"]`
  - [X] `examples/convert_tinylic_checkpoint.py`：`module.` 前缀剥离 + 1:1 落键；已实测 `candidate/ShiftLIC/checkpoint_q1.pth.tar`（实际是 TinyLIC q1）forward smoke

- [X] **Phase 4：TinyLIC 测试**
  - [X] `tests/test_models.py::TestModels::test_tinylic`：256×256 forward smoke + 双向 state_dict roundtrip + compress/decompress 比特流圆 trip
  - [X] `tests/test_zoo.py::TestCandidateModels::test_tinylic`：zoo factory smoke + pretrained=True 抛错

### ShiftLIC

- [X] **Phase 5：ShiftLIC 共享层**
  - [X] `compressai/layers/lic/shift.py`（新建）：`Shift4` / `ResidualBlockShift` / `CheapChannelV1` / `CheapCS1` / `channel_shuffle` / `ResidualShiftStack(in_ch, out_ch)`（inner = out_ch // 2，最后一层 double）
  - [X] `compressai/layers/lic/__init__.py` re-export；顶层 `compressai/layers/__init__.py` 通过 `from .lic import *` 自动透出
  - [X] `tests/test_layers.py::TestShift`：4 方向 shift 几何 + ResidualBlockShift forward + ResidualShiftStack 形状 + odd out_ch 拒绝 + CheapCS1 forward

- [X] **Phase 6：ShiftLIC 模型主体**
  - [X] `compressai/models/shiftlic.py::ShiftLIC(CompressionModel)`，cfg 化 `variant: Literal["small","middle","large"]`，共用 `_build_encoder/_decoder/_hyperencoder/_hyperdecoder`
  - [X] small/middle 走 ScaleHyperprior 风格（hyper_out=M，scale-only `GaussianConditional`，hyperencoder 喂 `torch.abs(y)`）
  - [X] large 实例化 `MultistageCheckerboardLatentCodec(gamma_mode="linear", make_cc_transform=ResidualShiftStack)`；hyperencoder 喂 raw y（与上游对齐）
  - [X] `from_state_dict` 自动推 variant（codec-only 前缀 → large；encoder 含 `CheapChannel*` → middle；否则 small）；N 从 `hyperencoder.0.conv2.weight` 推；M 从 `encoder.{16 if small else 18}.weight` 推
  - [X] `compressai/models/__init__.py` re-export `ShiftLIC`

- [X] **Phase 7：ShiftLIC zoo + checkpoint convert 占位**
  - [X] `compressai/zoo/image.py`：增 `shiftlic-small` / `shiftlic-middle` / `shiftlic-large` 三 entry；factory 由 `_shiftlic_factory(variant)` 闭包生成
  - [X] **不**注册裸 `shiftlic` 别名，强制三选一
  - [X] `examples/convert_shiftlic_checkpoint.py`：最小骨架（`module.` 剥离 + 1:1 + smoke），TODO 注上游放真实权重再补 forward diff

- [X] **Phase 8：ShiftLIC 测试**
  - [X] `tests/test_models.py::TestModels::test_shiftlic[small/middle/large]`：256×256 forward smoke + state_dict roundtrip + variant 推断
  - [X] `tests/test_zoo.py::TestCandidateModels::test_shiftlic[*]`：zoo factory smoke + pretrained=True 抛错
  - [X] **不做**：compress/decompress 圆 trip（无真实权重无法校验比特流）

### 收尾

- [X] 更新 `candidate/TODO.md`：在 `主流老式 baseline` 段尾追加 `TinyLIC` 与 `ShiftLIC` 条目，标注复用关系；`Shared Foundations` 回填 `MultistageMaskedConv2d` / `MultistageCheckerboardLatentCodec` / `compressai/ops/multiplex` / `nsa attn` / `lic/shift` / `natten` 可选依赖
- [X] 更新 `plan/lic-migration-roadmap.md`：把 TinyLIC 与 ShiftLIC 登记到候选池（14 → 16 个 model entry）

## Key Questions（实施前需拍板）

1. **NATTEN 依赖策略**：
   - (a) 严格可选 + 缺失走 vendored nattentorch fallback（macOS dev 友好，性能差）
   - (b) 严格可选 + 缺失则 `TinyLIC.__init__` 抛错（不维护 fallback）
   - 倾向 (a)，与现仓 `mamba_ssm` / `selective_scan_ref` 风格一致
2. **`MultistageCheckerboardLatentCodec` 命名**：
   - (a) 通用名 `MultistageCheckerboardLatentCodec`（推荐）
   - (b) 沿用上游 `TinyLICStagedLatentCodec`（ShiftLIC 复用时显得别扭）
3. **ShiftLIC variant 暴露**：
   - (a) 3 个 zoo entry（`shiftlic-small/middle/large`），与 `hpcm-base/large/phi` 一致
   - (b) 单 entry + `variant` kwarg（更紧凑，但 `from_state_dict` 自动推断更复杂）
   - 倾向 (a)
4. **TinyLIC 预训练权重镜像**：
   - 上游放在 NJU Box（私有链接，可能挂掉）。是否拷一份到 compressai S3？
   - 倾向：先不镜像，`candidate_model_urls["tinylic"] = {}`，文档注明上游链接；用户本地下载 + `from_state_dict` 加载
5. **`compress/decompress` 是否一并迁**：
   - TinyLIC 有公开权重，**强烈建议一起做并验证比特流圆 trip**（Phase 4）
   - ShiftLIC 无权重，可先留 `NotImplementedError`，等上游放权重再补；但因为 codec 是共享的，做 TinyLIC 时已经把全部代码写好了，ShiftLIC 直接打开开关即可，几乎零成本
   - 倾向：两个一起做

## Decisions Made

- **NATTEN**：可选依赖 + 缺失走 vendored `nattentorch` 纯 PyTorch fallback（与 mamba_ssm 风格一致；macOS dev 可跑）。
- **通用 codec 命名**：`MultistageCheckerboardLatentCodec`（ShiftLIC large 复用时语义清晰）。
- **ShiftLIC zoo 暴露**：3 个独立 entry `shiftlic-small/middle/large`，与 hpcm 系列一致。
- **TinyLIC 权重**：不镜像到 S3，`candidate_model_urls["tinylic"] = {}`，文档注上游 NJU Box 链接。
- **compress/decompress**：TinyLIC + ShiftLIC large 一并迁。codec 共享，ShiftLIC large 顺手打开开关。

## Errors Encountered

（实施过程中遇到的坑回填）

## Status

**Done** —— 9 phase 全部落地。

- TinyLIC + ShiftLIC small/middle/large 共 4 个 zoo entry 已注册，对应模型在 `compressai/models/{tinylic,shiftlic}.py`
- 通用 codec `compressai/latent_codecs/multistage_checkerboard.py::MultistageCheckerboardLatentCodec` 同时服务 TinyLIC 与 ShiftLIC large
- 共享层：`compressai/layers/layers.py::MultistageMaskedConv2d`、`compressai/layers/attn/nsa.py`（NSABlock / BasicViTLayer / ResViTBlock）、`compressai/layers/lic/shift.py`（Shift4 / ResidualBlockShift / CheapCS1 / ResidualShiftStack）、`compressai/layers/natten/`（lazy router + nattentorch fallback）、`compressai/ops/multiplex.py`
- pyproject `tinylic` extras：`natten; platform_system == 'Linux'`
- 测试：`tests/test_layers.py`（mask 形状 + multiplex round-trip + Shift 几何 + ResidualShiftStack/CheapCS1 forward）、`tests/test_models.py::TestModels::{test_tinylic, test_shiftlic[small/middle/large]}`、`tests/test_zoo.py::TestCandidateModels::{test_tinylic, test_shiftlic[*]}`
- convert 脚本：`examples/convert_tinylic_checkpoint.py`（已实测 `candidate/ShiftLIC/checkpoint_q1.pth.tar`，实际是 TinyLIC q1：N=128 / M=320 / 28.3M 参数 → 256² PSNR=43.01 dB / total_bpp=0.0175）、`examples/convert_shiftlic_checkpoint.py`（骨架，等真实 ShiftLIC 权重）

### Errors Encountered（实施过程中的坑）

1. **NATTEN PyPI 接口与 vendor 实现不兼容**：PyPI `natten.NeighborhoodAttention2D` 与 vendor `NeighborhoodAttention` 的 ctor 签名 + state_dict 键不一致；为保持 state_dict 1:1，目前总是用 vendor torch fallback；natten extras 仅作 future-fast-path 占位。日后若要接 PyPI natten，需写一个保留 `qkv`/`proj`/`rpb` 命名的 wrapper 调 `natten.functional.na2d_qk{rpb}` / `na2d_av`。
2. **`cc_transforms` inner channel 数搞错**：第一版我让 inner 通道 = `out_ch`（认为 out_ch 即是真实输出宽度），但上游 TinyLIC / ShiftLIC large 的真实结构是 inner = `out_ch // 2`，仅最后一层 double 到 `out_ch`（`out_ch = 2 * slice_size`，scale + mean）。`make_cc_transform` 接口签名里就是 `(in_ch, out_ch_doubled)`，工厂内部需要做 `inner = out_ch // 2` 再 ramp。第一次 convert real q1 weight 才发现，复测后已改正。
3. **ShiftLIC large 的 `gaussian_conditional` top-level alias 撞键**：原想让 `self.gaussian_conditional = self.latent_codec.gaussian_conditional` 给小/中/大三 variant 一份统一接口，结果 nn.Module 会把同一对象在 state_dict 里登记两次（`gaussian_conditional.*` + `latent_codec.gaussian_conditional.*`），加载顶层 alias 缺失时直接报 missing key。最终把 large 的顶层 alias 干掉，`update` / `aux_loss` 走 `named_modules()` 仍能找到 codec 内部那份。
4. **ShiftLIC variant 推断不能依赖 `gaussian_conditional`**：ShiftLIC small/middle 自己持有 `gaussian_conditional`，所以变体推断的 `_CODEC_PREFIXES` 不能包含 `gaussian_conditional.`，否则三 variant 都会被推断成 large。变体探测要用 codec-only 前缀：`entropy_parameters_` / `cc_transforms.` / `sc_transform_`。
5. **小通道 small variant 的 encoder 索引不同**：`encoder.18.weight` 只在 middle/large 存在（CheapCS1 占两个 idx）；small 用 `encoder.16.weight`。`from_state_dict` 必须先推 variant 再选 encoder 末层 idx。
6. **`candidate/ShiftLIC/checkpoint_q1.pth.tar` 实际是 TinyLIC checkpoint**：上游 ShiftLIC README 说权重 "will be made available soon"，候选目录唯一 `.pth.tar` 用的是 `g_a/g_s/h_a/h_s` + cosine slice [24,69,104,123]，是 TinyLIC q1 ckpt。补 ShiftLIC 数值等价校验需要等真实 ShiftLIC 权重。
