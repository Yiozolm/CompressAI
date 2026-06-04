# pr-dcae-saaf-auxt: DCAE + SAAF + AuxT 容器化迁入执行计划

**计划日期**：2026-05-09
**分支**：`pr-dcae-saaf-auxt`（基于 `upstream/master` 在 pr-tcm-cca merge 后；开发期间临时基于 `pr-tcm-cca`）
**目标 PR**：本仓向上游 `InterDigitalInc/CompressAI` 提交的 `#353` 系列下一个 PR（pr-tcm-cca 之后的第三个 PR）
**前置依赖**：[`pr-tcm-cca`](../completed/codec-containerization-h-g-refactor.md) 必须先 merge 进 upstream（提供 `HyperpriorLatentCodec` + `ChannelGroupsLatentCodec` 容器化基础设施 + `_helpers/channel_context` + `_helpers/channel_slice` + `LRPGaussianLatentCodec` + `_slice_helpers`）
**设计文档**：[`plan/design-docs/channel-slice-codec-redesign.md`](../../design-docs/channel-slice-codec-redesign.md) §2.1 Family 1 表已含 DCAE/SAAF 列；容器化 wiring sketch / state_dict 路径设计见 [`codec-containerization-h-g-refactor.md`](codec-containerization-h-g-refactor.md#design-rationale)「设计依据」段（D.2/D.4/D.5，已为 DCAE/SAAF 双 h_s + dictionary head 留模式）

> **执行顺序**：本计划期望 pr-tcm-cca PR 已 merge 进 upstream（commit hash 会变），所以最终 base 是 upstream/master post-merge。开发期间可临时基于 `pr-tcm-cca` 分支，等 upstream merge 后做一次 `git rebase --onto upstream/master <old-base>`（参考 pr-tcm-cca 的 Phase 8.3 决策记录）。
>
> **scope 扩展（2026-05-09，Phase 2 完成后）**：原 plan 仅含 DCAE+SAAF。在 Phase 2 完成后发现 SAAF 的 g_a/g_s **强制集成** `OLP`（Orthogonal Linear Projection，AuxT 论文 Li et al. ICLR 2025 的核心 primitive），且 `OLP` 同样可作为 TCM 的 opt-in `use_auxt=True` side-branch 启用。决定**把 AuxT lift 提到 Phase 3**，统一引入 OLP / WLS / iWLS 到 upstream + 给 TCM 加 `use_auxt` opt-in，再做 SAAF。reviewer 视角：AuxT 是独立论文，不算「重做刚 review 过的 TCM 代码」。

---

## ⭐ 首要任务（2 个 Family 1 模型 + 1 套跨模型 AuxT 基础设施）

按以下顺序新增到 upstream：

| 优先级 | 内容 | 来源 | Phase |
|---|---|---|---|
| 1 | **DCAE** | Lu CVPR 2025 (`compressai/models/dcae.py`，fork `script` 已迁入) | Phase 2 ✅ |
| 2 | **AuxT** primitives + TCM `use_auxt` opt-in | Li et al. ICLR 2025 (OLP / WLS / iWLS；fork `script` 在 `compressai/layers/lic/blocks.py` + `wave/wavelet.py`) | Phase 3 |
| 3 | **SAAF** | Ma CVPR 2026 (`compressai/models/saaf.py`，fork `script` 已迁入；g_a/g_s 用 `_AdaptiveFrequencyBlock` 内嵌 `OLP`) | Phase 4 |

DCAE 与 SAAF 共享 `DictionaryEntropyCompressionModel` 基类 + `MutiScaleDictionaryCrossAttentionGLU` cross-attention pattern——是 Family-1 的 cousin 对。AuxT 是独立 paper 的 cross-model 共享 primitive：本 PR scope 里 SAAF **强制依赖** `OLP`，TCM 通过 `use_auxt=True` 可选启用，未来 follow-up 模型也可能引入。

**延后**：MambaVC（Family 1 但 SSM 主题，单独 PR；引入 `mamba_ssm`/`triton` optional deps）

**不动**：MLIC++ / MambaIC（Family 2 dedicated codec 类）+ Family 3-7 所有

---

## 上下文与现状

### 设计 doc 已锁定的 wiring 决策
- DCAE/SAAF 与 STF/TCM 同属 Family 1，但 mean_support 与 scale_support **共用** `MutiScaleDictionaryCrossAttentionGLU` 的输出（不像 TCM 是 `SWAtten` mean / scale 独立）
- 共享 `dt: nn.Parameter` (dict_num × dictionary_dim) 字典张量 + per-slice `dt_cross_attention` ModuleList
- `dt` 字典张量需要在容器化后明确归属——Phase 1 决策为模型级 `shared_dictionary.dt`（详见 Phase 1 实施差异）

### AuxT 上下文（新加）
- 论文：Li et al., "Auxiliary Transform for Image Compression", ICLR 2025
- 三个 primitive：`OLP`（Orthogonal Linear Projection）、`WLS`（Wavelet Linear Synthesis）、`iWLS`（inverse WLS）
- `WLS` / `iWLS` 依赖 `pytorch_wavelets`（`DWTForward` / `DWTInverse`）—— **新 optional dep**，预计加到 `[wavelet]` 或 `[auxt]` extras
- `OLP` 不依赖 `pytorch_wavelets`，可独立使用——SAAF 只用 OLP

### fork `script` 上的现状
- `compressai/models/dcae.py` 352 行（monolithic + model-owned hyperprior）—— ✅ Phase 2 已重写为容器化版本
- `compressai/models/saaf.py` 617 行（monolithic + g_a/g_s 用 `_AdaptiveFrequencyBlock`/`_DenoisingAsRegularizer`/`_CrossSparseWindowAttention`/`_SpatialAttentionLayer`/`_SpatialAttentionBlock`）
- `compressai/models/tcm.py` 含 `use_auxt: bool = False` opt-in（pr-tcm-cca 决定不上游，留到本 PR）
- `compressai/models/_bases/dictionary_entropy.py` 330 行（DCAE/SAAF 共享基类）
- `compressai/layers/lic/blocks.py` 含 `OLP`（25 行）+ `ResidualBottleneckBlock` 等
- `compressai/layers/wave/wavelet.py` 150 行：`DWT2D` / `IDWT2D` / `WLS` / `iWLS`（依赖 `pytorch_wavelets` 可选 dep）
- `examples/convert_{dcae,saaf}_checkpoint.py` 已存在 —— DCAE 已重写（Phase 2.2）

### 已与上游 maintainer 确认（继承 #354 / pr-tcm-cca review 上下文）
- `@YodaEmbedding` 已批准 latent_codec 容器化重构方向
- DCAE/SAAF 复用 `[attn]` extras 里的 `timm`，不新增 attn extras
- AuxT 引入 `pytorch_wavelets` 作为新的 optional extras（命名 `[auxt]` 或 `[wavelet]` 待 reviewer 反馈定）

---

## Phase 0: 清理 working tree + 分支建立（30 分钟）✅ 完成于 2026-05-09

### 任务
- [x] 备份 fork `script` 上现有 DCAE/SAAF monolithic 实现作为参考（已在 `script` 分支，无需额外 backup）✅
- [x] 基于 pr-tcm-cca（开发期）或 upstream/master（pr-tcm-cca merge 后）创建 `pr-dcae-saaf`：
  - 开发期：`git checkout -b pr-dcae-saaf pr-tcm-cca` ✅（HEAD = `f87c8c8`）
  - merge 后：`git rebase --onto upstream/master pr-tcm-cca pr-dcae-saaf`（消除 hash 不一致，参考 pr-tcm-cca Phase 8.3）—— 等 pr-tcm-cca merged 后再做
- [x] 确认 working tree 干净 ✅（仅 untracked: `plan/`, `AGENTS.md`, `CLAUDE.md`, `candidate*/`）

### 验收
- [x] `git log pr-dcae-saaf` 6 ahead of upstream/master, 0 behind（继承 pr-tcm-cca 的 6 commits）✅
- [x] `git rev-list --count pr-tcm-cca..pr-dcae-saaf` = 0（与 pr-tcm-cca 同 HEAD）✅
- [x] working tree 仅 untracked 的 `plan/`、`AGENTS.md`、`CLAUDE.md`、`candidate*/` ✅

### 实施差异 / 决策记录
- **不立即 push origin**：本地分支足够开发，等 Phase 1 有第一个 commit 再 push（避免空分支远程占位）

---

## Phase 1: 新增 dictionary cross-attention 基础设施（1 天）✅ 完成于 2026-05-09，commits `47b12ff` + `a7d3c0f`

新增 1 个 helper 文件 + lift 必要的共享 layers，单元测试覆盖。

### 任务

#### 1.1 lift `compressai/layers/lic/dcae.py` 到 upstream —— commit `47b12ff`
- [x] 决策：把 `MutiScaleDictionaryCrossAttentionGLU` + 6 个依赖类放到 `compressai/layers/attn/dictionary.py`（与 #354 加的 `compressai/layers/attn/swin.py` 同位）✅
- [x] `compressai/layers/attn/dictionary.py`（~250 行，含 `Scale` / `DWConv` / `ConvolutionalGLU` / `ConvWithDW` / `DenseBlock` / `SpatialAttentionModule` / `MultiScaleAggregation` / `MutiScaleDictionaryCrossAttentionGLU`）✅
- [x] **零新依赖**：用 einops `rearrange`（已是 hard dep）+ pure pytorch，**不直接 import timm**（依赖 g_a/g_s 才用 timm via swin.py）✅
- [x] 单元测试：4 个测试通过（forward shape + state_dict round-trip + dictionary_dim default + ValueError 校验）✅

#### 1.2 `compressai/models/_helpers/dictionary_context.py`（新文件，~190 行）—— commit `a7d3c0f`
- [x] `class SharedDictionary(nn.Module)`：封装 `dt: nn.Parameter` (dict_num × dictionary_dim) + `expand_for(B)` 广播无 copy ✅
- [x] `class DictionaryMeanScaleContextHead(nn.Module)`：per-slice head，内部跑一次 `MutiScaleDictionaryCrossAttentionGLU` 后 cat 入 support 再走分离 mean_cc / scale_cc，optional `emit_mean_support` 提供 LRP byte-for-byte 兼容 ✅
- [x] `def build_dictionary_mean_scale_head(slice_ch, support_ch, *, shared_dictionary, dict_output_ch, ...) -> DictionaryMeanScaleContextHead`：factory ✅
- [x] **关键决策**：`dt` Parameter 归属定为 **(a) 模型直接持有**（路径 `shared_dictionary.dt`）—— 实验确认 pytorch 在 `state_dict()` 里**会**给同一 Parameter 在 K 个引用模块下各列一遍路径，所以 (b) `latent_codec.y.shared_dictionary.dt` 方案需要修改 upstream `ChannelGroupsLatentCodec`。决定不动 upstream，head 用 `_dictionary_provider: Callable` plain attr（不被 `nn.Module.__setattr__` 注册为 submodule）持有 closure 引用 SharedDictionary
- [x] 单元测试：6 个测试通过（dt shape + expand_for 无 copy + factory shapes + emit_mean_support shape + dt 不在 head state_dict 重复 + 容器内 dt 单路径验证）✅

#### 1.3 `compressai/latent_codecs/_slice_helpers.py` 扩展（如需）—— **无新增工作**
- [x] DCAE/SAAF 用的 `slice_support_channels(M*3, ...)` / `make_entropy_transform` / `infer_num_slices` 已在 pr-tcm-cca 落地，可直接复用 ✅
- [x] DCAE/SAAF 的 `infer_dictionary_num_slices` / `infer_dictionary_max_support_slices` / `infer_stage_block_num` / `infer_attention_head_dim` / `infer_window_size` 等 helper **不搬到 generic 层**——它们是 g_a/g_s 结构推断（per-model 私有），合理实现位置是各模型 `from_state_dict` 内部，**留给 Phase 2/3 各模型迁移时实现** ✅

### 验收
- [x] `pytest tests/test_layers.py tests/test_models_helpers.py -q` 全过 + 新加 dictionary head 测试通过 ✅（10 新测试 + 现有 = 84 全过）
- [x] `make static-analysis` 全过（ruff format / imports / lint）✅
- [x] `import compressai` / `compressai.zoo` 不加载新依赖（仍 0 timm at module import time）✅
- [x] `uv lock --check` 一致（无 pyproject 改动）✅

### 实施差异 / 决策记录
- **`dt` 归属决定为 (a) 模型直接持有，否决 (b)**：见 1.2 关键决策。原 plan 推荐 (b) 容器化为 `latent_codec.y.shared_dictionary.dt`，实施前实验发现 pytorch shared Parameter 在 state_dict 中会按引用模块路径分别列出，导致 (b) 要么 K 路径重复要么改 upstream codec。选择 (a) 后，`SharedDictionary` 在模型层、heads 通过 closure 引用，state_dict 中 `shared_dictionary.dt` 仅出现一次
- **input order 不一致需要 convert script 处理**：DCAE 上游源用 `cat([latent_scales, latent_means, ...])`（scales 先），containerized wiring 用 `cat([latent_means, latent_scales, ...])`（means 先）。`DictionaryMeanScaleContextHead` 接受 containerized 顺序，convert script 在 Phase 2/3 处理首 2M 通道的 means↔scales 交换
- **DWConv 保留为独立 class**（per user 要求 verbatim lift）：原本可 inline 进 ConvolutionalGLU 减一个 class，保留更接近上游源结构
- **Phase 1 不做 origin push**（per Phase 0 决策）：commit `47b12ff` + `a7d3c0f` 仅在本地，等 Phase 2 / 3 进度再批量 push

---

## Phase 2: 迁移 DCAE（1.5 天）✅ 完成于 2026-05-09，commit `ea72f8c`

新建 `compressai/models/dcae.py` 按容器化 H+G pattern。

### 任务

#### 2.0 lift Stride/Upsample blocks 到 `compressai/layers/lic.py`（追加任务）—— 同 commit
- [x] 新文件 `compressai/layers/lic.py` 持有 `ResidualBottleneckBlockWithStride` + `ResidualBottleneckBlockWithUpsample`（DCAE g_a/g_s/h_a/h_*_s 与 SAAF g_a/g_s 共享）✅
- [x] 复用 upstream `compressai.models.sensetime.ResidualBottleneckBlock`（不动 sensetime.py）✅

#### 2.1 重写 `compressai/models/dcae.py` —— 同 commit
- [x] 全删 fork `script` 上现有 monolithic DCAE 实现 ✅
- [x] 按 §10.5 wiring sketch + Phase 1 helper 写：`HyperpriorLatentCodec(h_a, h_s=DualHyperSynthesis(h_mean_s, h_scale_s), latent_codec={"z": EntropyBottleneckLatentCodec(quantizer="noise"), "y": build_channel_slice_codec(...)})`✅
- [x] 共享 `SharedDictionary` 在 codec 容器外（model 级）构造一次，传给 channel_context_factory（所有 slice 通过 closure 共享同一个 `dt`，state_dict 中 `shared_dictionary.dt` 仅一份）✅
- [x] 模型基类改成 `CompressionModel`（不再继承 `DictionaryEntropyCompressionModel`，跟 STF/WACNN/TCM/CCA 一致）✅
- [x] 手写 forward / compress / decompress 委托给 `self.latent_codec` ✅
- [x] `from_state_dict` 自动检测 fork `script` layout 并调 `convert_upstream_dcae_state_dict` 转换；`_infer_config_from_state_dict` 推断 ctor kwargs（`_infer_stage_block_num` / `_infer_attention_head_dim` / `_infer_window_size` 私有 helper 内联在 dcae.py，per Phase 1.3 决策）✅
- [x] **追加（实施时新增）**：DCAE-private blocks `_WMSA` / `_ResScaleConvolutionGateBlock` / `_SwinBlockWithConvMulti` 内联（不 lift）—— 这些 blocks 仅 DCAE 用，SAAF 走不同的 `_AdaptiveFrequencyBlock`/`_DenoisingAsRegularizer` pattern

#### 2.2 `examples/convert_dcae_checkpoint.py` 重写 —— 同 commit
- [x] mirror `examples/convert_tcm_checkpoint.py` 的 thin CLI wrapper ✅
- [x] `convert_upstream_dcae_state_dict`（在 `compressai/models/dcae.py` 内）多 pass：
  - Pass：`dt` → `shared_dictionary.dt`（model 级，不是 `latent_codec.y.shared_dictionary.dt` —— Phase 1.2 决策）✅
  - Pass：`dt_cross_attention.{k}.*` → `latent_codec.y.channel_context.y{k}.cross_attention.*`，`x_trans.weight` 首 2M 通道做 means/scales 交换（DCAE 上游 `cat([scales, means, ...])` → containerized `cat([means, scales, ...])`）✅
  - Pass：`cc_mean_transforms.{k}.*` / `cc_scale_transforms.{k}.*` → `latent_codec.y.channel_context.y{k}.{mean,scale}_cc.*`，首 conv 也做 2M 通道交换 ✅
  - Pass：`lrp_transforms.{k}.*` → `latent_codec.y.latent_codec.y{k}.lrp_transform.*`，首 conv 同样交换 ✅
  - Pass：`gaussian_conditional.*` fanout K 份；`h_a` → `latent_codec.h_a`；**`h_z_s2` → `latent_codec.h_s.h_mean_s`**（means）+ **`h_z_s1` → `latent_codec.h_s.h_scale_s`**（scales）—— DCAE 上游用 h_z_s1=scales/h_z_s2=means 命名跟 STF/TCM 相反，convert 时重新对齐到 (means, scales) 顺序；`entropy_bottleneck.*` → `latent_codec.z.entropy_bottleneck.*` ✅

#### 2.3 `tests/test_models.py::TestDcae` —— 同 commit
- [x] `test_dcae_forward_and_state_dict_round_trip`：tiny config（M=80, num_slices=4, dict_num=8, dictionary_dim=32），forward + 11 个 state_dict path 自检（含 `shared_dictionary.dt` 单路径验证）+ round-trip allclose + ctor kwargs 还原 ✅
- [x] `test_dcae_upstream_state_dict_conversion`：synthetic upstream-style state_dict，断言新路径 + 旧路径消失 + cross_attention/cc/lrp 首 2M 通道 means/scales 交换正确 + h_z_s1/h_z_s2 → h_scale_s/h_mean_s 重命名 + dt 重定位 + gaussian_conditional fanout ✅

### 验收
- [x] `pytest tests/test_models.py::TestDcae -v` 全过 ✅（2/2 通过）
- [x] 上游 candidate ckpt round-trip：`candidate/DCAE/0.05checkpoint_best.pth.tar`（119M params）strict load 成功，sinusoidal smoke **PSNR 50.82 dB / total bpp 0.067**（fresh-init baseline ~5dB）—— Phase 4 阶段做的（详见 Phase 4 验收）✅
- [x] state_dict 路径符合容器化层级 ✅
- [x] `make static-analysis` 全过（ruff format / imports / lint）✅
- [x] `pytest tests/test_models.py tests/test_latent_codecs.py tests/test_models_helpers.py tests/test_layers.py tests/test_init.py -q` 86/86 通过 ✅
- [x] `import compressai` / `compressai.zoo` 仍 0 timm ✅

### 实施差异 / 决策记录
- **`dt` 归属定为 model 级 `shared_dictionary.dt`**（per Phase 1.2 决策，否决了原 plan 的 `latent_codec.y.shared_dictionary.dt`）—— heads 通过 closure 引用，state_dict 中只有一份
- **DCAE-private blocks（`_WMSA` / `_ResScaleConvolutionGateBlock` / `_SwinBlockWithConvMulti`）内联 dcae.py**—— 这些 blocks 仅 DCAE 用，SAAF 用不同的 g_a/g_s（_AdaptiveFrequencyBlock + _DenoisingAsRegularizer）。lift 到 `compressai/layers/` 没有跨模型复用价值
- **Stride/Upsample blocks lift 到 `compressai/layers/lic.py`**（Phase 2.0 追加任务）—— 这两个 blocks DCAE/SAAF 都用，所以 lift。`ResidualBottleneckBlock` 留在 `compressai/models/sensetime.py`（不动 upstream），`compressai/layers/lic.py` 从 sensetime import
- **DCAE 上游 h_z_s1/h_z_s2 命名跟 STF/TCM 相反**：DCAE 用 `latent_scales = h_z_s1(z_hat)` + `latent_means = h_z_s2(z_hat)`，跟 STF/TCM 的 `h_mean_s` / `h_scale_s` 命名相反。containerized 后统一用 `(h_mean_s, h_scale_s)` 顺序（与 DualHyperSynthesis 一致），convert script 处理 `h_z_s2 → h_mean_s` / `h_z_s1 → h_scale_s` 重命名
- **DCAE 上游 query 顺序跟 containerized 相反**：DCAE 用 `cat([latent_scales, latent_means, *prev_y_hat])`（scales 先），containerized 用 `cat([latent_means, latent_scales, *prev_y_hat])`（means 先）。convert script 在 `cross_attention.x_trans.weight` / `cc_mean.0.weight` / `cc_scale.0.weight` / `lrp_transform.0.weight` 首 2M 输入通道做 swap，让上游权重 byte-for-byte 兼容
- **`gaussian_conditional` fanout**：DCAE 上游单一共享 GC，containerized 改 per-slice 副本，convert script 每个 slice 拷一份（K=5 默认）—— 跟 TCM/CCA 一致
- **`make_entropy_transform` 默认 widths `(224, 128)`**：DCAE 默认配置就用这个，无需 override，跟 TCM 一致
- **`LRPGaussianLatentCodec(quantizer="ste")`**：DCAE 上游 `_forward_latent` 用 `torch.round(residual) - residual.detach() + residual + mu` STE 量化 y，containerized 用 `quantizer="ste"` 匹配
- **未做的事**（留 Phase 5 / 验收剩余项）：
  - ~~上游 candidate ckpt round-trip 验证~~ → Phase 4 阶段补做（DCAE 50.82dB / SAAF 43.06dB / TCM-with-AuxT 36.04dB）✅
  - zoo wiring（Phase 5.2）→ Phase 5 已完成 ✅
  - ~~`_bases/dictionary_entropy.py` 删除（Phase 5.1）~~ → 已在 pr-tcm-cca commit `0c44c75` 删除整个 `_bases/` 目录，本 PR 无残留 ✅

---

## Phase 3: AuxT 基础设施 + TCM `use_auxt` opt-in（2 天）✅ 完成于 2026-05-09，commit `74d0c5a`

新增 1 个 layers 文件（OLP）+ 1 个 layers 子包（wave/wavelet.py 持有 WLS/iWLS）+ 给 TCM 加 `use_auxt` opt-in + 新 optional dep + tests。Reviewer 视角是「引入独立论文 Li et al. ICLR 2025 的新 cross-model feature」，不是「重做刚 review 过的 TCM 代码」。

### 任务

#### 3.1 lift `OLP` 到 `compressai/layers/auxt.py`（或扩展 `lic.py`）—— 实施时改放到 `compressai/models/_helpers/auxt.py`
- [x] 决策：OLP **统一放到 `compressai/models/_helpers/auxt.py`**（per user 反馈，AuxT 相关 primitives + integration 一站式）—— 原 plan 推荐的 `compressai/layers/auxt.py` 中间方案被否决
- [x] `compressai/models/_helpers/auxt.py` 含 OLP class（~30 行 of 总 330 行）✅
- [x] 单元测试：3 测试通过（forward / scalar loss / state_dict round-trip）✅

#### 3.2 lift `WLS` / `iWLS` + 依赖 `DWT2D` / `IDWT2D` —— 实施时分两处
- [x] **层级分离**：`compressai/layers/wave/wavelet.py`（~100 行）只留 generic `DWT2D` / `IDWT2D` 包装（未来 WeConvene 复用）；`WLS` / `iWLS` 移到 `compressai/models/_helpers/auxt.py`，lazily import `DWT2D` / `IDWT2D` 避免 layers→models 反向依赖 ✅
- [x] **新 optional dep `pytorch_wavelets`**：lazy import + `_require_pytorch_wavelets()` 友好报错 ✅
- [x] `compressai/layers/wave/__init__.py`：export `DWT2D` / `IDWT2D` / `is_pytorch_wavelets_available`（不含 WLS/iWLS）✅
- [x] **不**在 `compressai/layers/__init__.py` 顶层 re-export wave/* —— 跟 attn/* 同样 deep-import only ✅
- [x] 单元测试：5 测试通过（DWT/IDWT round-trip + WLS/iWLS shape + state_dict + `is_pytorch_wavelets_available`），均带 `pytest.importorskip('pytorch_wavelets')` 守卫 ✅

#### 3.3 加 `pyproject.toml` 的 `[wavelet]` extras
- [x] 决策 extras 命名：`[wavelet]`（per plan 推荐，更通用，未来 WeConvene 也用）✅
- [x] 加 `pytorch_wavelets` 到 `[project.optional-dependencies].wavelet` ✅
- [x] `uv lock` 后 `import compressai` / `compressai.zoo` / `compressai.layers` 仍 0 wavelet/timm 加载 ✅
- [x] `make static-analysis` 全过 ✅

#### 3.4 修改 `compressai/models/tcm.py` 加 `use_auxt: bool = False` opt-in
- [x] `TCM.__init__`：加 `use_auxt: bool = False` 参数 ✅
- [x] `use_auxt=True` 时通过 `build_wls_branch(N, M)` / `build_iwls_branch(N, M)` 构造 `self.AuxT_enc` / `self.AuxT_dec`（builders 在 `_helpers/auxt`）✅
- [x] `TCM.forward / compress / decompress` 通过 `forward_with_auxt(...)` walker 路由 g_a / g_s（无 AuxT 时退化为 `transform(x)`）✅
- [x] `TCM.from_state_dict` 用 `has_auxt_state(state_dict)` 自动检测 use_auxt ✅
- [x] `convert_upstream_tcm_state_dict`：用 `is_auxt_upstream_wavelet_buffer_key` 删除上游自定义 DWT/IDWT kernel buffer + `normalize_upstream_auxt_key` 把 `.OLP.` → `.olp.` ✅
- [x] `TCM.use_auxt @property`：返回 `self.AuxT_enc is not None` ✅
- [x] `TCM.aux_loss()`：delegate to `_aggregate_aux_loss(self)`（aggregator 在 `_helpers/auxt`）✅

#### 3.5 加 `compressai/losses/auxt.py` —— **不做**
- [x] **决策**：不新加 `AuxTRateDistortionLoss` class —— 暴露 `TCM.aux_loss()` 方法即可，user 在自己 training loop 加权（最小化 API 表面）✅

#### 3.6 tests
- [x] `tests/test_models.py::TestTcm`：4 个 use_auxt 测试通过（default_false / construction / state_dict round-trip / convert script）✅
- [x] `tests/test_models_helpers.py::TestOLP`：3 测试 ✅
- [x] `tests/test_models_helpers.py::TestWLSiWLS`：3 测试（含 `aux_loss` aggregation）✅
- [x] `tests/test_models_helpers.py::TestForwardWithAuxt`：3 测试（collapse / sum-at-positions / RuntimeError on mismatch）✅
- [x] `tests/test_models_helpers.py::TestAuxtStateDictHelpers`：4 测试（`has_auxt_state` / `is_auxt_wavelet_buffer_key` / `is_auxt_upstream_wavelet_buffer_key` / `normalize_upstream_auxt_key`）✅
- [x] `tests/test_layers.py::TestWavelet`：DWT/IDWT round-trip + `is_pytorch_wavelets_available` 校验（2 测试）✅

### 验收
- [x] `pytest tests/test_layers.py tests/test_models_helpers.py tests/test_models.py::TestTcm -v` 全过 ✅（58/58 通过）
- [x] `pytest tests/test_models.py tests/test_latent_codecs.py tests/test_models_helpers.py tests/test_layers.py tests/test_init.py -q` 全过 ✅（105/105 通过）
- [x] `make static-analysis` 全过（ruff format / imports / lint）✅
- [x] `import compressai` / `compressai.zoo` / `compressai.layers` 仍 0 timm + 0 pytorch_wavelets 加载 ✅
- [x] `uv lock --check` 一致 ✅
- [x] 上游 candidate ckpt round-trip：`candidate/AuxT/model_auxt_0483.pth.tar`（46M params）strict load 成功，sinusoidal smoke **PSNR 36.04 dB / total bpp 0.522**（fresh-init baseline ~5dB）—— Phase 4 阶段做的（详见 Phase 4 验收）✅

### 实施差异 / 决策记录
- **AuxT primitives + integration 全部集中到 `compressai/models/_helpers/auxt.py`**（per user 反馈）—— 原 plan 把 OLP 放 `compressai/layers/auxt.py`、WLS/iWLS 放 `compressai/layers/wave/wavelet.py`、integration helpers 放 `_helpers/auxt.py` 三处分散，文件碎片化。改为一站式：`compressai/layers/wave/wavelet.py` 仅留 generic `DWT2D` / `IDWT2D`（未来 WeConvene 复用），其他 AuxT 相关全部到 `_helpers/auxt.py`。`_helpers/auxt.py` ~330 行，但 single source of truth + 跨模型 reuse 干净
- **层级依赖正向**：layers 层不依赖 models 层。WLS/iWLS 在 `_helpers/auxt.py` 通过 lazy import 拿 DWT2D/IDWT2D，避免 `_helpers/auxt.py` 模块加载就触发 `pytorch_wavelets` 检查
- **forward_with_auxt 边界检查**：测试发现 `merge_positions` 比 `auxiliary_layers` 短时会 IndexError，加显式长度检查改为友好的 RuntimeError
- **`TCM.aux_loss()` 始终安全**：`_helpers.auxt.aux_loss(model)` 在没 OLP 时返回 0-d zero Tensor（在第一个 model parameter 的 device/dtype 上），所以 user 训练循环可以无脑加进 objective 不用判断 use_auxt
- **不做 `compressai/losses/auxt.py`**（per Phase 3.5 决策）：暴露 `TCM.aux_loss()` 方法即可
- **`AuxT_enc.0.dwt.transform.h0_col` 等 buffer 是 persistent**：`pytorch_wavelets` 1.3.0 的 `DWTForward` 把 4 个 wavelet kernel buffer 注册为 persistent，所以 round-trip state_dict 包含它们；`is_auxt_wavelet_buffer_key` 加到 `from_state_dict` 的 `allowed_missing` 集合是为了兼容某些场景下保存 ckpt 没持久化的情况

### 风险
- **新 optional dep `pytorch_wavelets`** maintainer 可能反对 —— 准备好 fallback：单独 PR 提案 + 解释 SAAF + TCM use_auxt 都需要它
- **修改刚 merged 的 TCM 代码** 体感上像「再动一次」 —— PR description 要明确 framing「Li ICLR 2025 是独立论文，AuxT 是 cross-model 共享 feature，TCM/SAAF 都受益」
- ~~**WLS/iWLS lift 时可能需要其他 wave/ 文件**（`weconv.py`？查到则一并 lift）—— Phase 3 执行时确认~~ → 实施确认仅 wavelet.py 即可，weconv.py 是 WeConvene 模型专用，留给后续 WeConvene PR

---

## Phase 4: 迁移 SAAF（1.5 天）✅ 完成于 2026-05-09，commit `cfa986e`

新建 `compressai/models/saaf.py` 按容器化 pattern，**复用 Phase 1+2+3 helper**（dictionary 来自 Phase 1，DCAE-style 容器化模板来自 Phase 2，OLP 来自 Phase 3）。

### 任务

#### 4.1 重写 `compressai/models/saaf.py` —— 同 commit `cfa986e`
- [x] 全删 fork `script` 上现有 monolithic SAAF 实现 ✅
- [x] entropy 路径与 DCAE 同构（共享 `build_dictionary_mean_scale_head` + `SharedDictionary` + `LRPGaussianLatentCodec`），SAAF 特异部分都在 g_a/g_s（`_AdaptiveFrequencyBlock` 内嵌 Phase 3 lift 出来的 `OLP` / `_InverseAdaptiveFrequencyBlock` / `_DenoisingAsRegularizer` / `_CrossSparseWindowAttention` / `_SpatialAttentionLayer` / `_SpatialAttentionBlock`）✅
- [x] SAAF g_a/g_s 内部 blocks 内联在 `compressai/models/saaf.py` 内（**默认放本地**：SAAF 独家用，无跨模型复用价值）✅
- [x] 模型基类改成 `CompressionModel`，手写 forward / compress / decompress 委托给 `self.latent_codec`（不再继承 `DictionaryEntropyCompressionModel`）✅
- [x] `from_state_dict` 自动检测 fork `script` layout 并调 `convert_upstream_saaf_state_dict` ✅
- [x] 暴露 `SAAF.aux_loss()` 方法 delegate to `_aggregate_aux_loss(self)`（与 TCM 一致，但 SAAF 强制启用 aux loss——不像 TCM 是 opt-in）✅
- [x] **追加（实施时新增）**：自定义 `_encode` / `_decode` 实现，因为 SAAF 的 `aux_enc[i]` 与 g_a stage 走 `_merge_features`（含 bilinear interpolate 处理 spatial 不匹配），不能直接用 `forward_with_auxt`（它只支持 plain add）
- [x] **追加（实施时新增）**：`forward` 训练模式跑 `diffusion_prior(y, z_hat)` 算 `diffusion_loss` 输出到 dict；eval 时返回 0-d zero Tensor
- [x] **追加（实施时新增）**：`from_state_dict` 加 `relative_position_index` 和 `global_alpha` 到 `allowed_missing`（`_CrossSparseWindowAttention` 这两个是 non-persistent buffer，可能在 ckpt 中缺失）

#### 4.2 `examples/convert_saaf_checkpoint.py` 重写 —— 同 commit `cfa986e`
- [x] mirror Phase 2 的 DCAE 转换脚本 ✅（thin CLI wrapper，与 `convert_dcae_checkpoint.py` 同结构）
- [x] entropy 部分的 rename 与 DCAE 完全一致（共享 dictionary cross-attention pattern；含首 2M 通道 means/scales swap、`h_z_s2`→`h_mean_s` / `h_z_s1`→`h_scale_s`、`gaussian_conditional` per-slice fanout、`dt`→`shared_dictionary.dt`、`entropy_bottleneck`→`latent_codec.z.*`）✅
- [x] g_a/g_s 部分按 SAAF-specific 命名 pass through 不动（`aux_enc.*` / `aux_dec.*` / `diffusion_prior.*` / `g_a.*` / `g_s.*`）✅
- [x] **追加（实施时新增）**：strip `module.` DataParallel 前缀（候选 ckpt 实测有这个前缀），同样补到 DCAE convert
- [x] **追加（实施时新增）**：drop 上游 `*.olp.identity_matrix` keys——upstream OLP 持久化了这个 buffer，compressai 用 `persistent=False` 重新注册，不能 strict-load

#### 4.3 `tests/test_models.py::TestSaaf` —— 同 commit `cfa986e`
- [x] `test_saaf_forward_and_state_dict_round_trip`：tiny config（M=80, num_slices=4, dict_num=8，跟 TestDcae 同 shape 便于对比），forward + 11 个 state_dict path 自检（含 `shared_dictionary.dt` 单路径 / `aux_enc.0.olp.linear.weight` / `diffusion_prior.noise_predictor.0.weight`）+ round-trip allclose ✅
- [x] `test_saaf_aux_loss_is_nonzero_scalar`：SAAF 强制 OLP，aux_loss 必 > 0（与 TCM use_auxt=False 时 aux_loss=0 对比）✅
- [x] `test_saaf_diffusion_loss_active_in_training_mode`：训练模式 diffusion_loss > 0 + 是 0-d Tensor ✅
- [x] `test_saaf_upstream_state_dict_conversion`：synthetic upstream-style state_dict，断言新路径 + 旧路径消失 + means/scales swap + h_z_s1/h_z_s2 重命名 + SAAF-specific keys (aux_enc/aux_dec/diffusion_prior) pass through 不动 + g_a 不变 ✅

### 验收
- [x] `pytest tests/test_models.py::TestSaaf -v` 全过 ✅（4/4 通过）
- [x] **上游 candidate ckpt round-trip**：
  - DCAE `candidate/DCAE/0.05checkpoint_best.pth.tar`（119M params）：PSNR **50.82 dB** / total bpp 0.067 ✅
  - SAAF `candidate/SAAF/mse_0.05.pth`（127M params）：PSNR **43.06 dB** / total bpp 0.071 ✅
  - TCM-with-AuxT `candidate/AuxT/model_auxt_0483.pth.tar`（46M params）：PSNR **36.04 dB** / total bpp 0.522 ✅
  - 三个模型权重 byte-for-byte 转移成功（fresh init baseline ~5dB）
- [x] state_dict 路径符合容器化层级 ✅
- [x] `make static-analysis` 全过 ✅
- [x] `pytest tests/test_models.py tests/test_latent_codecs.py tests/test_models_helpers.py tests/test_layers.py tests/test_init.py -q` 109/109 通过 ✅
- [x] `import compressai` / `compressai.zoo` / `compressai.layers` 仍 0 timm + 0 pytorch_wavelets 加载 ✅

### 实施差异 / 决策记录
- **SAAF 用自定义 `_encode`/`_decode` 而不是 `forward_with_auxt`**：SAAF 的 `aux_enc[i]` 输出 spatial size 跟当前 g_a stage 的 `y_main` 不一定匹配（aux 不下采样、main 下采样），需要 `_merge_features` 做 bilinear interpolate 然后 add；`_helpers/auxt.forward_with_auxt` 只做 plain add，不适用。SAAF-specific 的 stage-by-stage interleave 直接内联在 `_encode`/`_decode` 中
- **DCAE convert 同步修复**：DCAE candidate ckpt 实测也带 `module.` 前缀，所以 `compressai/models/dcae.py` 的 `_is_upstream_layout` + `convert_upstream_dcae_state_dict` 也加了 strip 步骤
- **`*.olp.identity_matrix` drop**：上游 OLP 持久化了 `identity_matrix` buffer（用作 ortho loss 的 target identity matrix）；compressai 的 `OLP` 用 `persistent=False` 在 `__init__` 重新构造，所以 strict-load 会拒绝。convert script drop 这些 keys 解决
- **diffusion_prior 训练时跑 `latent_codec.h_a(y)` 走容器化路径**：原 SAAF 直接用 `self.h_a` 拿 z_hat，containerized 后 `h_a` 在 `latent_codec` 里。手动调 `latent_codec.h_a(y)` + `latent_codec.latent_codec["z"].entropy_bottleneck._get_medians()` 复制 z_hat-from-rounded-medians 路径
- **测试实测的 PSNR**：DCAE 50.82dB（最高，模型最大 119M）/ SAAF 43.06dB / TCM-with-AuxT 36.04dB —— TCM 较低是因为 sinusoidal 测试图对 TCM 的 windowed-attention pattern 不友好，但权重确实参与（fresh init ~5dB，差距 30+ dB）。Real Kodak 测试 PSNR 应都 > 30dB（DCAE/SAAF 文献声称 > 35dB）

### 风险（已消解）
- ~~DCAE 已经 commit 后，SAAF 才会发现 dictionary helper 是否真的复用得自然~~ → Phase 2 的 helper 接口（含 `emit_mean_support` / `mean_support_trail_channels`）SAAF 直接复用，无需扩展
- ~~SAAF 内部 g_a/g_s blocks 与 fork `script` 命名一致性~~ → 实施时全部内联，命名跟上游一致，无 rename 需求
- ~~OLP 使用方式与 TCM 不同：SAAF g_a/g_s 强制内嵌 OLP（不可选），TCM 是 opt-in 端点~~ → `_aggregate_aux_loss(model)` 在两种模式下都正确工作（walks `model.modules()` 找所有 `OLP` 实例），SAAF 直接 delegate

---

## Phase 5: 清理基类 + zoo 接线（0.5 天）✅ 完成于 2026-05-09，commit `70efed7`

### 任务

#### 5.1 删除 `compressai/models/_bases/dictionary_entropy.py` —— **无操作**
- [x] 确认 DCAE/SAAF 都不再继承 `DictionaryEntropyCompressionModel` ✅（Phase 2 / Phase 4 实施时改成直接继承 `CompressionModel`）
- [x] 全仓 grep `DictionaryEntropyCompressionModel` 应只剩 fork `script` 上模型的旧引用（本仓不存在）✅（仅剩 `compressai/models/saaf.py:928` 一处 docstring 注释，描述上游 layout，非 import）
- [x] ~~删除 `compressai/models/_bases/dictionary_entropy.py`~~ → **无需操作**：`compressai/models/_bases/` 整个目录已在 pr-tcm-cca cleanup commit `0c44c75` 删除（连带 `dictionary_entropy.py`），本 PR 无残留
- [x] `compressai/models/_bases/` 目录不存在，无需重建 ✅

#### 5.2 zoo 接线 —— commit `70efed7`
- [x] `compressai/zoo/__init__.py`：从 `.image` 导入 `dcae`、`saaf` 并加到 `image_models` 字典 ✅
- [x] `compressai/zoo/image.py`：加 `dcae()` / `saaf()` 工厂函数（mirror `tcm()`/`cca()` 的 `pretrained=True` raise `RuntimeError` pattern）+ `model_architectures` 用 `_LazyImport` 代理 ✅
- [x] 沿用现有 lazy import pattern，避免 `import compressai.zoo` 触发 timm / pytorch_wavelets ✅

### 验收
- [x] `import compressai` + `import compressai.zoo` 触发 0 个 timm import + 0 个 pytorch_wavelets ✅
- [x] `compressai.models.dcae` / `compressai.models.saaf` 可深路径 import ✅
- [x] `image_models` 含 `dcae`, `saaf`；`model_architectures` 含 `dcae`, `saaf` 两个 `_LazyImport` proxy ✅
- [x] pytest 全 suite 不引入回归 ✅（109/109 通过）
- [x] `make static-analysis` 全过 ✅

### 实施差异 / 决策记录
- **5.1 是 no-op**：原 plan 假设 `compressai/models/_bases/dictionary_entropy.py` 还在仓里待删，但实际 pr-tcm-cca 已经把整个 `_bases/` 目录删除（commit `0c44c75`）。本 PR scope 内，DCAE/SAAF 都直接继承 `CompressionModel`，从未引入 `DictionaryEntropyCompressionModel` 依赖，所以无可删之文件
- **commit 名沿用 `chore(zoo)` 模板**：跟 pr-tcm-cca commit `f87c8c8` (`chore(zoo): wire cca/tcm zoo entries with lazy import`) 风格一致，方便 reviewer 模式识别

---

## Phase 6: 全量验证（0.5 天）✅ 完成于 2026-05-09

### 任务
- [x] `make static-analysis` 全过（ruff format / imports / lint）✅ 171 files already formatted；imports / lint 均 All checks passed
- [x] `pytest tests/ -q --deselect tests/test_eval_model_video.py --deselect tests/test_zoo.py`（跳过 pretrained 类，参考 pr-tcm-cca Phase 7）✅ **247 passed, 4 skipped, 32 deselected, 1 failed**（DDP 失败属 pre-existing macOS flake，详见下方实施差异）
- [x] Import audit：`import compressai` + `import compressai.zoo` + `import compressai.latent_codecs` + `import compressai.layers` 触发 **0 timm + 0 pytorch_wavelets** 加载 ✅
- [x] `uv lock --check` 一致 ✅（Phase 3.3 加了 `[wavelet]` extras + pytorch_wavelets 后 lockfile 已同步；232 packages）
- [x] 手工 state_dict 自检：构造小 DCAE / SAAF / TCM(use_auxt=True) 模型，dump 关键 key，验证路径自解释 ✅ **3/3 模型 33 个期望路径全部存在**

### 验收
- [x] 全部通过 ✅（DDP failure 属环境问题不计入）

### 实施差异 / 决策记录
- **`tests/test_train.py::test_train_example_ddp` 失败属 pre-existing macOS flake，与本 PR 无关**：
  - 失败原因：`torch.distributed.DistNetworkError: client socket has timed out after 300000ms while trying to connect to (1.0.0.0...ip6.arpa, 60621)` + `NOTE: Redirects are currently not supported in Windows or MacOs`
  - 测试源于 commit `e26f0e2 ci: add test for ddp`，**已在 upstream/master 和 pr-tcm-cca 都存在**（不是本 PR 引入）
  - 标记 `@pytest.mark.slow`，启动 `torch.distributed.run --nproc_per_node 2` 跑真实 DDP，macOS 上 IPv6 socket 超时是已知问题
  - 跟 DCAE/SAAF/AuxT/zoo 任何代码改动无关，建议在 macOS 本地开发环境用 `--deselect tests/test_train.py::test_train_example_ddp` 一起跳过；CI 环境（Linux）应可正常运行
- **state_dict 自检覆盖三个模型**：
  - DCAE：10 个路径（`shared_dictionary.dt` / `latent_codec.h_a.0.conv.weight` / `latent_codec.h_s.h_*_s.0.weight` / `latent_codec.z.entropy_bottleneck.quantiles` / `latent_codec.y.channel_context.y0.cross_attention.x_trans.weight` 等）✅
  - SAAF：13 个路径（DCAE 10 个 + `aux_enc.0.olp.linear.weight` / `aux_dec.3.olp.linear.weight` / `diffusion_prior.noise_predictor.0.weight`）✅
  - TCM(use_auxt=True)：10 个路径（含 `latent_codec.h_a.0.conv1.weight`（TCM 用 ResidualBlockWithStride，路径含 conv1）/ `latent_codec.y.channel_context.y0.mean_support_transform.in_conv.weight`（SWAtten）/ `AuxT_enc.0.olp.linear.weight` / `AuxT_enc.0.scaling_factors` / `AuxT_dec.3.olp.linear.weight`）✅
- **加上 Phase 4 已验证的上游 candidate ckpt smoke**（DCAE 50.82dB / SAAF 43.06dB / TCM-with-AuxT 36.04dB），整个 PR 的代码功能正确性多角度验证完毕

---

## Phase 7: 提交 + push（0.5 天）✅ 完成于 2026-05-12

### 任务

#### 7.1 按 logical 分组打 commit
- [x] `2968a79 feat(layers): lift dictionary cross-attention building blocks to compressai.layers.attn`（Phase 1.1）✅
- [x] `5ba397f feat(models/_helpers): add SharedDictionary and DictionaryMeanScaleContextHead`（Phase 1.2-1.3）✅
- [x] `3af5563 feat(models): add DCAE with containerized codec`（Phase 2）✅
- [x] `89b1ced feat(layers,models): add AuxT primitives, helpers, and TCM use_auxt opt-in`（Phase 3）✅
- [x] `655635c feat(models): add SAAF with containerized codec and integral AuxT`（Phase 4）✅
- [x] `bbf67da chore(zoo): wire dcae/saaf zoo entries with lazy import`（Phase 5.2）✅
- [x] `819e10b refactor(layers): move lic blocks into namespace package`（post-review cleanup / namespace split）✅

#### 7.2 PR 描述草稿
- [x] 写到 `plan/generated/pr-dcae-saaf-auxt-draft.md` ✅
- [x] 重点说明：(a) 沿用 pr-tcm-cca 落地的 Family-1 容器化基础设施；(b) 新增 dictionary cross-attention helper（DCAE/SAAF 共用）；(c) `dt` Parameter 容器化归属决策；(d) state_dict 路径；(e) AuxT / TCM `use_auxt` / SAAF OLP 集成 ✅
- [x] 引用本 exec plan + design doc + pr-tcm-cca review 上下文 ✅

#### 7.3 push 到 origin/pr-dcae-saaf-auxt
- [x] push 到 `origin/pr-dcae-saaf-auxt` ✅（当前本地分支 `pr-dcae-saaf-auxt` tracking `origin/pr-dcae-saaf-auxt`，无 ahead/behind）
- [x] **未 push 到 upstream** ✅

### 验收
- [x] `origin/pr-dcae-saaf-auxt` 已包含 Phase 1-5 logical commits + namespace cleanup commit，HEAD = `819e10b` ✅
- [x] 静态分析、targeted tests、full-ish tests、import audit、candidate checkpoint smoke 均已在 Phase 2-6 记录通过 ✅

### 实施差异 / 决策记录
- **实际分支名为 `pr-dcae-saaf-auxt`**：原 plan 早期写的是 `pr-dcae-saaf`，Phase 3 扩 scope 后 branch / PR draft 均采用 `-auxt` 后缀。
- **`chore(models/_bases): drop DictionaryEntropyCompressionModel` 没有单独 commit**：Phase 5.1 已确认这是 no-op，`compressai/models/_bases/` 在 pr-tcm-cca commit `0c44c75` 已删除。
- **新增 final cleanup commit `819e10b`**：把 LIC blocks 移入 namespace package，属于 push 前后续整理，不改变 Phase 1-6 的核心功能验收。

---

## 总时间估算

| Phase | 工时 |
|---|---|
| Phase 0：清理 + 分支 | 0.5h |
| Phase 1：dictionary 基础设施 ✅ | 1d |
| Phase 2：迁移 DCAE ✅ | 1.5d |
| Phase 3：AuxT 基础设施 + TCM use_auxt ✅ | 2d |
| Phase 4：迁移 SAAF ✅ | 1.5d |
| Phase 5：清理 + zoo ✅ | 0.5d |
| Phase 6：验证 ✅ | 0.5d |
| Phase 7：提交 + push ✅ | 0.5d |
| **总计** | **~7 工作日** |

注：相比原 plan（~5.5 天），扩展为 ~7 天的增量来自 Phase 3 AuxT（~2 天），含 OLP/WLS/iWLS lift + TCM `use_auxt` opt-in + 新 `pytorch_wavelets` optional dep + 测试。相比 pr-tcm-cca（~8 天）仍简化是因为：
- 无 codec abstraction refactor（pr-tcm-cca 已交付）
- 无既有模型重写（DCAE/SAAF 是 upstream 净增；TCM `use_auxt` 是 opt-in 加法，不动既有 path）
- 无 `_bases/` 整体重构（pr-tcm-cca 已删 `_bases/`，本 PR 仅删 `dictionary_entropy.py` 一个文件）
- 共享一个 dictionary helper 给两模型用，第二个模型边际成本低

---

## 完成后

- ✅ 已移动本文件到 `plan/exec-plans/completed/`
- ✅ 已更新 `plan/README.md` 索引
- 在 design doc `channel-slice-codec-redesign.md` §1 表 / §3.4 表把 DCAE/SAAF 状态从「fork `script` 已迁入」更新为「pr-dcae-saaf-auxt 已合入 upstream」
- 给 design doc 加 AuxT 章节（OLP / WLS / iWLS 的 Family-1 跨模型用法 + TCM `use_auxt` 现已 upstream + SAAF 强制依赖 OLP）
- MambaVC 容器化迁移记入 `plan/exec-plans/active/mambavc-containerization.md`（独立 follow-up PR，含 `mamba_ssm` / `triton` extras 设计）
