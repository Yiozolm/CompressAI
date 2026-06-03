# Channel-Slice Latent Codec 跨模型抽象重设计

**记录日期：** 2026-05-09
**最近修订：** 2026-05-11（FTIC 重新评估：基于 `candidate/FTIC/` 原作者代码复核，纠正"FTIC 用 3-参数 Gaussian / 完全不走 LatentCodec / 单独 Family 8 / 不参与 codec refactor"四处错判，详见 §2.7、§3.1、§7）
**触发：** 在 `pr-tcm-cca` 分支整合 TCM/CCA 时发现：
- TCM 不能直接用 `SliceEntropyCompressionModel`，因为它的 entropy head widths 写死了 STF 的 5-conv `(224, 176, 128, 64)`，TCM 用的是 3-conv `(224, 128)`；TCM 只能 bypass 基类、自己 wire 一遍 `ChannelSliceLatentCodec`
- CCA 完全没用 codec / 基类，因为它要变长 slice (`slice_proportions`) + aux entropy 用 skip-most-recent support
- 后续 PR 还会迁入 ELIC 的 group 切分 + checkerboard、MLIC++ 的多参考 spatial+channel 上下文等更多变种

按现状一个模型一个模型补丁式接入，会把 codec 变成 swiss army knife。本文档梳理 channel-slice 家族的真实变异维度，提出抽象层重设计方向。

---

## 1. 调研对象（17 个 entropy head；fork `script` 仍是 17 个 codec 类——CMIC 与 FTIC 不引入新 codec 类：CMIC 直接复用 upstream `ChannelGroupsLatentCodec` + `CheckerboardLatentCodec`（与 GLIC 同构），FTIC 在 fork `script` 上以 `TCAEntropyModel` + `GsnConditionalLocScaleShift` 直接调用方式落地，但**结构上是 Family 1 channel-slice AR 的 transformer-context 变体**，可以容器化为 `ChannelGroupsLatentCodec` + 适配 leaf——详见 §2.7、§3.1）

| 来源模型 | 论文/年代 | 文件 | 状态 |
|---|---|---|---|
| STF (`SymmetricalTransFormer`) | Zhu CVPR 2022 | `compressai/models/stf.py` | pr-stf-wacnn 已合入 |
| WACNN | Zhu CVPR 2022 | `compressai/models/stf.py` | pr-stf-wacnn 已合入 |
| TCM | Liu CVPR 2023 | `compressai/models/tcm.py` | pr-tcm-cca 待合入 |
| CCA-main (`CCAModel`) | Han NeurIPS 2024 | `compressai/models/cca.py` | pr-tcm-cca 待合入 |
| CCA-aux (`_CCAAuxEntropyModel`) | Han NeurIPS 2024 | `compressai/models/cca.py` 内部类 | pr-tcm-cca 待合入 |
| ELIC (`Elic2022Official` / `Elic2022Chandelier`) | He CVPR 2022 | `compressai/models/sensetime.py` | 仓库已有 |
| **GLIC** | Chen CVPR 2026 | `compressai/models/glic.py` | ✅ 已迁入 compressai（PR #4，2026-06-03；与 ELIC 同构使用 upstream `ChannelGroupsLatentCodec`，无新 codec 类）|
| **CMIC** | Chen CVPR 2024 | `compressai/models/cmic.py` | ✅ 已迁入 compressai（PR #5，与 MambaIC 合并）；与 ELIC/GLIC 同构使用 upstream `ChannelGroupsLatentCodec` + `CheckerboardLatentCodec`，**无新 codec 类**；novelty 在 content-aware Mamba context blocks |
| MLIC++ | Jiang TIP 2024 | `compressai/models/mlicpp.py` | fork `script` 已迁入，未上游 |
| **DCAE** | Wu 2024 | `compressai/models/dcae.py` | fork `script` 已迁入，未上游 |
| **SAAF** | Ma CVPR 2026 | `compressai/models/saaf.py` | fork `script` 已迁入（与 DCAE 共享 `DictionaryEntropyCompressionModel` 基类）|
| **HPCM** | He 2024 | `compressai/models/hpcm.py` | fork `script` 已迁入，已有专属 latent codec |
| **MambaIC** | Zeng CVPR 2025 | `compressai/models/mambaic.py` + `latent_codecs/mambaic.py` | ✅ 已迁入 compressai（PR #5，与 CMIC 合并），专属 codec（评估后保留 dedicated）|
| **MambaVC** | Qin 2024 | `compressai/models/mambavc.py` | fork `script` 已迁入，**已用 `SliceEntropyCompressionModel`** ✅ |
| **WeConvene** | Wang ECCV 2024 | `compressai/models/weconvene.py` + `latent_codecs/weconvene.py` + `weconvene_support.py` | fork `script` 已迁入，专属 codec |
| **TinyLIC / ShiftLIC large** | Lu arXiv 2022 / Bao TCSVT 2025 | `compressai/models/{tinylic,shiftlic}.py` + `latent_codecs/multistage_checkerboard.py` | fork `script` 已迁入，**两模型共享同一 codec** |
| **FTIC** | Li ICLR 2024 | `compressai/models/ftic.py` | fork `script` 已迁入，**无新 codec 类**——直接 `TCAEntropyModel` + `GsnConditionalLocScaleShift`；channel-slice causality 通过 `MaskedSliceChannelAttention` 内置 mask + 教师强制式输入右移实现，**结构上等价于 Family 1 channel-slice AR**（见 §2.7、§3.1） |
| Entroformer | Qian ICLR 2022 | `compressai/models/damo.py` + `latent_codecs/transformer_ar.py` | fork `script` 已迁入，bitstream `NotImplementedError` |
| RefBasedAR (qian2021-ref) | Qian ICLR 2021 | `compressai/models/damo.py` + `latent_codecs/ref_autoregressive.py` | fork `script` 已迁入，bitstream `NotImplementedError` |

**fork `script` 现存 17 个 codec 类，分布在 7 个功能 family + 4 个 wrapper/legacy 桶**——见 §3.1 和 §3.4 完整分类。CMIC / GLIC / FTIC 三个模型不在此 17 类中各占一席：CMIC 与 GLIC 直接复用 upstream `ChannelGroupsLatentCodec`，FTIC 走自带的 `TCAEntropyModel` + `GsnConditionalLocScaleShift`，三者均**未引入新 codec 类**。其中 FTIC 的 entropy 路径**结构上属于 Family 1 channel-slice AR 的 transformer-context 变体**（不是独立 family），原作者代码不走 LatentCodec 接口仅是实现选择，详见 §2.7。

现有抽象层：
- `compressai/latent_codecs/channel_slice.py::ChannelSliceLatentCodec`
- `compressai/models/_bases/slice_entropy.py::SliceEntropyCompressionModel`

fork `script` 上还存在：
- `compressai/models/_bases/dictionary_entropy.py::DictionaryEntropyCompressionModel`（DCAE/SAAF 共享，与 `SliceEntropyCompressionModel` 90% 重合）
- `compressai/latent_codecs/hpcm.py::HierarchicalProgressiveLatentCodec`（HPCM 专属，800+ 行 mask 逻辑）
- `latent_codecs/{mambaic,weconvene,multistage_checkerboard,transformer_ar,ref_autoregressive}.py`（5 个其他专属 codec）

ELIC / GLIC / CMIC 用 upstream `ChannelGroupsLatentCodec`，MLIC++ 用 `MLICPlusPlusLatentCodec`，MambaIC 用 `MambaICLatentCodec`。

---

## 2. 变异维度对比

本节按 family 分组对比 7 个 family 共 15 个 entropy head 的 10 个变异维度（外加一行「独特点」捕获 family 特有的设计细节）。**FTIC 原本被列为 Family 8，2026-05-11 修订后归入 Family 1 的 transformer-context 变体**——见 §2.7。Family 编号、命名与结构族叙事见 §3.1；17 个 codec 类完整分类见 §3.4。Family 内不存在的轴标 `n/a`。

### 2.1 Family 1：纯 channel-slice，1-pass

| 维度 | STF/WACNN | TCM | CCA-main | CCA-aux | DCAE | SAAF | MambaVC |
|---|---|---|---|---|---|---|---|
| Slice 切法 | `chunk(K)` 等大 | `chunk(K)` 等大 | `split(slice_sizes)` 变长 | 同 main | `chunk(K)` 等大 | `chunk(K)` 等大（默认 K=5）| `chunk(K)` 等大（已用 `SliceEntropyCompressionModel`）|
| Support 截断 | `max_support_slices` clamp | 同 | use-all-prior | **skip-most-recent**（`y_hat[: max(i-1, 0)]`） | `max_support_slices` clamp | `max_support_slices` clamp（默认 5）| `max_support_slices` clamp |
| mean_support transform | None (Identity) | `SWAtten` per slice | `_NAFTransform` per slice | `_NAFTransform` per slice | **`MutiScaleDictionaryCrossAttentionGLU` 拼接 dictionary_info（与 scale 共享）** | **`MutiScaleDictionaryCrossAttentionGLU`（与 DCAE 共享 pattern；与 scale 共享）** | 走基类默认 |
| scale_support transform | None | `SWAtten` per slice | `_NAFTransform` per slice | `_NAFTransform` per slice | **同 mean（共用 cross-attention 输出）** | **同 mean（共用 cross-attention 输出）** | 走基类默认 |
| CC head 架构 | 5-conv `(in)→224→176→128→64→slice` | 3-conv `(in)→224→128→slice` | 3-conv `(in)→hidden(224)→128→slice` | 同 main | 3-conv `(in)→224→128→slice`（同 TCM）| 走基类 `make_entropy_transform` 默认 widths | 走基类默认 |
| LRP | yes, `0.5*tanh` | yes, `0.5*tanh` | yes, `0.5*tanh` | yes, **只对前 K-2 slices** | yes, `0.5*tanh`（全 slice）| yes（走 `DictionaryEntropyCompressionModel` 默认）| yes |
| 分布 | Gaussian | Gaussian | Gaussian | Gaussian | Gaussian | Gaussian | Gaussian |
| Intra-slice 空间上下文 | none | none | none | none | none | none | none |
| Compress/decompress 循环 | 1-pass per slice | 1-pass per slice | 无 (research-only) | n/a（嵌于 main 内）| 1-pass per slice | 1-pass per slice | 1-pass per slice |
| Hyperprior 归属 | 模型拥有 `h_a/h_*_s/entropy_bottleneck`，codec 收 `(means, scales)` | 同 STF | 同 STF | 共享 main 的 hyper | 模型拥有 `h_a/h_z_s1/h_z_s2/entropy_bottleneck`（与 STF 同构）| 模型拥有（与 DCAE/STF 同构）| 模型拥有 |
| 独特点 | — | SWAtten 是 TCM 的 windowed attention support 变换 | 变长 slice + NAF support；`em_hidden=224`、`em_layers` 可配 | 与 main 共享 entropy infrastructure；前 K-2 slice 用 LRP | support 多塞 dictionary cross-attention；fork `script` 上单设 `DictionaryEntropyCompressionModel` 基类 | g_a/g_s 用 sparse spatial attention + adaptive frequency blocks（不在 entropy loop）；entropy 路径与 DCAE 同构（共享 `DictionaryEntropyCompressionModel` + `dt` Parameter + `dt_cross_attention`）| fork `script` 上**已经走通** `SliceEntropyCompressionModel`，是 C+A 抽象的隐性受益者 |

### 2.2 Family 2：channel-slice + intra-slice 空间上下文，2-pass

| 维度 | ELIC | GLIC | CMIC | MLIC++ | MambaIC |
|---|---|---|---|---|---|
| Slice 切法 | `split(groups)` 不等大 list | `split([16, 16, 32, 64, M-128])` 不等大 list | `split([M//4]*4)` 等大 4 group（默认）| `chunk(slice_num)` 等大 | `chunk(num_slices)` 等大 |
| Support 截断 | use-all-prior | use-all-prior | use-all-prior | use-all-prior | use-all-prior |
| mean_support transform | rolled into joint head | rolled into `param_aggregation` joint head | rolled into `param_aggregation` joint head（与 GLIC 同构）| `LinearGlobalInterContext` + `ChannelContext` per slice | 第一遍 SWAtten（分离）；第二遍由 `context_vss` 统一供给 |
| scale_support transform | rolled into joint head | 同 mean | 同 mean | 同 mean | 同 mean（第一遍分离 SWAtten；第二遍合一）|
| CC head 架构 | 3-conv joint, 输出 `2*group_ch`（mean+scale） | `channel_context` 3-step `sequential_channel_ramp(GatedTransformCNN)` 输出 `2*groups[k]`；`param_aggregation` MLP 输出 `2*groups[k]` | `CMICChannelContextBlock` per group + `param_aggregation` MLP 输出 `2*group_ch`（与 GLIC 同构）| `EntropyParameters` MLP，**anchor / nonanchor 两套独立 head** | 第一遍走 STF/TCM 的 `make_entropy_transform`；第二遍 `chunk(2)` 后各 SWAtten 细化 |
| LRP | **no** | **no**（与 ELIC 一致）| **no**（与 ELIC/GLIC 一致）| yes, **anchor / nonanchor 两套** LRP | yes |
| 分布 | Gaussian | Gaussian | Gaussian | Gaussian | Gaussian |
| Intra-slice 空间上下文 | **CheckerboardMaskedConv2d 两遍** | **CheckerboardMaskedConv2d 两遍**（`spatial_context[k]`，与 ELIC 同构）| **CheckerboardMaskedConv2d 两遍** via `CMICSpatialContextBlock`（与 ELIC/GLIC 同构）| **多参考**：LocalContext (windowed attn) + LinearGlobalIntraContext + checkerboard 切分 | **CheckerboardMaskedConv2d** + **VSS (Mamba) state-space context block** per slice |
| Compress/decompress 循环 | per-group → `CheckerboardLatentCodec` 内 2-pass | per-group → `CheckerboardLatentCodec` 内 2-pass（与 ELIC 同构）| per-group → `CheckerboardLatentCodec` 内 2-pass（与 ELIC/GLIC 同构）| per-slice 2-pass (anchor → nonanchor) | per-slice 2-pass (anchor → nonanchor checkerboard) |
| Hyperprior 归属 | `HyperpriorLatentCodec` 包 `ChannelGroupsLatentCodec` | `HyperpriorLatentCodec` 包 `ChannelGroupsLatentCodec`（与 ELIC 同构，已是终态；z 用 `quantizer="ste"`）| `HyperpriorLatentCodec` 包 `ChannelGroupsLatentCodec`（与 ELIC/GLIC 同构，已是终态；z 用 `quantizer="ste"`）| **codec 自带** `h_a/h_s/entropy_bottleneck/gaussian_conditional` | 模型拥有（与 STF/TCM/CCA 同构）|
| 独特点 | mean+scale 联合预测；不等大 group 列表；走 ELIC pattern 的 codec-owned hyper | g_a/g_s 用 GFA (Graph Feature Aggregation) branches + `OLP` 正交 loss；entropy 路径与 ELIC 同构使用 upstream codec primitives（**无新 codec 类**）| g_a/g_s 用 **content-aware Mamba** (`_ContentAwareMamba` SSM) blocks + `OLP` 正交 loss；entropy 路径完全沿 ELIC/GLIC pattern（**无新 codec 类**）| 多参考 spatial+channel 上下文；anchor / nonanchor 各一套独立 head + LRP | 第二遍以 SSM (Mamba VSS) 替代 ELIC 的 MLP；第二遍每 slice 共用 `context_vss[i]` |

### 2.3 Family 3：HPCM（hierarchical spatial AR, multi-resolution）

| 维度 | HPCM |
|---|---|
| Slice 切法 | **不切** —— 全 M 通道一起处理 |
| Support 截断 | n/a（不是 channel-slice，改用 mask AR）|
| mean_support transform | n/a（mean/scale 联合预测，见独特点）|
| scale_support transform | n/a |
| CC head 架构 | spatial prior `(3M→2M)` 联合输出 `(μ, σ)`，3 个 stage 各一个 prior |
| LRP | 无（mask 加权累加替代）|
| 分布 | **GeneralizedGaussianConditional, β=1.5** —— 不是单 Gaussian |
| Intra-slice 空间上下文 | **3 stage hierarchical mask AR**：s1 (H/4, 2 step) → s2 (H/2, 4 step) → s3 (H, 6 step)，共 12 pass over multi-resolution masks |
| Compress/decompress 循环 | 12-pass total；bitstream 在 fork `script` 上仍 `raise NotImplementedError`（仅 forward 通了）|
| Hyperprior 归属 | **codec 自带** `means_hyper`/`scales_hyper` 学习参数 + `GeneralizedGaussianConditional` |
| 独特点 | `adaptor_s{1,2,3}` (1×1 conv ladder) + `context_net` + 可选 `attn_s{1,2,3}` (windowed cross-attention)；`adaptive_params` (10 × 学习参数) per-step 调制 |

### 2.4 Family 4：WeConvene（wavelet 两分支）

| 维度 | WeConvene |
|---|---|
| Slice 切法 | 先 DWT 切成 low (M ch) + high (3M ch) 两支，每支再 `chunk(num_slices)` |
| Support 截断 | use-all-prior（per 分支内）|
| mean_support transform | 分离（`cc_mean_transforms_low/high`），STF 风格 |
| scale_support transform | 分离（`cc_scale_transforms_low/high`），STF 风格 |
| CC head 架构 | STF 风格 5-conv（每分支一套）|
| LRP | yes（两支各一套；high LRP 还要拼 `y_low_hat`）|
| 分布 | 单 Gaussian × 2（two `GaussianConditional` 实例）|
| Intra-slice 空间上下文 | 无（核心创新在 wavelet 预变换，不在 entropy loop）|
| Compress/decompress 循环 | **2× 顺序 1-pass loops**——low 分支全解码，再 high 分支条件于 `y_low_hat` 解码 |
| Hyperprior 归属 | 模型拥有 |
| 独特点 | codec 内置 `DWT2D`/`IDWT2D`；high 分支 LRP 拼接 `y_low_hat` |

### 2.5 Family 5：TinyLIC / ShiftLIC large（multistage 非均匀 checkerboard）

| 维度 | TinyLIC | ShiftLIC large |
|---|---|---|
| Slice 切法 | `split(slice_sizes)` **固定 4 slice**，cosine `gamma_func` schedule，M=320 大约 `[16, 47, 80, 177]` | `split(slice_sizes)` **固定 4 slice**，linear schedule |
| Support 截断 | use-all-prior（per slice 内）| 同 |
| mean_support transform | n/a（联合预测）| n/a |
| scale_support transform | n/a（联合预测）| n/a |
| CC head 架构 | **联合**——`entropy_parameters_{1..4}` MLP 输出 `2*slice_size` 后 `chunk(2)` | 同 TinyLIC |
| LRP | n/a（multistage mask 替代逐 slice LRP）| 同 |
| 分布 | 单 Gaussian | 单 Gaussian |
| Intra-slice 空间上下文 | 5× `MultistageMaskedConv2d`（手工 mask A/B/C/B/B，kernel 3,5,5,5,5）| 同 |
| Compress/decompress 循环 | per-slice **可变 sub-stage** 内嵌固定 4-slice 外层循环：slice 0 = **4 sub-stage**（2×2 checkerboard 四象限，masks A/B/C 顺序）；slice 1, 2 = 各 **2 sub-stage**（checkerboard halves）；slice 3 = **1 sub-stage**（无空间上下文，仅 hyper）| 同 TinyLIC |
| Hyperprior 归属 | 模型拥有，codec 收 `params = h_s(z_hat)`（2*M ch）| 同 |
| 独特点 | 手工 4+2+2+1 sub-stage schedule + quadrant-aware mask；`cc_transforms` 工厂参数化让两模型注入不同内部 block，无需 fork codec | 与 TinyLIC **共享同一 codec 类**（`MultistageCheckerboardLatentCodec`），仅 schedule 函数不同 |

### 2.6 Family 6 / 7：Entroformer / RefBasedAR（无切片全分辨率 AR）

| 维度 | Entroformer (Family 6) | RefBasedAR (Family 7) |
|---|---|---|
| Slice 切法 | **不切** | **不切** |
| Support 截断 | n/a | n/a |
| mean_support transform | n/a（联合预测）| n/a（联合预测）|
| scale_support transform | n/a（联合预测）| n/a（联合预测）|
| CC head 架构 | 联合 `param_net` MLP，输出 `num_parameter * M` 通道 | 三 cascade 各自 joint head（logit_pi + mean + log-σ）|
| LRP | n/a | n/a |
| 分布 | 单 Gaussian (`num_parameter=2`) | **K=3 GMM** (`GaussianMixtureConditional`) |
| Intra-slice 空间上下文 | 因果 Transformer (`y_ar = TransDecoder`) over 全 latent map | 5×5 mask-A `MaskedConv2d` (local) + `_SearchTransfer` (cosine 全局参考搜索) + `_Conv2dUnfold` |
| Compress/decompress 循环 | 单 forward pass；compress `NotImplementedError` | 3 cascade head 单 pass；compress `NotImplementedError` |
| Hyperprior 归属 | **codec 自带** `y_hyper_encode/decode` + `LearnedGaussianBottleneck` | **codec 自带** `LearnedGaussianBottleneck` + `GaussianMixtureConditional` |
| 独特点 | 全分辨率 Transformer-AR；hyperprior 移入 codec | log-sum-exp soft-attention 混合参考补丁；K=3 GMM；`mask_unfold` 不可训练 buffer |

### 2.7 Family 1 变体：FTIC（masked-channel transformer 作为 channel context）

> **2026-05-11 修订**：基于 `candidate/FTIC/models/{flic,tca,entropy_models}.py` 原作者代码复核，修正三处错判：
> 1. **不是 3-参数 Gaussian**——`GsnConditionalLocScaleShift` 的 `forward(inputs, scales, means)` 只用 2 参数，"shift" 指的是 `inputs - round(mean)` + `mean - round(mean)` 这套 **integer / fractional mean 解耦** 的 indexed-CDF 优化（Charm/Balle 系一直在用），分布本身仍是 2-参数 Gaussian、跟现仓 `GaussianConditional` 数学等价
> 2. **第三路输出 `lrp` 是标准 LRP**——按 `flic.py:455-456` 走 `0.5 * tanh(lrp)` 加在 `y_hat` 上，跟 MLIC/STF/CCA-main 同款，不是分布参数
> 3. **结构上是 channel-slice AR**——`tca.py:164` 的 `torch.cat((start_token, y[:, :-C//slices]), dim=1)` 把输入整体右移一个 slice、左侧补 hyper 派生的 start_token；再叠 `MaskedSliceChannelAttention` 的 head-内 causal mask（`tca.py:111-116`），slice i 的 (means/scales/lrp) 严格只依赖 slice 0..i-1 的 y_hat，与 `ChannelGroupsLatentCodec` 的 channel-AR 语义一致；compress/decompress（`flic.py:496-516` / `flic.py:556-577`）也已经写成 K=5 次 slice 循环。原作者用 grouped conv (`groups=slices`) 把 K 路 head 折叠成单个 conv，是实现技巧而非结构差异。

| 维度 | FTIC |
|---|---|
| Slice 切法 | `chunk(num_slices)` 等大（默认 K=5；M 必须可整除）|
| Support 截断 | use-all-prior（mask + 输入右移共同保证：slice i 看 0..i-1）|
| mean_support transform | n/a（mean / scale / LRP 由单个 `TCAEntropyModel` 联合产出）|
| scale_support transform | n/a |
| CC head 架构 | `TCAEntropyModel` = `TCA` (depth × `TCABlock`：`MaskedSliceChannelAttention` + `SliceGroupedMLP` + `ConvPositionalEncoding`) → `entropy_parameters_net`（**3 个 grouped conv 层 `groups=slices`**），输出 `(means, scales, lrp)` over 整个 y |
| LRP | yes（全 K slice，与 Family 1 同款 `0.5 * tanh` 后加在 `y_hat`）|
| 分布 | **单 Gaussian**——`GsnConditionalLocScaleShift` = indexed CDF 表 + integer mean shift（leaf-codec 优化），数学上等价于 `GaussianConditional` |
| Intra-slice 空间上下文 | n/a（intra-slice 空间结构由 TCA Swin-style window attention 给出，但**不分 anchor/nonanchor**——这条与 Family 1 完全一致）|
| Compress/decompress 循环 | **forward 1-pass**（TCA 借 channel-causal mask 一次产出全部 K slice 的参数；ground-truth y 可见）；**compress K-pass per slice + 1 次 TCA**（同样因 y 可见）；**decompress K-pass**（每 slice 解出后回写 `y_hat_coded`，再重跑 TCA 得到下个 slice 的参数）—— K-pass decompress 与 Family 1 完全一致 |
| Hyperprior 归属 | 模型拥有 `h_a/h_mean_s/h_scale_s/entropy_bottleneck`（与 STF 双 h_s 同构）|
| 独特点 | **唯一真正特异点是 channel-context 模块**：用 transformer + masked attention 替代 Family 1 的 per-slice MLP/SWAtten/NAF。其余维度（chunk(K) 切片、use-all-prior、joint mean/scale/lrp head、Gaussian leaf、K-pass decompress、双 h_s 模型主拥有）全部落在 Family 1 范围内。可容器化为 `ChannelGroupsLatentCodec(channel_context={...TCA adapter...}, latent_codec={LRPGaussianLatentCodec ×K})`，详见 §7、§8 修订段 |

---

## 3. 关键发现

### 3.1 实际是七个结构族 + 一个 Family 1 transformer-context 变体（FTIC）

> **2026-05-11 修订**：原稿把 FTIC 列为独立的 Family 8 (transformer-AR over channel slices)，复核后 demote 为 Family 1 的 transformer-context 变体——理由见 §2.7 三处错判修订。Family 总数从 8 调到 7，FTIC 列在 Family 1 末尾。

- **Family 1（纯 channel-slice，1-pass）**：STF, WACNN, TCM, **DCAE**, CCA-main, CCA-aux, **MambaVC**，**FTIC**（transformer-context 变体）
  - 共享同一个 forward loop：split → 逐 slice 算 (mu, scale) → Gaussian → quantize → 可选 LRP
  - 真正差异收敛在 5 个轴：slice 切法、support 截断、support 变换、CC head 宽度、LRP 范围
  - DCAE 是 cousin（support 多塞 dictionary cross-attention），fork `script` 上为它和 SAAF 单设了 `DictionaryEntropyCompressionModel` 基类
  - **MambaVC 已经在用 `SliceEntropyCompressionModel`** ✅
  - **FTIC**：把 per-slice MLP/SWAtten/NAF channel-context 换成"masked-channel transformer (TCA)"——TCA 借 channel-causal mask + 输入右移在 1 个 forward 内同时产出全部 K slice 的 (means/scales/lrp)；leaf 用 `GsnConditionalLocScaleShift`（indexed Gaussian + integer mean shift trick，等价于 `GaussianConditional`）。原作者代码不走 LatentCodec 接口仅是实现选择，结构上完全可容器化为 `ChannelGroupsLatentCodec(channel_context={...TCA adapter...}, latent_codec={LRPGaussianLatentCodec ×K})`
- **Family 2（channel-slice + intra-slice 空间上下文，2-pass）**：ELIC, GLIC, **CMIC**, MLIC++, **MambaIC**
  - 每 slice/group 内部 anchor/nonanchor 两遍 AC（checkerboard masked-conv）
  - ELIC/GLIC/CMIC/MLIC++ 用 mean+scale 联合预测；MambaIC 第一遍分离、第二遍联合
  - 上下文模块各异（ELIC: SWAtten/MLP，GLIC: GFA graph，CMIC: content-aware Mamba，MLIC++: 多参考，MambaIC: VSS Mamba SSM）
  - ELIC/GLIC/CMIC 直接复用 upstream `ChannelGroupsLatentCodec`，**无新 codec 类**；MLIC++ 与 MambaIC 各有 dedicated codec 类
- **Family 3（hierarchical spatial AR, multi-resolution）**：HPCM
  - 不切片；3 个分辨率阶段 × mask gather/scatter；GeneralizedGaussian(β=1.5)
- **Family 4（wavelet 两分支）**：WeConvene
  - DWT 后两支顺序 1-pass loop；high 分支条件于 `y_low_hat`
- **Family 5（multistage 非均匀 checkerboard）**：TinyLIC, ShiftLIC large（**两模型共享同一 codec**）
  - schedule 切片 + 4+2+2+1 sub-stage + 联合 mean+scale + 5 个手工 mask
- **Family 6（全分辨率 Transformer AR）**：Entroformer
  - 不切片；codec 自带 hyperprior；bitstream NotImpl
- **Family 7（参考搜索 GMM）**：RefBasedAR
  - 不切片；K=3 GMM；全局 cosine 参考搜索；codec 自带 hyperprior；bitstream NotImpl

> **~~Family 8 (transformer-AR over channel slices)~~ — 已撤销**：原列 FTIC 为独立 family，2026-05-11 复核后归入 Family 1 transformer-context 变体（见上 Family 1 段末尾 + §2.7 修订）。

### 3.2 七个 family 的 forward loop 结构性不同，不能强行合并

- 1-pass (F1) vs 2-pass (F2) vs 12-pass over multi-resolution (F3) vs 2×顺序 1-pass (F4) vs 4+2+2+1 sub-stage (F5) vs Transformer 全分辨率 (F6) vs cascade GMM (F7) ——是真实的结构差异
- mean+scale **联合 vs 分离**影响 `cc_*_transforms` 的 ModuleList 结构和 state_dict key
- F6/F7 把 hyperprior 和 / 或 entropy head 完全 collapse 到单个 nn.Module，跟 F1-F5 的"codec 收 (means, scales)"模型主拥有的设计哲学不同
- F4-F7 跟 F1 的差异是**不可参数化的架构选择**（wavelet 切分 / 联合-vs-分离 / 切片-vs-不切片 / 单 Gaussian vs GMM）
- **F1 内部 FTIC 变体**只是 channel-context 模块换成 transformer + 输入右移 + masked attention，forward loop 形态、leaf 类型、K-pass decompress 都仍是 F1 标准

**fork `script` 上现状已经认识到这点**：17 个 codec 类，7 个走 LatentCodec 接口的 family + FTIC 当前以 `nn.Module` + `GsnConditionalLocScaleShift` 直接调用形式落地（容器化为 follow-up，见 §7、§8），**`per-model codec class` 是 maintainer 的明确方向**。

### 3.3 现状 Family 1 内部的真实痛点

- `SliceEntropyCompressionModel._init_slice_entropy` 把 `widths = (224, 176, 128, 64)` 写死，TCM 不能用
- `ChannelSliceLatentCodec.forward` 用 `y.chunk(num_slices)`，CCA-main 不能用（要变长）
- `ChannelSliceLatentCodec` 没有 skip-most-recent support 模式，CCA-aux 不能用
- `ChannelSliceLatentCodec` 没有「跳过 LRP」开关，无 LRP 的变体不能用
- `_bases/dictionary_entropy.py` 与 `_bases/slice_entropy.py` 90% 重合——要么吸收为可选 hook（`support_builder`），要么承认其作为兄弟基类

修前 4 处就让 Family 1 的 STF/WACNN/TCM/CCA-main/CCA-aux/MambaVC 共享同一个 codec + 基类。第 5 处（DCAE）单独决策。

**Family 4 (WeConvene) 和 2-MambaIC 都已经在 fork `script` 调用 `_bases` 的 helper**（`slice_support_channels` / `lrp_support_channels` / `make_entropy_transform`），证明 widths 参数化已经不只是 Family 1 内部需求——C+A 设计**广泛地服务于 5 个 codec 类**。

### 3.4 fork `script` 现存 17 个 codec 类完整分类

> **2026-05-10 修订**：本节描述 fork `script` 上 17 个 codec 类的快照。Family 2 上游迁入路线图（[`plan/exec-plans/active/family2-roadmap.md`](../exec-plans/active/family2-roadmap.md)）的 PR-1 决定**部分容器化 MLIC++**：把 `MLICPlusPlusLatentCodec` monolith 拆为 (a) 外层用 upstream `ChannelGroupsLatentCodec`，(b) 内层新增 sibling leaf `MultiContextCheckerboardLatentCodec`（与 `CheckerboardLatentCodec` 同位）。下表 MLIC++ 行同时反映原有状态与目标状态；MambaIC 是否同样改写留 PR-3 评估。

| Family | Codec 类 | 用于 | 循环 | 分布 | 关键特征 |
|---|---|---|---|---|---|
| 1 | `ChannelSliceLatentCodec` | STF, WACNN, TCM, DCAE, **SAAF**, CCA-main, CCA-aux, MambaVC | 1-pass per slice | Gaussian | 分离 mean/scale，可选 LRP |
| 2 | `ChannelGroupsLatentCodec` | ELIC, **GLIC**, **CMIC**（PR-1 后再加 MLIC++）| per-group → checkerboard 2-pass | Gaussian | mean+scale 联合（ELIC/GLIC/CMIC）；分离（MLIC++ via 新 leaf）；groups 大小列表 |
| 2 | ~~`MLICPlusPlusLatentCodec`~~（PR-1 删除）→ **`MultiContextCheckerboardLatentCodec`**（PR-1 新增 sibling leaf）| MLIC++（PR-1 后）；MambaIC 候选（PR-3 评估）| per-slice anchor→nonanchor 2-pass | Gaussian | 双 EP head（anchor/nonanchor）+ 可插拔 spatial/channel intra-context + 可选 per-pass LRP；与 `CheckerboardLatentCodec` 同位 sibling |
| 2 | `MambaICLatentCodec` | MambaIC（PR #5 已迁入；**评估后保留 dedicated** —— 分离 mean/scale head + 跨层 slice 装配与 `MultiContextCheckerboardLatentCodec` 单 head 契约冲突）| per-slice anchor→nonanchor 2-pass | Gaussian | 分离 mean/scale + VSS (Mamba SSM) 上下文 |
| 3 | `HierarchicalProgressiveLatentCodec` | HPCM | 12-pass over 3 resolutions | GeneralizedGaussian (β=1.5) | 手工 mask schedule，codec 自带 hyper params |
| 4 | `WeChARMLatentCodec` | WeConvene | 2× 顺序 1-pass loops (low → high) | Gaussian × 2 | codec 内置 DWT/IDWT；high 条件于 low |
| 5 | `MultistageCheckerboardLatentCodec` | TinyLIC, ShiftLIC large | per-slice 可变 sub-stage (4/2/2/1) | Gaussian | 联合 mean+scale；schedule 切片；5 个手工 mask |
| 6 | `TransformerARLatentCodec` | Entroformer | 单 pass forward (compress NotImpl) | Gaussian (`num_p=2`) | codec 自带 hyperprior；`LearnedGaussianBottleneck` |
| 7 | `RefAutoregressiveLatentCodec` | RefBasedAR (qian2021-ref) | 3-cascade 单 pass (compress NotImpl) | **K=3 GMM** | 全局 cosine 参考搜索 + soft-attn 混合 |
| Building block | `HyperLatentCodec` | 内嵌于其他 codec | 1-pass | n/a | wraps `EntropyBottleneck` for `z` |
| Building block | `HyperpriorLatentCodec` | 内嵌（如 ELIC stack） | 1-pass | Gaussian | hyper + 内嵌 codec 组合 |
| Building block | `EntropyBottleneckLatentCodec` | base hyper 模型 | 1-pass | n/a | 薄 `EntropyBottleneck` wrapper |
| Building block | `GaussianConditionalLatentCodec` | base mean-scale 模型 | 1-pass | Gaussian | 薄 `GaussianConditional` wrapper |
| Legacy | `RasterScanLatentCodec` | Minnen2018 风格 joint AR | per-pixel raster scan | Gaussian | mask-conv AR (单 map) |
| Legacy | `CheckerboardLatentCodec` | He2021 / Lu2022 风格 | 2-pass (anchor, nonanchor) | Gaussian | spatial-only checkerboard |
| Gain | `GainHyperLatentCodec` | gain-modulated hyper | 1-pass | n/a | per-channel gain 调制 |
| Gain | `GainHyperpriorLatentCodec` | gain-modulated hyperprior | 1-pass | Gaussian | gain 包 hyperprior |

**总：17 个 codec 类 = 7 功能 family + 4 wrapper/legacy/gain 桶**。FTIC 不在此 17 类中——它在 fork `script` 上以 `TCAEntropyModel` (`nn.Module`) + `GsnConditionalLocScaleShift`（**indexed Gaussian + integer mean shift trick，等价于 2-参数 `GaussianConditional`**）直接调用形式落地，所以既不在表中、也不算"新 codec 类"。**结构上属于 Family 1 channel-slice AR 的 transformer-context 变体**（见 §2.7、§3.1），容器化为 `ChannelGroupsLatentCodec(channel_context={...TCA adapter...}, latent_codec={LRPGaussianLatentCodec ×K})` 是 follow-up（详见 §7、§8）。

---

## 4. 候选方向

| 方向 | 思路 | 评价 |
|---|---|---|
| **A** | 给 `ChannelSliceLatentCodec` 加 4 个可选参数：`slice_sizes`、`lrp_transforms=None`、`support_filter`、`lrp_scale` | 改动小、向后兼容、覆盖 Family 1 全部需求。不解决 Family 2-8（也不应该解决） |
| **B** | 把 codec 内部循环里的「算 (mu, scale)」和「应用 LRP」抽成 Strategy 对象 | 每个 strategy 又是 `nn.Module`，state_dict 多一层路径；既有 ckpt 全部要 rename。**净增复杂度，不减少运行时循环重复**——而真正昂贵的是 per-model context 模块（不是 loop）。**不推荐** |
| **C** | 多姐妹 codec 类（17 个 codec 类 + FTIC 直接走 `nn.Module`，分布在 7 个 family + 4 个 wrapper）。文档里说清楚每个对应哪个家族 | 这就是 fork `script` 现状。Family 2-7 维持独立，Family 1 仍要做 A 才解决问题（FTIC 作为 Family 1 transformer-context 变体亦同） |
| **D** | 只改基类（widths、support_transform_factory、use_lrp、slice_sizes），不动 codec | 让 TCM 不再 bypass 基类，但 codec 没加 `slice_sizes` 的话 CCA-main 还是用不了 |
| **E** | 在 `ChannelSliceLatentCodec` 加可选 `support_builder: Callable[...]` callable，把 DCAE 的 `dictionary_info` 拼接接进来 | 净删 ~330 LoC（删 `_bases/dictionary_entropy.py`），但引入比 `support_filter` 更宽的 callable hook。需评估是否越过「避免 swiss army knife」红线 |
| **F** | 把 Family 2 三家（ELIC/MLIC++/MambaIC）合并成一个带 `intra_slice_context_factory` 参数的类 | 但联合 vs 分离 mean/scale 让 state_dict 不兼容。**第 4 个 Family 2 dedicated-codec 用户出现前不动**，否则强行 dedupe 风险大。**注（2026-05-09 补充）**：Family 2 实际有 5 个 model 用户（ELIC、GLIC、CMIC、MLIC++、MambaIC），但 ELIC / GLIC / CMIC 都**直接用 upstream `ChannelGroupsLatentCodec`**——没有引入新的 dedicated codec class。dedupe 阈值算的是 dedicated 类数量（仍只有 MLIC++/MambaIC 两个），不是 model 用户数，所以 F 仍不动 |
| **G** ⭐ | **`HyperpriorLatentCodec` 嵌套：把 `h_a` / `h_s` / `entropy_bottleneck` 收进 codec，模型只剩 g_a + g_s + latent_codec**。沿用 ELIC 的现有 pattern，所有 11 个 model-owned hyperprior 模型迁过去 | **codec-owned hyperprior**。代价：state_dict key 路径全变，需要扩展 `HyperpriorLatentCodec` 支持双 h_s。**与 H 联合实施于本 PR** |
| **H** ⭐ | **容器化重写 `ChannelSliceLatentCodec`**：从单体改为 ELIC 风格的容器（`channel_context: Mapping[str, Module]` + `latent_codec: Mapping[str, Module]`），forward 只做 dispatch。引入 `LRPGaussianLatentCodec` leaf + `MeanScaleContextHead` context head | **本 PR 推荐方向（与 G 联合实施）**。pedagogical clarity：模型 → 容器 → leaf 三层，state_dict 路径自解释，跟 ELIC pattern 完全收敛。代价：5 个 Family 1 模型 ckpt 路径全变，删 `_bases/{slice_entropy,dictionary_entropy}.py`。详见 §5 + §10 |

---

## 5. 推荐方向：H + G（本 PR `pr-tcm-cca` 直接实施）

> **✅ 实施完成（2026-05-09）**：本 PR 6 commits `c6d556a..f87c8c8` 已 push 到 `Yiozolm/CompressAI:pr-tcm-cca`，开 PR 待 InterDigital upstream review。执行计划见 [`plan/exec-plans/completed/codec-containerization-h-g-refactor.md`](../exec-plans/completed/codec-containerization-h-g-refactor.md)。本节后续内容保留作为决策与设计记录。

**决策（2026-05-09 更新）**：从「短期 C+A + 长期 H+G」**pivot 到 H+G 直接实施**。本 `pr-tcm-cca` 分支变成 codec 容器化重构 PR，提交给上游 `InterDigitalInc/CompressAI` 时为「下一个 PR」。

### ⚠️ 重大设计修订（2026-05-09，二次审查 upstream codec 后）

对照 `upstream/master` 的 `compressai/latent_codecs/`（11 个 codec 文件，**不是 fork master 的 17 个**）发现以下复用机会，避免重复造轮子：

| 原计划新增/重写 | 修订后 | 理由 |
|---|---|---|
| 重写 `ChannelSliceLatentCodec` 为容器 | **删除** `ChannelSliceLatentCodec`（pr-stf-wacnn 加的）| 跟 upstream `ChannelGroupsLatentCodec` 接口完全一致（`groups: List[int]` = 等大或变长 slice 都支持），是重复轮子 |
| 新增 `_channel_context.py` 含 `MeanScaleContextHead` | **不新增**——降为应用层 helper，放到 `compressai/models/_helpers/channel_context.py` | upstream `ChannelGroupsLatentCodec.channel_context` 字典里就放普通 `nn.Module`，没有 ChannelContextHead 抽象 |
| 新增 `lrp_gaussian.py` (~60 行) | 在 upstream `compressai/latent_codecs/gaussian_conditional.py` **同文件**加 `LRPGaussianLatentCodec(GaussianConditionalLatentCodec)` subclass（~30 行追加，跟基类放一起）| upstream `GaussianConditionalLatentCodec` 已支持 `entropy_parameters` 钩子和 `quantizer="ste"`，LRP 只需 override forward 加后处理；subclass 跟基类同文件更自然，避免文件碎片 |
| 扩展 `HyperLatentCodec` 加 `quantizer="ste"` 处理 CCA z STE | **零改动** | upstream `EntropyBottleneckLatentCodec(quantizer="ste")` **已实现**（`_get_medians()` + `quantize_ste`），CCA z STE 直接用 |
| —— | **新增改动**：扩展 upstream `ChannelGroupsLatentCodec` 加 `max_support_slices: int = -1` + `support_filter: Optional[Callable] = None`（~10 行 diff，向后兼容默认 use-all-prior）| upstream 是 use-all-prior，STF/TCM 要 `max_support_slices` clamp，CCA-aux 要 skip-most-recent |

净 LoC 影响：本 PR scope 净 **~−520 行**（详见 §10.10）；含 DCAE/MambaVC follow-up 累计 ~−825 行。

### 5.1 主方案概览

模型类只剩 `g_a + g_s + latent_codec`，latent_codec 沿 ELIC pattern **完全容器化**——直接用 upstream `HyperpriorLatentCodec` + `ChannelGroupsLatentCodec` 嵌套，每片一个独立的 context module 和 leaf codec，state_dict 路径直接反映模型层级。

```python
class WACNN(SimpleVAECompressionModel):  # 基类只剩通用 forward/compress/decompress
    def __init__(self, N=192, M=320, num_slices=10, max_support_slices=5):
        super().__init__()
        self.g_a = ...
        self.g_s = ...
        self.latent_codec = HyperpriorLatentCodec(           # ← upstream 已有
            h_a=_h_a(M, N),
            h_s=DualHyperSynthesis(_h_mean_s(M, N), _h_scale_s(M, N)),  # ← 新增 25 行 adapter
            latent_codec={
                "z": EntropyBottleneckLatentCodec(EntropyBottleneck(N), quantizer="noise"),  # ← upstream 已有
                "y": ChannelGroupsLatentCodec(                # ← upstream 已有，只加 2 个可选参数
                    groups=[M // num_slices] * num_slices,    # 等大切片 = list of K equal sizes
                    max_support_slices=max_support_slices,    # NEW (default -1)
                    channel_context={                          # 应用层构造，普通 nn.Module
                        f"y{k}": _build_mean_scale_head(...)
                        for k in range(1, num_slices)
                    },
                    latent_codec={
                        f"y{k}": LRPGaussianLatentCodec(...)  # ← 新增 30 行 subclass
                        for k in range(num_slices)
                    },
                ),
            },
        )
```

### 5.2 本 PR scope

> **2026-05-09 实施修正**：原稿把 DCAE / MambaVC 列入本 PR 容器化范围，但实施期间决定**延后**到独立 follow-up PR（fork `script` 已迁入旧 monolithic 版本，重写为容器化版本属独立工作量）。本 PR scope 实际为 5 个模型：STF, WACNN, TCM, CCA-main, CCA-aux。

| 模型 | 处理 |
|---|---|
| **完全容器化（H + G）** | STF, WACNN, TCM, CCA-main, CCA-aux（5 个 Family 1 模型）|
| **延后 follow-up** | DCAE, MambaVC（Family 1，独立 PR 处理容器化迁移）|
| **仅 G**（monolithic codec 包进 `HyperpriorLatentCodec`） | MambaIC（Family 2，本 PR 不动）|
| **已是终态** | ELIC, MLIC++, HPCM, Entroformer, RefBasedAR（5 个已 codec-owned，本 PR 不动）|
| **不变** | WeConvene, TinyLIC, ShiftLIC（Family 4/5，sibling codec 类，不容器化）|

**新增基础设施**（仅 2 个新文件 + 2 处 upstream 改动，~155 行）：
- `compressai/latent_codecs/_hyper_synthesis.py`（**新文件**，25 行）— `DualHyperSynthesis` adapter（双 h_s 模型用）
- `compressai/latent_codecs/_slice_helpers.py`（**新文件**，120 行）— 把 `make_entropy_transform` / `infer_*` 等 helper 从 `_bases/` 搬过来
- `compressai/latent_codecs/gaussian_conditional.py`（**改 upstream 既有文件**，+30 行）— 在 `GaussianConditionalLatentCodec` 之后追加 `LRPGaussianLatentCodec(GaussianConditionalLatentCodec)` subclass
- `compressai/latent_codecs/channel_groups.py`（**改 upstream 既有文件**，+10 行 diff）— 加 `max_support_slices: int = -1` + `support_filter: Optional[Callable] = None` 两个可选参数（向后兼容默认 use-all-prior，ELIC 不受影响）

**应用层 helper**（不进 `latent_codecs/`）：
- `MeanScaleContextHead` 之类的 mean/scale 分离 context module 是 application 层 helper，放在 `compressai/models/_helpers/` 或每个 model 文件本地的工厂函数

**删除**：
- `compressai/latent_codecs/channel_slice.py` — pr-stf-wacnn 加的 `ChannelSliceLatentCodec`，跟 upstream `ChannelGroupsLatentCodec` 重复
- `compressai/models/_bases/slice_entropy.py` — 整个基类删除，helpers 搬到 `_slice_helpers.py`
- `compressai/models/_bases/dictionary_entropy.py` — DCAE/SAAF 共享基类，跟 `slice_entropy.py` 90% 重复，dictionary 职责下放给 application 层的 `DictionaryMeanScaleContextHead`（DCAE/SAAF 上岸时再做）

### 5.3 估算总 LoC：本 PR scope 净 ~−520 行

> **2026-05-09 实施修正**：估算假设 fork `script` 是 base（TCM/CCA 当作既有文件 trim）。**实际 PR diff** 是 +4596 / −655 = 净 **+3941**——因为本 PR 提交给 upstream 的 base 是 `upstream/master`，TCM/CCA 在 upstream 不存在，必须从零写入。详见 §10.10 修正。

详细 per-file 表见 §10.10。本 PR scope（STF/WACNN/TCM/CCA + 基础设施）的 ~−520 行主要来自：
- 删 `compressai/latent_codecs/channel_slice.py`（pr-stf-wacnn 加的，−270 行）
- 删 `compressai/models/_bases/slice_entropy.py`（−260 行）
- 3 个 Family 1 模型 codec 构造瘦身（净 −330 行）
- 抵消：2 个新 codec 文件 +145 行 + upstream `channel_groups.py` +10 行 + upstream `gaussian_conditional.py` +30 行 + 2 个应用层 helper 文件 +120 行 + 3 个转换脚本 rename map +90 行 + 测试 fixture +50 行

**完整 H+G refactor 跨 PR 累计 ~−825 行**（本 PR + 后续 DCAE/MambaVC follow-up）。

### 5.4 主要风险

| 风险 | 严重度 | 缓解 |
|---|---|---|
| 5 个 Family 1 模型 state_dict key 路径全变（DCAE/MambaVC 延后到 follow-up）| 高 | 每个 `convert_*_checkpoint.py` 加 `_DIRECTION_GH_RENAMES` rename map（详见 §10.9）；上游 ckpt round-trip 仍可工作 |
| STF/WACNN（PR #354 已合 / 待合）需要回炉重写 | 中 | 用户已与 maintainer 沟通；本 refactor PR 在其后提交，同时**删除** PR #354 加入的 `ChannelSliceLatentCodec`（重复轮子）|
| `HyperpriorLatentCodec` 接口要扩展支持双 h_s | 低 | 用 `DualHyperSynthesis(h_mean_s, h_scale_s)` adapter，~25 行新文件，**零改动** upstream 现有类 |
| `ChannelGroupsLatentCodec` 加 `max_support_slices` + `support_filter` | 低 | 向后兼容（默认值不变 ELIC 行为），~10 行 diff |
| CCA z STE-quantize 需要 codec 支持 | **零** | upstream `EntropyBottleneckLatentCodec(quantizer="ste")` 已支持 |
| `GaussianConditional` 现共享、新设计 per-slice 副本 | 低 | K=10 副本，每模型多 ~几十 KB buffer。可接受，文档说明 |
| 测试 fixture 重新生成 | 中 | 必须做，跟 state_dict 路径变化是同一个工作 |

**详细设计**：见 §10——11 个子节覆盖 API 决策、leaf 清单、context head、factory helper、各模型 wiring sketch、hyperprior 适配、state_dict 路径、基类去留、转换脚本、LoC 估算、dead-end 评估。

---

## 6. 备选方案：C + A（已不采纳，保留作为决策记录）

> **状态**：早期短期方案，曾考虑作为「pr-tcm-cca 内只做小改动 + 长期单独 refactor PR」的折衷。2026-05-09 决定 pivot 到 H+G 直接实施后**已不采纳**。以下保留作为设计取舍的历史记录。

**思路**：保持多姐妹 codec 类（C），对 `ChannelSliceLatentCodec` 做精准扩展（A），不动 hyperprior 归属、不动 state_dict 路径。

### 6.1 `ChannelSliceLatentCodec` 加 4 个可选参数

```python
ChannelSliceLatentCodec(
    cc_mean_transforms,
    cc_scale_transforms,
    lrp_transforms=None,                  # NEW: None → 跳过 LRP
    gaussian_conditional=None,
    mean_support_transforms=None,
    scale_support_transforms=None,
    *,
    slice_sizes: list[int] | None = None,  # NEW: 变长切片，None → chunk(num_slices) 老路径
    num_slices=None,
    max_support_slices=-1,
    support_filter: Callable[[int, list[Tensor]], list[Tensor]] | None = None,  # NEW: 自定义 support 截断
    quantizer="ste",
    lrp_scale=0.5,                         # NEW
)
```

### 6.2 `SliceEntropyCompressionModel._init_slice_entropy` 加 4 个 kwarg

```python
def _init_slice_entropy(
    self, latent_channels, entropy_bottleneck_channels,
    num_slices, max_support_slices,
    *,
    widths=(224, 176, 128, 64),                       # NEW: STF 默认，TCM 传 (224, 128)
    slice_sizes=None,                                 # NEW
    support_transform_factory=None,                   # NEW
    use_lrp=True,                                     # NEW
    mean_support_transforms=None,
    scale_support_transforms=None,
):
```

### 6.3 各模型映射（C+A 方案下）

| 模型 | 调用方式 |
|---|---|
| STF / WACNN | 不动，默认 widths 和切法都对 |
| TCM | `_init_slice_entropy(M, hyper_channels, num_slices, max_support_slices, widths=(224, 128), mean_support_transforms=swatten_list, scale_support_transforms=swatten_list)` |
| CCA-main | `_init_slice_entropy(..., slice_sizes=resolved_sizes, support_transform_factory=NAFTransform)` |
| CCA-aux | `ChannelSliceLatentCodec` 子类（~30 行），`support_filter=lambda i, slices: slices[: max(i-1, 0)]`，前 K-2 slice 启用 LRP |
| ELIC / MLIC++ | 不动 |

### 6.4 估算 LoC（C+A 方案下）

| 文件 | ±LoC |
|---|---|
| `_bases/slice_entropy.py` | −15 |
| `latent_codecs/channel_slice.py` | +25 |
| `models/tcm.py` | −65 |
| `models/cca.py` | −70 |
| 其他 | 0 |
| **总计** | **−125** |

### 6.5 为什么 pivot 离开 C+A

- C+A 净 LoC −125，H+G 净 LoC −580，**4.5x 大但价值更高**：state_dict 路径自解释、跟 ELIC pattern 完全统一、`_bases/` 整个目录可删
- C+A 实施后 pedagogical clarity 没明显改善——仍是「单体类 + 多个 ModuleList 同位」的 mental model
- 既然 maintainer 接受后续重构，**C+A 反而是浪费的中间步骤**——做了之后还要再做 H+G，等于两次 state_dict rename 工作
- H+G 现在做的迁移面比之后做小（pr-tcm-cca 内只动 STF/TCM/CCA/DCAE/MambaVC 5 个模型，未来再做就要叠加更多新模型）

---

## 7. 诚实承认

> **2026-05-11 修订**：原稿把 FTIC 排除在 codec 容器化之外，理由是"3-参数 Gaussian 跟现仓 leaf 不兼容 / 完全 bypass codec 接口"——两条都已撤销。FTIC 实际是 Family 1 transformer-context 变体，可纳入 `ChannelGroupsLatentCodec` 容器；`GsnConditionalLocScaleShift` 升格到 `compressai/entropy_models/`（或合进 `GaussianConditional` 的 indexed-CDF 模式）属于 leaf 优化，不构成阻塞 FTIC 容器化的 hard blocker。

- **不能用一个 codec 类涵盖所有 17 个 entropy head**——fork `script` 现状是 17 个 codec 类、7 个走 LatentCodec 接口的功能 family + 4 个 wrapper/legacy 桶；FTIC 当前以 `nn.Module` + `GsnConditionalLocScaleShift` 直接调用形式存在，结构上属 Family 1 transformer-context 变体（见 §2.7）
- **能容器化的是 Family 1（7 个模型：STF/WACNN/TCM/CCA-main/CCA-aux/DCAE/SAAF/MambaVC + FTIC transformer-context 变体）+ Family 2 的 MambaIC**——这是 H+G 推荐方案的范围；GLIC / CMIC 与 ELIC 一样已经直接用 upstream `ChannelGroupsLatentCodec`，无需进一步容器化
- **Family 2 内部 dedicated codec 类（MLIC++/MambaIC，加上 ELIC/GLIC/CMIC 直接用 upstream）保留**——联合 vs 分离 mean/scale 让 state_dict 不兼容，强行合并代价大于收益（详见 §4 F-option 注）
- **Family 3-7（HPCM/WeConvene/TinyLIC/ShiftLIC/Entroformer/RefBasedAR）保持兄弟 codec 类**——它们的 forward loop 各自有不可参数化的架构差异，硬塞容器只增加间接性、无抽象收益
- **FTIC 容器化是本 codec refactor 的可达目标，但本 PR 不做**——具体 scope 切分：(a) `GsnConditionalLocScaleShift` 升格到 `compressai/entropy_models/`（或扩 `GaussianConditional` 加 indexed-CDF 模式），(b) 写一个 `TCAChannelContext` adapter 把 TCA 输出的全 K-slice 参数按 slice 切给 `ChannelGroupsLatentCodec.channel_context`，(c) 替换 `compressai/models/ftic.py` 的 `forward / compress / decompress` 改成调 `latent_codec`——单独 follow-up PR，roadmap Phase 8
- **fork `script` 上的目录现状证实"per-model codec class"是 maintainer 当前的方向**：17 个 codec 类，5 个新模型（MambaIC/WeConvene/HPCM/TinyLIC+ShiftLIC/MambaVC）都各自 / 共享独立 codec
- **`GaussianConditional` 在容器化后改 per-slice 副本**（K=10 份），每模型多 ~几十 KB buffer。可接受
- **Entroformer / RefBasedAR 至今 `compress`/`decompress` `NotImplementedError`**——bitstream 实现是另一份 follow-up
- **本 PR 的 5 个 Family 1 模型 state_dict key 路径全变**——上游 ckpt round-trip 通过 `convert_*_checkpoint.py` rename map 保持工作（详见 §10.9），但本仓里 5 个模型的训出 ckpt（如有）需要重新生成（DCAE/MambaVC 延后到 follow-up PR，原本数到 6 个）

---

## 8. 不在本文档 scope

- ELIC / MLIC++ / HPCM / WeConvene 等 Family 2-8 内部的进一步抽象（它们各自已有独立 codec 类 / 模块，未来需要时单独评估）
- Family 2 内部 dedicated codec（MLIC++/MambaIC）的进一步合并（Direction F）——等第 4 个 Family 2 dedicated-codec 用户出现（ELIC/GLIC/CMIC 共用 upstream `ChannelGroupsLatentCodec`，不计入 dedupe 阈值）
- Entroformer / RefBasedAR 的 bitstream 实现
- TinyLIC ↔ ShiftLIC large 之间 `make_cc_transform` 接口的进一步抽象
- DCAE 的 `support_builder` 抽象（option E）——本 PR 通过 `DictionaryMeanScaleContextHead` subclass 解决，不引入更宽 callable hook
- HPCM 的 bitstream `compress`/`decompress` 实现
- **FTIC 容器化的 follow-up PR scope**——`GsnConditionalLocScaleShift` 升格到 `compressai/entropy_models/`（或扩 `GaussianConditional` 加 indexed-CDF 模式）+ `TCAChannelContext` adapter + `compressai/models/ftic.py` 改成调 `latent_codec`（roadmap Phase 8，已不再视为 hard blocker——见 §7 修订段）
- 训练脚本侧需不需要适配新的 codec 接口（训练侧后续工作）

---

## 9. 引用

- `compressai/latent_codecs/channel_slice.py` — 现有 codec
- `compressai/latent_codecs/{mambaic,weconvene,weconvene_support,multistage_checkerboard,transformer_ar,ref_autoregressive,hpcm,channel_groups,mlicpp,hyperprior,hyper,checkerboard,gaussian_conditional,entropy_bottleneck}.py` — fork `script` 上其他 codec 类
- `compressai/models/_bases/slice_entropy.py` — 现有基类
- `compressai/models/_bases/dictionary_entropy.py`（fork `script`） — DCAE/SAAF 共享基类
- `compressai/models/{stf,tcm,cca,sensetime,mlicpp,mambaic,mambavc,weconvene,tinylic,shiftlic,damo,hpcm,dcae,glic,cmic,ftic}.py` — 16 个 entropy head 实现（CMIC / GLIC 走 upstream `ChannelGroupsLatentCodec`；FTIC 走自带 `TCAEntropyModel` + `GsnConditionalLocScaleShift`，结构上属 Family 1 transformer-context 变体——见 §2.7、§3.1）
- `candidate/FTIC/models/{flic,tca,entropy_models}.py` — FTIC 原作者代码，2026-05-11 复核基础
- `plan/design-docs/cca-cross-model-extension.md` — CCA 作为跨模型插件的方向（互补文档）
- `plan/product-specs/lic-migration-roadmap.md` — 整体 LIC 模型迁入路线图（FTIC = Phase 8、CMIC = Phase 2）

---

## 10. Direction H + G 联合 refactor 详细设计

> **§5.5 是高层概览**；本节是为后续 refactor PR 准备的实施细节。10 个子节覆盖 API 决策、leaf 清单、context head 设计、ergonomic helper、各模型 wiring sketch、hyperprior 适配、state_dict 路径、基类去留、转换脚本、LoC 估算。

### 10.1 复用 upstream `ChannelGroupsLatentCodec` 作为容器

最终选定 **复用 upstream `ChannelGroupsLatentCodec` + 最小扩展**（不是新建 `ChannelSliceLatentCodec`）：

```python
# upstream compressai/latent_codecs/channel_groups.py 现有签名（不动）：
@register_module("ChannelGroupsLatentCodec")
class ChannelGroupsLatentCodec(LatentCodec):
    def __init__(
        self,
        latent_codec: Mapping[str, LatentCodec],
        channel_context: Mapping[str, nn.Module],
        *,
        groups: List[int],
        max_support_slices: int = -1,                                                  # NEW (~10 行 diff)
        support_filter: Optional[Callable[[int, List[Tensor]], List[Tensor]]] = None,  # NEW
        **kwargs,
    ): ...

    def forward(self, y: Tensor, side_params: Tensor) -> Dict: ...
```

**关键决策**：

| 决策 | 选择 | 理由 |
|---|---|---|
| 复用 vs 新建 | **复用 `ChannelGroupsLatentCodec`** | 接口与设想的 `ChannelSliceLatentCodec` 完全一致，没有合理新建理由 |
| 容器形态 | `Mapping[str, Module]`（`y0..yK-1`）| upstream 既有 |
| forward 签名 | `(y, side_params)` 与 ELIC 同 | upstream 既有；STF/TCM/CCA 双 h_s 用 `DualHyperSynthesis` adapter |
| Slice 切法 | 复用 `groups: List[int]` | upstream 既有；`[M//K]*K` 等大、`[s0..sN]` 变长，list 形式涵盖两者 |
| `support_filter` callable | **新增可选参数** | CCA-aux skip-most-recent 必需，~3 行 API surface，向后兼容 |
| `max_support_slices` clamp | **新增可选参数**，默认 `-1`（=use-all-prior 当前行为）| STF/TCM 必需，向后兼容 ELIC |
| `support_builder` callable | **不引入** | DCAE 的 dictionary 拼接职责下放给应用层 channel_context module 自己（§10.3），避免引入 swiss army knife hook |
| LRP 放哪 | **leaf 内**（`LRPGaussianLatentCodec`）| 自我说明性强；CCA-aux 原计划前 K-2 切片用 LRP leaf、后 2 切片用普通 leaf 更诚实（**实施修正**：published ckpt 含 K 份 LRP 权重，最终全 K slice 都用 `LRPGaussianLatentCodec`）|
| 数据传递 | `_get_ctx_params` 返回的是 Tensor（保持 upstream 接口）；leaf 不需要 `mean_support` 时 LRP 由 channel_context module 内部输出预先 stash 到中间 buffer | 比新设计的 dict 接口更轻量；保留 upstream 兼容性 |

### 10.2 Leaf codec 清单

| Class | 来源 | 构造签名 | forward 签名 | state_dict | Family / 用户 |
|---|---|---|---|---|---|
| `GaussianConditionalLatentCodec` | **upstream 已有** | `(scale_table=None, gaussian_conditional=None, entropy_parameters=None, quantizer="noise"|"ste", chunks=("scales","means"))` | `(y, ctx_params: Tensor)` | `gaussian_conditional.*` + `entropy_parameters.*` | F2 leaf（ELIC checkerboard 内嵌）。原计划 CCA-aux 后 2 slice 用，**实施时改为全 K slice 都用 LRP leaf** |
| **`LRPGaussianLatentCodec`** | **追加在 upstream `gaussian_conditional.py` 末尾**（subclass `GaussianConditionalLatentCodec`，~30 行）| `(lrp_transform, *, lrp_scale=0.5, **gc_kwargs)` —— 透传 `entropy_parameters` / `quantizer` / `chunks` 给基类 | `(y, ctx_params: Tensor)` —— LRP 在基类 forward 后做后处理 | 基类 fields + `lrp_transform.*` | F1 leaf（带 LRP）：STF/WACNN/TCM/CCA-main 大部分 slice |
| `CheckerboardLatentCodec` | **upstream 已有** | 现有 | 现有 | 现有 | F2（ELIC 内嵌）|
| `EntropyBottleneckLatentCodec(quantizer="ste")` | **upstream 已有** | 现有 | `(y)` | `entropy_bottleneck.*` | z 编码（CCA 用 STE，其他用 noise）—— **upstream `EntropyBottleneckLatentCodec.forward` 已支持 `quantizer="ste"` 自动用 `_get_medians()`** |

**为什么 LRP 单独一个 leaf 而不是 `GaussianConditionalLatentCodec` 的可选参数**：

- 不扩展 upstream `GaussianConditionalLatentCodec` 的接口（ELIC 等已用模型零风险）
- leaf 类型自我说明：`latent_codec.y3.lrp_transform.0.weight` 一眼看出哪些 slice 有 LRP
- ~~CCA-aux 混合 leaf 类型（前 K-2 用 `LRPGaussianLatentCodec`、后 2 用 upstream `GaussianConditionalLatentCodec`）比 `use_lrp_until=3` 这种 hidden 整数更诚实~~ → **实施修正**：published ckpt 含 K 份 LRP 权重，strict-load 兼容性要求全部 K slice 都用 LRP leaf；这条 rationale 失效
- 用 subclass 而不是从头新写：~30 行而不是 60 行；继承基类的 `_chunk` / quantizer 处理 / compress / decompress 逻辑，只 override forward / compress / decompress 加 LRP 后处理

### 10.3 应用层 channel_context module

`upstream/master` 的 ELIC 在 `ChannelGroupsLatentCodec.channel_context` 字典里放的是**普通 `nn.Module`**——没有 `ChannelContextHead` 这种 codec primitive 抽象。本 PR 沿用这个约定：

- mean / scale 分离的 head（STF/TCM/CCA）用普通 `nn.Module` 工厂构造，放在 `compressai/models/_helpers/` 或每个 model 文件本地
- 所有 head 满足 `forward(prior_y_hat_concat) -> ch_ctx_params` 的简单签名（跟 ELIC `channel_context` 字典里的 module 一样）
- `ch_ctx_params` 通过 `ChannelGroupsLatentCodec._get_ctx_params` 跟 `side_params` cat 后传给 leaf

具体的工厂函数草稿：

```python
# compressai/models/_helpers/channel_context.py（新建文件，~80 行，应用层 helper）
def build_mean_scale_head(
    slice_ch: int, support_ch: int,
    widths: Tuple[int, ...],
    support_transform_factory: Optional[Callable[[int, int], nn.Module]] = None,
) -> nn.Module:
    """Construct a mean/scale 分离 head producing 2*slice_ch channels.

    Used by STF/WACNN/TCM/CCA-main and CCA-aux. The leaf is then
    GaussianConditionalLatentCodec with chunks=("scales","means")
    or LRPGaussianLatentCodec with the same chunk order.
    """
    ...
```

DCAE 的 `dictionary cross-attention` 也作为应用层 helper（DCAE/SAAF 上岸时再加），不进 `latent_codecs/`。

### 10.4 应用层 factory helper：`build_channel_slice_codec`

K=10 个 `"y0".."y9"` 字典字面量太啰嗦。提供应用层 helper（不是 codec method）：

```python
# compressai/models/_helpers/channel_slice.py（新建文件，~40 行）
def build_channel_slice_codec(
    *, groups: List[int], side_channels: int,
    max_support_slices: int = -1,
    support_filter: Optional[Callable] = None,
    leaf_factory: Callable[[int, int], LatentCodec],
    channel_context_factory: Callable[[int, int, int], nn.Module],
) -> ChannelGroupsLatentCodec:
    """leaf_factory(k, slice_ch_k) → leaf；channel_context_factory(k, slice_ch_k, support_ch_k) → head。
    内部生成 'y0'..'yK-1' 字典传给 ChannelGroupsLatentCodec 构造器。"""
```

模型侧调用：

```python
self.latent_codec = HyperpriorLatentCodec(
    h_a=..., h_s=DualHyperSynthesis(h_mean_s, h_scale_s),
    latent_codec={
        "z": EntropyBottleneckLatentCodec(EntropyBottleneck(N), quantizer="noise"),
        "y": build_channel_slice_codec(
            groups=[M // K] * K, side_channels=2 * M, max_support_slices=5,
            leaf_factory=lambda k, ch: LRPGaussianLatentCodec(
                lrp_transform=make_entropy_transform(lrp_in(M, ch, k, MS), ch, widths=(224, 176, 128, 64))),
            channel_context_factory=lambda k, ch, sup: build_mean_scale_head(
                slice_ch=ch, support_ch=sup, widths=(224, 176, 128, 64)),
        ),
    },
)
```

**Rejected alternatives**：作为 `ChannelGroupsLatentCodec` 的 classmethod（侵入 upstream class API）、`uniform()`（不支持变长 slice）、`nn.ModuleList`（破坏 ELIC 命名约定）。

放在 `compressai/models/_helpers/` 而不是 `latent_codecs/`，因为它是 application-layer ergonomic wrapper，不是 codec primitive。

### 10.5 各模型 wiring sketch

> **2026-05-09 实施修正**（CCA-aux 部分）：以下 sketch 与实际实现有 2 处偏离：
> 1. **mixed leaf 方案被否决**：sketch 写 `if k < num_slices - 2 then LRPGaussianLatentCodec else GaussianConditionalLatentCodec`，但 published upstream checkpoint 包含全部 K 份 LRP 权重——为 strict-load 兼容**所有 K 个 slice 都用 `LRPGaussianLatentCodec`**（最后 2 个 slice 的 LRP 计算结果会被 `support_filter` 过滤掉，浪费一点点算力但不影响 likelihoods）
> 2. **缺 `support_count_fn`**：CCA-aux 的 `support_filter=skip_most_recent` 让选出的 prior 数量跟默认 `min(k, MS)` clamp 不一致，head 输入宽度算错。实际实现新加 `support_count_fn=lambda k: max(k-1, 0)` 显式声明 prior 数量。
>
> 其他模型（WACNN/STF/TCM/CCA-main）的 sketch 与实际实现 byte-for-byte 一致。详见 [`plan/exec-plans/completed/codec-containerization-h-g-refactor.md`](../exec-plans/completed/codec-containerization-h-g-refactor.md) Phase 5 实施差异段。

```python
# WACNN / SymmetricalTransFormer
self.latent_codec = HyperpriorLatentCodec(
    h_a=stf_h_a(M, N),
    h_s=DualHyperSynthesis(stf_h_mean_s(M, N), stf_h_scale_s(M, N)),
    latent_codec={
        "z": EntropyBottleneckLatentCodec(EntropyBottleneck(N), quantizer="noise"),
        "y": build_channel_slice_codec(
            groups=[M//K]*K, side_channels=2*M, max_support_slices=5,
            leaf_factory=lambda k, ch: LRPGaussianLatentCodec(
                lrp_transform=make_entropy_transform(..., widths=(224,176,128,64))),
            channel_context_factory=lambda k, ch, sup: build_mean_scale_head(
                ..., widths=(224,176,128,64))),
    },
)

# TCM —— 同 WACNN，把 widths 改成 (224, 128)，加 SWAtten support_transform_factory

# CCA-main —— 同 WACNN，groups=_resolve_slice_sizes(M, slice_proportions)，
#   max_support_slices=-1，widths=(em_hidden, 128)，support_transform_factory=NAFTransform，
#   z leaf 的 quantizer="ste"

# CCA-aux —— 不走 HyperpriorLatentCodec 包装（aux 不是真 hyperprior，是 y 自己再编一次）：
self.aux_entropy_model = _CCAAuxEntropyModel(  # ~30 行 thin wrapper
    y_entropy_bottleneck=EntropyBottleneck(M),
    inner=build_channel_slice_codec(   # ← Phase 1 helper, 包 ChannelGroupsLatentCodec
        groups=slice_sizes,
        side_channels=2 * M,
        side_in_context=True,
        max_support_slices=-1,
        support_filter=lambda k, prior: prior[: max(k - 1, 0)],  # skip-most-recent
        leaf_factory=lambda k, ch: (
            LRPGaussianLatentCodec(lrp_transform=make_entropy_transform(...))
            if k < num_slices - 2
            else GaussianConditionalLatentCodec()
        ),
        channel_context_factory=lambda k, ch, sup: build_mean_scale_head(
            slice_ch=ch, support_ch=sup, side_split=M, widths=(em_hidden, 128),
            support_transform_factory=lambda c_in, c_out: NAFTransform(...),
        ),
    ),
)

# DCAE —— 用 application 层的 build_dictionary_mean_scale_head + SharedDictionary：
#   （DCAE 上岸时再做，本 PR 不涉及）

# SAAF —— 与 DCAE 同构（共享 DictionaryEntropyCompressionModel + dictionary cross-attention pattern）：
#   （SAAF 上岸时跟 DCAE 同批迁，本 PR 不涉及）

# MambaVC —— 同 TCM（widths=(224,128) + SWAtten support），自动适配
#   （MambaVC 上岸时再做，本 PR 不涉及）

# ELIC —— 不变，已是终态（HyperpriorLatentCodec + ChannelGroupsLatentCodec + CheckerboardLatentCodec）

# MLIC++ —— 不变（codec 已 codec-owned hyper，monolithic 内部不拆）（fork `script` 已迁入但本 PR 不动）

# MambaIC —— 仅 G：把 monolithic codec 包进 HyperpriorLatentCodec（fork `script` 已迁入但本 PR 不动）

# HPCM, WeConvene, TinyLIC, ShiftLIC, Entroformer, RefBasedAR —— 不动
```

### 10.6 Hyperprior 适配（Direction G 具体）

**`DualHyperSynthesis` adapter**（新文件 `_hyper_synthesis.py`，~25 行）：

```python
class DualHyperSynthesis(nn.Module):
    """Concatenate outputs of two parallel h_s heads along channel dim.
    Used by Family 1 models (STF/TCM/CCA/DCAE/MambaVC) with separate h_mean_s/h_scale_s."""
    def __init__(self, h_mean_s, h_scale_s):
        super().__init__()
        self.h_mean_s, self.h_scale_s = h_mean_s, h_scale_s
    def forward(self, z_hat):
        return torch.cat([self.h_mean_s(z_hat), self.h_scale_s(z_hat)], dim=1)
```

选 adapter 而非扩展 `HyperpriorLatentCodec` 接口的理由：
- `HyperpriorLatentCodec` 接口零变化，对 ELIC 等现有用户零风险
- "two heads" 本身是 documented design，命名为 `DualHyperSynthesis` 在 `__init__` 里直接可见

**CCA z STE quantize**：`EntropyBottleneckLatentCodec(quantizer="ste")` upstream 已支持，直接用。原 §5.5 担心的 `HyperLatentCodec.quantizer` 扩展**不需要**——`HyperLatentCodec` 在 upstream 已被 `EntropyBottleneckLatentCodec` 替代。

**单 h_s 模型**（ELIC, MLIC++）：直接 `h_s=h_s`，不用 wrapper。

### 10.7 State_dict 路径设计（STF 为例）

> **2026-05-09 实施修正**：原稿写作 `latent_codec.latent_codec.*` 双层前缀，但 `HyperpriorLatentCodec.__init__` 把 `self.latent_codec = {...}` 设为普通 dict（不是 nn.ModuleDict），子模块通过 `self.y` / `self.z` 注册——所以路径是单层 `latent_codec.y.*` / `latent_codec.z.*`，下表已更新。`ChannelGroupsLatentCodec` 内部确实有真 nn.ModuleDict `self.latent_codec`，所以 leaves 那一层仍是 `latent_codec.y.latent_codec.y{k}.*`。

| 旧（`pr-tcm-cca` 当前）| 新（H + G 后，commit `8b3ea4d` 验证）|
|---|---|
| `entropy_bottleneck.quantiles` | `latent_codec.z.entropy_bottleneck.quantiles` |
| `h_a.0.weight` | `latent_codec.h_a.0.weight` |
| `h_mean_s.0.weight` | `latent_codec.h_s.h_mean_s.0.weight` |
| `h_scale_s.0.weight` | `latent_codec.h_s.h_scale_s.0.weight` |
| `latent_codec.cc_mean_transforms.{k}.0.weight` | `latent_codec.y.channel_context.y{k}.mean_cc.0.weight` |
| `latent_codec.cc_scale_transforms.{k}.0.weight` | `latent_codec.y.channel_context.y{k}.scale_cc.0.weight` |
| `latent_codec.lrp_transforms.{k}.0.weight` | `latent_codec.y.latent_codec.y{k}.lrp_transform.0.weight` |
| `latent_codec.gaussian_conditional._scale_table` | `latent_codec.y.latent_codec.y{k}.gaussian_conditional._scale_table`（per-slice 副本，K 份）|

**Family 1 `side_in_context=True` 模式额外约束**：channel_context 字典覆盖 `y0..yK-1`（不是 ELIC 默认的 `y1..yK-1`），所以 `latent_codec.y.channel_context.y0.{mean,scale}_cc.*` 在 STF/WACNN/TCM/CCA 的 state_dict 中存在。`infer_num_slices` helper 自动检测 y0 是否存在并据此选择是否 +1。

**WMSA wrapper 路径**：`compressai.layers.attn.swin.WMSA` 内部把 WindowAttention 注册为 `self.attn`，所以 conv_b 内的 attention 参数路径是 `*.conv_b.<i>.attn.attn.{qkv,proj,relative_position_*}`（双层 attn）。上游 Zou et al. ckpt 是单层 `*.conv_b.<i>.attn.{qkv,...}`——`convert_upstream_stf_state_dict` 通过 `_nest_winmsa_keys` 正则自动 nesting 适配。

**LRP byte-for-byte 兼容**：通过 `MeanScaleContextHead(emit_mean_support=True)` + `LRPGaussianLatentCodec(mean_support_trail_channels=M+slice_ch*support_count)`，新 LRP transform 第一 conv 输入宽度跟旧 `M + slice_ch*(support_count+1)` 完全一致——上游 `lrp_transforms.{k}.*` 权重直接转移，无需 fine-tune。

newcomer 可读性：路径就是模型层级——`model → outer hyperprior codec → y branch (= ChannelGroupsLatentCodec) → 第 k 个 slice 的 leaf → Gaussian`。

**GaussianConditional 共享问题**：原本 1 个共享，现在 K 个 per-slice 副本。`_scale_table`、`_offset`、`_cdf_length` 等 buffer 重复 K 份，每模型多 ~几十 KB。可接受，文档里说明。

### 10.8 基类去留

**`SliceEntropyCompressionModel`：删除**。`_init_slice_entropy` / `_hyper_priors` / `_compress_latent` / `_decompress_latent` 全部职责转移到 upstream `HyperpriorLatentCodec` + 扩展后的 `ChannelGroupsLatentCodec` + leaves。Family 1 模型直接继承 `SimpleVAECompressionModel`（upstream 已有，是 `CompressionModel` 加通用 forward/compress/decompress 委托给 `self.latent_codec`）。

**`DictionaryEntropyCompressionModel`：本 PR 不动**（DCAE/SAAF 不在本 PR scope；它们上岸的 follow-up PR 把 dictionary cross-attention 职责下放到 application 层 helper 后再删基类）。

**`ChannelSliceLatentCodec`（pr-stf-wacnn 加的）：删除**。重复 upstream `ChannelGroupsLatentCodec`，没合理保留理由。

**Helper 函数（`infer_num_slices` / `infer_max_support_slices` / `slice_support_channels` / `lrp_support_channels` / `make_entropy_transform`）保留**，搬到 `compressai/latent_codecs/_slice_helpers.py`。`infer_num_slices` 等的 prefix scan 要更新到新 state_dict 路径。

**净效果**：`compressai/models/_bases/` 在本 PR 后只剩 `dictionary_entropy.py`（DCAE/SAAF 上岸时再删）。

### 10.9 转换脚本影响

每个 `convert_*_checkpoint.py` 加一段 `_DIRECTION_GH_RENAMES`：

```python
_DIRECTION_GH_RENAMES = {
    # Hyperprior 进 codec
    "h_a.": "latent_codec.h_a.",
    "h_mean_s.": "latent_codec.h_s.h_mean_s.",
    "h_scale_s.": "latent_codec.h_s.h_scale_s.",
    "entropy_bottleneck.": "latent_codec.latent_codec.z.entropy_bottleneck.",
    # ChannelSlice 容器化（per-slice 循环生成）：
    # cc_mean_transforms.{k}. → latent_codec.latent_codec.y.channel_context.y{k}.mean_cc.
    # cc_scale_transforms.{k}. → latent_codec.latent_codec.y.channel_context.y{k}.scale_cc.
    # lrp_transforms.{k}. → latent_codec.latent_codec.y.latent_codec.y{k}.lrp_transform.
    # gaussian_conditional. → latent_codec.latent_codec.y.latent_codec.y{k}.gaussian_conditional.（K 份）
}
```

| 脚本 | LoC 增加 |
|---|---|
| `convert_stf_checkpoint.py` | ~25 |
| `convert_tcm_checkpoint.py` | ~25 |
| `convert_cca_checkpoint.py` | ~30（aux 路径多一层）|
| `convert_dcae_checkpoint.py` | ~30（dt/dt_cross_attention 移位 + 共享 dictionary 落点）|
| `convert_mambavc_checkpoint.py` | ~25 |
| `convert_mlicpp_checkpoint.py` | ~5（仅顶层 prefix add）|
| `convert_mambaic_checkpoint.py` | ~10 |

**STF PR #354 后续 refactor 需要再生测试 fixture**（`tests/data/states/*`）。

### 10.10 LoC 估算（修订版，本 PR scope = STF/WACNN/TCM/CCA）

> **2026-05-09 实施修正**：以下表格估算总计 **~−520 行**，但**实际 PR diff 是 +4596 / −655 = 净 +3941**。偏离根因是估算把 TCM (`-100`) 和 CCA (`-180`) 当作既有文件 trim，但本 PR 提交给 upstream 的 base 是 `upstream/master`，**TCM/CCA 在 upstream 不存在**——必须从零写入容器化版本，TCM ~700 行 + CCA ~1100 行 + CCA loss ~130 行 + 2 个 convert script ~250 行 + per-model 测试 ~500 行，全部是净增。下表保留作为「假设 fork 是 base」的理论估算；真实 PR diff 见 [`plan/generated/pr-tcm-cca-draft.md`](../generated/pr-tcm-cca-draft.md) 的 Commits 段。

| 文件 | ±LoC | 说明 |
|---|---|---|
| `compressai/latent_codecs/channel_slice.py` | **−270** | DELETE（重复 upstream `ChannelGroupsLatentCodec`）|
| `compressai/latent_codecs/channel_groups.py` | **+10** | 加 `max_support_slices` + `support_filter` 两个可选参数（向后兼容）|
| `compressai/latent_codecs/gaussian_conditional.py` | **+30**（追加 subclass）| 在 upstream 文件末尾加 `LRPGaussianLatentCodec(GaussianConditionalLatentCodec)`，与基类同文件 |
| `compressai/latent_codecs/_hyper_synthesis.py`（新）| **+25** | `DualHyperSynthesis` |
| `compressai/latent_codecs/_slice_helpers.py`（新——搬家）| **+120** | `make_entropy_transform` / `infer_*` 等 |
| `compressai/latent_codecs/__init__.py` | **+5** | exports |
| `compressai/models/_helpers/channel_slice.py`（新）| **+40** | application-layer `build_channel_slice_codec` factory |
| `compressai/models/_helpers/channel_context.py`（新）| **+80** | application-layer `build_mean_scale_head` 工厂 |
| `compressai/models/_bases/slice_entropy.py` | **−260** | DELETE |
| `compressai/models/_bases/__init__.py` | **−10** | trim（保留对 `dictionary_entropy` 的 import）|
| `compressai/models/stf.py`（WACNN + STF）| **−50** | 用容器化 wiring 替代 `_init_slice_entropy` |
| `compressai/models/tcm.py` | **−100** | 删 helper + ModuleList plumbing |
| `compressai/models/cca.py` | **−180** | main forward 4 行；aux 变 thin wrapper |
| `convert_*_checkpoint.py`（STF/TCM/CCA 3 个）| **+90** | rename map |
| 测试 fixture | **+50** | 新 path-shape 测试 + 重新生成 tiny state dict |
| **总计（fork-baseline 假设下）** | **~−520** | 主要来自 `channel_slice.py` 删除 + `_bases/slice_entropy.py` 删除 + 3 个 Family 1 模型 codec 构造瘦身 |

**实际 PR diff（upstream-baseline）**：23 files, +4596 / −655，净 +3941。delta 主要来自 TCM/CCA 在 upstream 不存在，必须从零写入容器化版本（TCM ~700 + CCA ~1100 + CCA loss ~130 + 2 convert scripts ~250 + per-model 测试 ~500）。

如果**算上 DCAE/MambaVC follow-up PR**：
- `compressai/models/dcae.py` −90 + `compressai/models/_bases/dictionary_entropy.py` −310 + `compressai/models/_helpers/channel_context.py` +80 (DictionaryHead) + DCAE convert script +30
- `compressai/models/mambavc.py` −40 + MambaVC convert script +25
- 累计 follow-up 净 LoC ~−305

合计完整 H+G refactor 跨 PR 总计 **~−825 行**。

### 10.11 跟 upstream codec 的复用关系（本 PR PR description 用）

| upstream codec | 本 PR 怎么用 |
|---|---|
| `LatentCodec`（base.py）| 父类，所有 leaf / 容器都继承 |
| `HyperpriorLatentCodec`（hyperprior.py）| **核心容器**：模型 wiring 顶层 |
| `EntropyBottleneckLatentCodec`（entropy_bottleneck.py）| z 编码 leaf；CCA 用 `quantizer="ste"`，其他用默认 `"noise"` |
| `GaussianConditionalLatentCodec`（gaussian_conditional.py）| Family 1 leaf 基类（`LRPGaussianLatentCodec` subclass 用于全部 K 个 slice，含 CCA-aux）|
| `ChannelGroupsLatentCodec`（channel_groups.py）| **核心容器**：本 PR 复用 + 加 2 个可选参数（`max_support_slices` + `support_filter`）|
| `CheckerboardLatentCodec` | 不用（Family 2 ELIC 内嵌用，本 PR Family 1 不涉及）|
| `HyperLatentCodec` | 不用（upstream 已 deprecated）|
| `RasterScanLatentCodec`, `gain/*` | 不用（其他 family 用）|

### 10.12 诚实评估 / dead-end

- **HPCM, WeConvene, TinyLIC, ShiftLIC, Entroformer, RefBasedAR 不容器化**——它们的 forward loop 有不可分解的结构，强行套容器会引入间接性而无抽象收益
- **MLIC++ 内部不容器化**——anchor/nonanchor LRP 双套 + 多参考 contexts 形成 6 路分裂，对单一用户做 scaffolding 不值得

  > **2026-05-10 修订**：本结论已被 [`plan/exec-plans/active/family2-roadmap.md`](../exec-plans/active/family2-roadmap.md) PR-1 推翻。修订理由：(a) MLIC++ codec 拆解发现「双 EP head + 多 context + 双 LRP」 pattern 的 6 路分裂可被分层抽象 —— 外层 K-slice 循环复用 upstream `ChannelGroupsLatentCodec`，内层 anchor/nonanchor 双 pass 抽成 `CheckerboardLatentCodec` 的 sibling leaf `MultiContextCheckerboardLatentCodec`；(b) MambaIC（PR-3）有可能复用同一 sibling leaf，"单一用户" 假设可能不成立；(c) 教学价值：sibling leaf 让用户理解 MLIC++ 与 ELIC 的真实差异（separate heads + LocalContext intra-spatial-context vs shared head + CheckerboardMaskedConv2d），比单体类更 pedagogical。详细抽象方案见 [`plan/exec-plans/active/pr-mlicpp-upstreaming.md`](../exec-plans/active/pr-mlicpp-upstreaming.md) §抽象设计草稿。
  >
  > **2026-05-10 Phase 1 实施进展**：`compressai/latent_codecs/multi_context_checkerboard.py`（311 行）+ `compressai/latent_codecs/_checkerboard_helpers.py`（145 行，sibling 共享 single source of truth）已落地，7 个单元测试全过（含 ELIC 等价回归 `test_matches_checkerboard_latent_codec_when_heads_are_shared`：sibling 配 `spatial_context_anchor=_ZeroContext(out_channels)` + 共享 EP head 时 forward 数值与 upstream `CheckerboardLatentCodec._forward_twopass` 完全一致）。`spatial_context_*=None` 语义定为「skip y_ctx 槽」（不 zero-pad），保证 MLIC++ k=0 anchor head 输入维度可直接是 `2M` 不需要 padding，byte-for-byte 上游 ckpt 兼容。Phase 2-9（mlic 子包 lift / factory / model / convert / zoo / 测试）待动手。
- **`ChannelGroupsLatentCodec` 不合并到 `ChannelSliceLatentCodec`**——前者用 joint mean+scale + per-group inner spatial codec（如 Checkerboard），后者用 separate mean/scale + per-slice flat leaf。强行合并需要 §4 option F 的 `intra_slice_context_factory` 抽象，留给第 4 个 Family 2 用户出现时再考虑
- **`GaussianConditional` per-slice 副本**：尝试过共享但 PyTorch state_dict 不 dedupe by-id，K 份副本各 ~几 KB 实用上 acceptable
