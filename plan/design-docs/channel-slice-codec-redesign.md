# Channel-Slice Latent Codec 跨模型抽象速查表

**记录日期：** 2026-05-09
**最近修订：** 2026-06-04（重构：拆分决策/实施记录到 [`codec-containerization-h-g-refactor.md`](../exec-plans/completed/codec-containerization-h-g-refactor.md)「设计依据」段；本文 §1–§4 保留为 codec 家族分类速查表。**同日补充调研**：盘点 `compressai/models/` 新增的 8 个 entropy head，新增 Family 8（空间 raster-scan AR）/ Family 9（非高斯 VQ），见 §2.8/§2.9/§3.1）

本文档梳理 CompressAI fork 上**全部 entropy 编码家族**的真实**变异维度**与**结构分类**，是跨模型 latent codec 抽象的参考底图。命名沿用「channel-slice」是历史原因（最初只调研 channel-slice 家族），现已扩展到九个结构族。

> **本文档定位（2026-06-04 起）**：纯参考表——codec 家族变异维度（§2）+ 九族结构分类（§3）+ 容器化适用边界（§4）。当初由此调研推导出的 **H+G 容器化重构**（候选方向对比 A–H、推荐方案、详细 API/state_dict 设计、LoC 估算）已实施完毕并整体并入 [`codec-containerization-h-g-refactor.md`](../exec-plans/completed/codec-containerization-h-g-refactor.md)「设计依据」段，本文不再重复。
>
> **触发背景**：在 `pr-tcm-cca` 分支整合 TCM/CCA 时发现 `SliceEntropyCompressionModel` 把 entropy head widths 写死了 STF 的 5-conv `(224,176,128,64)`（TCM 用 3-conv `(224,128)`、CCA 要变长 slice + skip-most-recent support），按现状逐模型打补丁会把 codec 变成 swiss army knife——遂有此调研。
>
> **2026-06-04 补充调研**：`compressai/models/` 自首轮调研后又增 8 个 entropy head——**ContextFormer / GLLMM / Informer / LBHIC / NIC / TIC / NVTC / TBTC**（均已注册 zoo entry，且**全部 hand-written、不走 `LatentCodec` 接口**）。它们大多落在首轮未覆盖的 **Family 8（空间 raster-scan AR，不切片）** 与 **Family 9（非高斯 VQ）**，并把 Minnen2018-joint baseline（mbt2018 / cheng2020）确立为 Family 8 的祖先。详见 §2.8/§2.9、§3.1。

---

## 1. 调研对象（25 个 entropy head）

> **status 列反映当前事实**（2026-06-04）。多数 channel-slice 模型已沿 upstream 约定逐 PR 迁入 `InterDigitalInc/CompressAI`；fork `script` 全量主干保留各模型原始落地形态。表分两段：**首轮调研的 channel-slice / 容器化家族（17 head，Family 1–7）** + **2026-06-04 补充调研的新增 entropy head（8 head，Family 8/9 + F1/F6 变体）**。

### 1.1 首轮调研：channel-slice / 容器化家族（Family 1–7）

| 来源模型 | 论文/年代 | 文件 | 状态 |
|---|---|---|---|
| STF (`SymmetricalTransFormer`) | Zhu CVPR 2022 | `compressai/models/stf.py` | 已上游迁入（PR #3 系）|
| WACNN | Zhu CVPR 2022 | `compressai/models/stf.py` | 已上游迁入（PR #3 系）|
| TCM | Liu CVPR 2023 | `compressai/models/tcm.py` | 已上游迁入（pr-tcm-cca）|
| CCA-main (`CCAModel`) | Han NeurIPS 2024 | `compressai/models/cca.py` | 已上游迁入（pr-tcm-cca）|
| CCA-aux (`_CCAAuxEntropyModel`) | Han NeurIPS 2024 | `compressai/models/cca.py` 内部类 | 已上游迁入（pr-tcm-cca）|
| ELIC (`Elic2022Official` / `Elic2022Chandelier`) | He CVPR 2022 | `compressai/models/sensetime.py` | 仓库已有 |
| **GLIC** | Chen CVPR 2026 | `compressai/models/glic.py` | 已上游迁入（PR #4；与 ELIC 同构用 upstream `ChannelGroupsLatentCodec`，无新 codec 类）|
| **CMIC** | Chen CVPR 2024 | `compressai/models/cmic.py` | 已上游迁入（PR #5，与 MambaIC 合并）；与 ELIC/GLIC 同构用 upstream `ChannelGroupsLatentCodec` + `CheckerboardLatentCodec`，**无新 codec 类**；novelty 在 content-aware Mamba context blocks |
| MLIC++ | Jiang TIP 2024 | `compressai/models/mlicpp.py` | 已上游迁入（pr-mlicpp，含 `MultiContextCheckerboardLatentCodec` sibling leaf）|
| **DCAE** | Wu 2024 | `compressai/models/dcae.py` | 已上游迁入（pr-dcae-saaf-auxt）|
| **SAAF** | Ma CVPR 2026 | `compressai/models/saaf.py` | 已上游迁入（pr-dcae-saaf-auxt；与 DCAE 共享 dictionary cross-attention pattern）|
| **HPCM** | He 2024 | `compressai/models/hpcm.py` | fork `script` 已迁入，专属 latent codec |
| **MambaIC** | Zeng CVPR 2025 | `compressai/models/mambaic.py` + `latent_codecs/mambaic.py` | 已上游迁入（PR #5，与 CMIC 合并）；专属 codec（评估后保留 dedicated）|
| **MambaVC** | Qin 2024 | `compressai/models/mambavc.py` | fork `script` 已迁入；**不上游迁入**（仅 arXiv 预印本，见 roadmap §E.14）|
| **WeConvene** | Wang ECCV 2024 | `compressai/models/weconvene.py` + `latent_codecs/weconvene.py` + `weconvene_support.py` | 已上游迁入（PR #7，新 `WeChARMLatentCodec` + `[wavelet]`）|
| **TinyLIC / ShiftLIC large** | Lu arXiv 2022 / Bao TCSVT 2025 | `compressai/models/{tinylic,shiftlic}.py` + `latent_codecs/multistage_checkerboard.py` | 已上游迁入（PR #6）；**两模型共享同一 codec** |
| **FTIC** | Li ICLR 2024 | `compressai/models/ftic.py` | 已上游迁入（PR #8，新 `GsnConditionalLocScaleShift`）；**无新 codec 类**——直接 `TCAEntropyModel` + `GsnConditionalLocScaleShift`，**结构上等价于 Family 1 channel-slice AR 的 transformer-context 变体**（见 §2.7、§3.1）|
| Entroformer | Qian ICLR 2022 | `compressai/models/damo.py` + `latent_codecs/transformer_ar.py` | fork `script` 已迁入，bitstream `NotImplementedError` |
| RefBasedAR (qian2021-ref) | Qian ICLR 2021 | `compressai/models/damo.py` + `latent_codecs/ref_autoregressive.py` | fork `script` 已迁入，bitstream `NotImplementedError` |

### 1.2 补充调研（2026-06-04）：新增 entropy head（Family 8/9 + F1/F6 变体）

> 这 8 个 head **全部 hand-written**（在 model 自己的 `forward`/`compress`/`decompress` 里直连 entropy_model，**不走 `LatentCodec` 接口、不进 §3.4 的 codec 类清单**），多数落在首轮未覆盖的 **Family 8（空间 raster-scan AR，不切片）/ Family 9（非高斯 VQ）**。zoo entry 已注册。详见 §2.8/§2.9、§3.1。

| 来源模型 | 论文/年代 | 文件 | 归族 / 要点 |
|---|---|---|---|
| **ContextFormer** | Koyuncu ECCV 2022 | `compressai/models/contextformer.py` + `layers/attn/contextformer.py` | **Family 6 时空变体**：spatio-channel 因果 Transformer-AR（channel segment + spatial token）+ **GMM K=3**；compress/decompress `NotImplementedError` |
| **TIC** | Lu DCC 2022 | `compressai/models/tic.py` | **Family 8**：不切片 + 单 Gaussian + **因果注意力窗 (CAM 5×5)** 替代 MaskedConv 的光栅 AR；compress 已实现 |
| **GLLMM** | Fu/Liang TIP 2023 | `compressai/models/gllmm.py` | **Family 8**：不切片 + MaskedConv-A 5×5 光栅 AR + **Gaussian-Laplacian-Logistic 混合**（3 族 × K=3）；compress 已实现 |
| **Informer** | Kim CVPR 2022 | `compressai/models/informer.py` | **Family 8**：不切片 + MaskedConv 光栅 AR + 单 Gaussian + **全局/局部双 hyperprior（cross-attention）**；compress 已实现 |
| **LBHIC** | paper-only（块混合）| `compressai/models/lbhic.py` + `layers/lic/lbhic.py` | **Family 8**：**空间 block 光栅**（非通道切片）+ MaskedConv-AR + GMM K=3 + 块间预测/边界后处理；compress `NotImplementedError` |
| **NIC / NLAIC** | Chen TIP 2021 | `compressai/models/nic.py` | **Family 8**：不切片 + **3D MaskedConv 11³（通道+空间联合 AR）** + GMM K=3 + 非局部注意力；**无 compress/decompress** |
| **NVTC** | Feng CVPR 2023 | `compressai/models/{nvtc,nvtc_support}.py` + `layers/lic/nvtc.py` | **Family 9（非高斯）**：entropy-constrained **VQ + 学习类别先验**（`DiscreteConditionalEntropyModel`），无 hyperprior、无连续高斯；compress/decompress `NotImplementedError` |
| **TBTC**（zyc2022-*）| Zhu/Yang ICLR 2022 | `compressai/models/tbtc.py` | **一文两范式**：`-charm` 变体 = **Family 1** 通道 ChARM（chunk(10) 等大、use-all-prior、分离 mean/scale、无 LRP）；`-hyperprior` 变体 = **不切片 mean-scale baseline**（Family 0）；compress 均已实现 |

### 1.3 Family 8 的祖先：仓内已有的 Minnen2018-joint baseline

> Family 8（空间 raster-scan AR）的祖先是 CompressAI 早就有的经典 baseline——补充调研把它们一并列出以呈现完整谱系（它们不是新迁入对象，是参照原点）。

| 来源模型 | 论文/年代 | 文件 | 归族 / 要点 |
|---|---|---|---|
| `JointAutoregressiveHierarchicalPriors` (mbt2018) | Minnen NeurIPS 2018 | `compressai/models/google.py` | **Family 8 祖先**：不切片 + MaskedConv-A 5×5 光栅 AR + 单 Gaussian + 模型自拥 hyper |
| Cheng2020Anchor / Cheng2020Attention | Cheng CVPR 2020 | `compressai/models/waseda.py` | **Family 8 祖先**：继承 mbt2018 的 MaskedConv 光栅 AR（Anchor 加 residual block，Attention 加 attention）|
| `MeanScaleHyperprior` (mbt2018-mean) / `ScaleHyperprior` (bmshj2018-hyperprior) | Ballé ICLR 2018 / Minnen 2018 | `compressai/models/google.py` | **Family 0 baseline**：无空间 AR、不切片，纯 hyperprior mean-scale（TBTC `-hyperprior` 与 gained 系列的 baseline）|

**计数**：首轮 17 head（Family 1–7）+ 补充 8 head（Family 8/9 + F1/F6 变体）= **25 个 entropy head**；外加 3 个 Minnen2018/Ballé baseline 祖先（Family 8 / Family 0）。其中只有 Family 1–7 的 17 个走 `LatentCodec` 接口（**17 个 codec 类**，见 §3.4）；补充调研的 8 head 与 3 个 baseline 祖先**全部 hand-written**，不引入新 codec 类。CMIC / GLIC / FTIC 复用 upstream codec primitives，不在 17 类中各占一席。

现有抽象层（fork `script`）：
- `compressai/latent_codecs/channel_slice.py::ChannelSliceLatentCodec`
- `compressai/models/_bases/slice_entropy.py::SliceEntropyCompressionModel`
- `compressai/models/_bases/dictionary_entropy.py::DictionaryEntropyCompressionModel`（DCAE/SAAF 共享，与 `SliceEntropyCompressionModel` 90% 重合）
- `compressai/latent_codecs/hpcm.py::HierarchicalProgressiveLatentCodec`（HPCM 专属，800+ 行 mask 逻辑）
- `latent_codecs/{mambaic,weconvene,multistage_checkerboard,transformer_ar,ref_autoregressive}.py`（5 个其他专属 codec）

ELIC / GLIC / CMIC 用 upstream `ChannelGroupsLatentCodec`，MLIC++ 用 `MLICPlusPlusLatentCodec`，MambaIC 用 `MambaICLatentCodec`。

> **上游 migration 线说明**：STF/WACNN/TCM/CCA 在上游已按 H+G 容器化（删除 `ChannelSliceLatentCodec` + `SliceEntropyCompressionModel`，改用 upstream `HyperpriorLatentCodec` + `ChannelGroupsLatentCodec` 嵌套）。本文 §1/§3.4 的 17-codec 快照描述的是 fork `script` 全量主干现状；容器化后的 state_dict 路径与设计见 [exec-plan 设计依据段](../exec-plans/completed/codec-containerization-h-g-refactor.md#design-rationale)。

---

## 2. 变异维度对比

本节按 family 分组对比各 family 的 10 个变异维度（外加一行「独特点」捕获 family 特有的设计细节）。**FTIC 归入 Family 1 的 transformer-context 变体**（§2.7）；**§2.8/§2.9 是 2026-06-04 补充调研新增的 Family 8（空间 raster-scan AR）/ Family 9（非高斯 VQ）**；ContextFormer 作为 Family 6 时空变体并入 §2.6。Family 编号、命名与结构族叙事见 §3.1；codec 类完整分类见 §3.4。Family 内不存在的轴标 `n/a`。

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

> **TBTC `-charm`（Zhu/Yang ICLR 2022，补充调研）也属 Family 1**：通道 ChARM——`chunk(num_slices)` 等大（默认 10）、use-all-prior、**分离 mean/scale 支持变换**（两套独立 ModuleList，多数 ChARM 共用单 head 出 2×通道，这里完全解耦）、**无 LRP**、无 intra-slice 空间上下文，是最干净的「ChARM」范例。但**hand-written**（不走 `ChannelGroupsLatentCodec`）。同文件的 `-hyperprior` 变体不切片，属 Family 0 mean-scale baseline（见 §1.3）。详见 §1.2。

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

### 2.6 Family 6 / 7：Entroformer / RefBasedAR（无切片全分辨率 AR）+ ContextFormer（F6 时空变体）

| 维度 | Entroformer (Family 6) | RefBasedAR (Family 7) | ContextFormer (F6 时空变体) |
|---|---|---|---|
| Slice 切法 | **不切** | **不切** | 通道 segment（Ncs=4）用于 token 化，非 codec 切片 |
| Support 截断 | n/a | n/a | n/a（causal mask 看全部已编码 token）|
| mean_support transform | n/a（联合预测）| n/a（联合预测）| n/a（联合）|
| scale_support transform | n/a（联合预测）| n/a（联合预测）| n/a |
| CC head 架构 | 联合 `param_net` MLP，输出 `num_parameter * M` 通道 | 三 cascade 各自 joint head（logit_pi + mean + log-σ）| 联合 MLP（Linear×3），输出 `3·mixtures·segment_ch`（logits+means+log_scales）|
| LRP | n/a | n/a | no |
| 分布 | 单 Gaussian (`num_parameter=2`) | **K=3 GMM** (`GaussianMixtureConditional`) | **GMM K=3** (`GaussianMixtureConditional`) |
| Intra-slice 空间上下文 | 因果 Transformer (`y_ar = TransDecoder`) over 全 latent map | 5×5 mask-A `MaskedConv2d` (local) + `_SearchTransfer` (cosine 全局参考搜索) + `_Conv2dUnfold` | **时空因果 Transformer-AR**：(H·W·Ncs) token 序列右移 + 上三角 causal mask（`cfo`/`sfo` 排序）|
| Compress/decompress 循环 | 单 forward pass；compress `NotImplementedError` | 3 cascade head 单 pass；compress `NotImplementedError` | teacher-forced 单 pass；compress/decompress `NotImplementedError` |
| Hyperprior 归属 | **codec 自带** `y_hyper_encode/decode` + `LearnedGaussianBottleneck` | **codec 自带** `LearnedGaussianBottleneck` + `GaussianMixtureConditional` | **模型拥有** `h_a/h_s/entropy_bottleneck`（hand-written，无 codec）|
| 独特点 | 全分辨率 Transformer-AR；hyperprior 移入 codec | log-sum-exp soft-attention 混合参考补丁；K=3 GMM；`mask_unfold` 不可训练 buffer | **spatio-channel 联合**因果 Transformer（segment + spatial token 一起 AR）+ GMM；比 Entroformer 多了 channel 维 AR 与混合分布 |

> **ContextFormer（Koyuncu ECCV 2022）归族**：与 Entroformer 同属 Transformer-AR，列为 **Family 6 时空变体**——区别是它对 **channel segment + spatial position 联合**自回归（Entroformer 仅 spatial-AR），且 leaf 用 **K=3 GMM**（Entroformer 单 Gaussian）。hand-written、不走 `LatentCodec`，compress/decompress 仍 `NotImplementedError`。

### 2.7 Family 1 变体：FTIC（masked-channel transformer 作为 channel context）

FTIC 把 Family 1 的 per-slice MLP/SWAtten/NAF channel-context **换成一个 masked-channel transformer (TCA)**，其余维度全部落在 Family 1 范围内。三个易误判处澄清如下：

1. **不是 3-参数 Gaussian**——`GsnConditionalLocScaleShift` 的 `forward(inputs, scales, means)` 只用 2 参数，"shift" 指的是 `inputs - round(mean)` + `mean - round(mean)` 这套 **integer / fractional mean 解耦**的 indexed-CDF 优化（Charm/Ballé 系一直在用），分布本身仍是 2-参数 Gaussian、跟现仓 `GaussianConditional` 数学等价。
2. **第三路输出 `lrp` 是标准 LRP**——按 `0.5 * tanh(lrp)` 加在 `y_hat` 上，跟 MLIC/STF/CCA-main 同款，不是分布参数。
3. **结构上是 channel-slice AR**——`tca.py` 用 `torch.cat((start_token, y[:, :-C//slices]), dim=1)` 把输入整体右移一个 slice、左侧补 hyper 派生的 start_token；再叠 `MaskedSliceChannelAttention` 的 head-内 causal mask，使 slice i 的 (means/scales/lrp) 严格只依赖 slice 0..i-1 的 y_hat，与 `ChannelGroupsLatentCodec` 的 channel-AR 语义一致；compress/decompress 也已写成 K=5 次 slice 循环。原作者用 grouped conv (`groups=slices`) 把 K 路 head 折叠成单个 conv，是实现技巧而非结构差异。

> 复核基础：`candidate/FTIC/models/{flic,tca,entropy_models}.py` 原作者代码（2026-05-11）。

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
| 独特点 | **唯一真正特异点是 channel-context 模块**：用 transformer + masked attention 替代 Family 1 的 per-slice MLP/SWAtten/NAF。其余维度（chunk(K) 切片、use-all-prior、joint mean/scale/lrp head、Gaussian leaf、K-pass decompress、双 h_s 模型主拥有）全部落在 Family 1 范围内。可容器化为 `ChannelGroupsLatentCodec(channel_context={...TCA adapter...}, latent_codec={LRPGaussianLatentCodec ×K})`，详见 §4 |

### 2.8 Family 8：空间 raster-scan AR（不切片，Minnen2018-joint 谱系）

> **2026-06-04 补充调研新增**。这一族**不做 channel 切片**——整张全分辨率 latent 一起编码，因果性由**空间自回归**（MaskedConv / 3D MaskedConv / 因果注意力窗）提供，是 Minnen2018-joint（mbt2018）的直系后代。与 Family 6（Entroformer 全分辨率 Transformer-AR）的区别：F8 的空间上下文是**局部**因果窗（5×5 conv / 5×5 attention / 11³ 3D-conv），不是全局 transformer。全部 hand-written，无 `LatentCodec` 类。

| 维度 | mbt2018 (祖先) | Cheng2020 (祖先) | TIC | GLLMM | Informer | LBHIC | NIC/NLAIC |
|---|---|---|---|---|---|---|---|
| Slice 切法 | 不切 | 不切 | 不切 | 不切 | 不切 | **空间 block**（128px tile，逐块光栅）| 不切 |
| Support 截断 | n/a | n/a | n/a | n/a | n/a | n/a（块间用已解码上/左块）| n/a |
| mean_support transform | n/a（联合）| n/a | n/a | n/a | n/a | n/a | n/a |
| scale_support transform | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| CC head 架构 | `entropy_parameters` 1×1 MLP 输出 2M（联合 mean+scale）| 同 mbt2018 | 1×1 MLP（**GELU**）输出 2M，输入拼 CAM 输出 | 1×1 MLP `4C→640→1280→30C`，输出 GLL 混合参数 | 1×1 MLP，输入 = MaskedConv 上下文 **+ 全局 prior cross-attention (`ca_s`)** | 1×1 conv `4M→3M→3M→3·gmm·M` | **3D 1×1×1 conv** MLP 输出 9 通道（3 component × wms）|
| LRP | no | no | no | no | no | no（BPM 是 image-domain 后处理）| no |
| 分布 | 单 Gaussian | 单 Gaussian | 单 Gaussian | **GLL 混合**（Gaussian+Laplace+Logistic，3 族 × K=3）| 单 Gaussian | **GMM K=3** | **GMM K=3** |
| Intra-slice 空间上下文 | **MaskedConv2d-A 5×5** 光栅 | 同 mbt2018（+ residual/attention block）| **因果注意力窗 (CAM 5×5)** | **MaskedConv2d-A 5×5** | **MaskedConv2d 5×5 + 全局/局部双 hyper cross-attn** | **MaskedConv2d 5×5**（逐块）| **3D MaskedConv-A 11³（通道+空间联合 AR）** + 非局部注意力 |
| Compress/decompress 循环 | 逐像素光栅（已实现）| 同 mbt2018 | 逐像素光栅（已实现）| 逐像素光栅（已实现，需 `build_cdf` per-pixel）| 逐像素光栅（已实现）| 逐块光栅（**compress `NotImplementedError`**）| 单 forward（**无 compress/decompress 方法**）|
| Hyperprior 归属 | 模型拥有 | 模型拥有 | 模型拥有 | 模型拥有 | 模型拥有 **×2（全局+局部）** | 模型拥有 | 模型拥有（z 用 `EntropyBottleneck(filters=(3,3,3))`）|
| 独特点 | channel-slice 出现前的经典 baseline | residual/attention 强化 transform，entropy 路径同 mbt2018 | 空间上下文用**因果注意力窗**替 MaskedConv | **GLL 混合分布**（唯一非高斯-混合的连续模型）+ per-pixel build_cdf | **全局 token cross-attention hyperprior** 叠在局部 MaskedConv 之上 | **image-domain 块预测 + 边界后处理**（视频帧内预测移植）| **3D masked-conv 联合通道+空间 AR** + 非局部注意力（NLAIC 签名）|

### 2.9 Family 9：非高斯 / 向量量化（VQ）

> **2026-06-04 补充调研新增**。这一族**不是连续高斯 VAE**——用 entropy-constrained 向量量化 + 学习的**类别（categorical）先验**，没有 mean/scale、没有 hyperprior、没有 `GaussianConditional`。Family 1–8 的 10 个维度大多 `n/a`，下表描述其实际机制。

| 维度 | NVTC |
|---|---|
| Slice 切法 | 不切通道；**空间 block 切分 + 多级（n_stage=3）多层残差 VQ codebook 栈** |
| Support 截断 | use-all-prior（残差累积：层 L 的 prior 由已解码层的累积重建预测）|
| mean / scale support | **n/a（无 mean/scale）** |
| CC head 架构 | **类别-logit head**：`prior_estimator` conv→ResBlock→conv 产出 per-block 向量 → MLP 映射到 `cb_size` 个 softmax logits（codebook 上的条件 PMF）|
| LRP | n/a（有 residual-VQ refinement，概念类似但不是 `0.5*tanh`）|
| 分布 | **学习的类别 / VQ codebook 先验（非高斯）**——`DiscreteConditionalEntropyModel` + `ECVQLastDim`（entropy-constrained VQ）+ `ConditionalVectorQuantization` |
| Intra-slice 空间上下文 | 无（空间结构由 block partition + `DepthwiseBlockFC` 块内混合给出；上下文跨 VQ 层而非空间 AR）|
| Compress/decompress 循环 | **compress/decompress 均 `NotImplementedError`**（仅 forward / rate 估计；上游标 range coder 为 TODO）|
| Hyperprior 归属 | **无 hyperprior**（无 z / h_a / h_s）；条件来自跨层 conditional VQ 先验 |
| 独特点 | entropy-constrained VQ + 学习条件类别先验 + 多级残差 VQ refinement。**仓里唯一非高斯/非连续 entropy 范式**，且无可用 range coder |

---

### 3.1 九个结构族 + 两个变体（FTIC / ContextFormer）

> **Family 总数 = 9**。首轮调研给出 Family 1–7（FTIC 曾被列为独立 Family 8 transformer-AR，复核后 demote 为 Family 1 transformer-context 变体，见 §2.7）。**2026-06-04 补充调研**新增 **Family 8（空间 raster-scan AR，不切片）** 与 **Family 9（非高斯 VQ）**，并把 ContextFormer 列为 Family 6 时空变体（§2.6）。

- **Family 1（纯 channel-slice，1-pass）**：STF, WACNN, TCM, **DCAE**, CCA-main, CCA-aux, **MambaVC**，**FTIC**（transformer-context 变体），**TBTC `-charm`**（补充调研，最干净的 ChARM 范例，分离 mean/scale + 无 LRP，hand-written）
  - 共享同一个 forward loop：split → 逐 slice 算 (mu, scale) → Gaussian → quantize → 可选 LRP
  - 真正差异收敛在 5 个轴：slice 切法、support 截断、support 变换、CC head 宽度、LRP 范围
  - DCAE 是 cousin（support 多塞 dictionary cross-attention），fork `script` 上为它和 SAAF 单设了 `DictionaryEntropyCompressionModel` 基类
  - **MambaVC 已经在用 `SliceEntropyCompressionModel`**
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
- **Family 6（全分辨率 Transformer AR）**：Entroformer，**ContextFormer**（时空变体）
  - 不切片；Transformer-AR over latent；Entroformer codec 自带 hyperprior、单 Gaussian、bitstream NotImpl；ContextFormer spatio-channel 联合 AR + GMM K=3、模型自拥 hyper、bitstream NotImpl
- **Family 7（参考搜索 GMM）**：RefBasedAR
  - 不切片；K=3 GMM；全局 cosine 参考搜索；codec 自带 hyperprior；bitstream NotImpl
- **Family 8（空间 raster-scan AR，不切片）⭐ 补充调研新增**：TIC, GLLMM, Informer, LBHIC, NIC/NLAIC（祖先 = mbt2018 / Cheng2020）
  - 整张 latent 一起编码，因果性靠**空间自回归**（局部因果窗）：MaskedConv2d（mbt2018/GLLMM/Informer/LBHIC）/ 因果注意力窗 CAM（TIC）/ 3D MaskedConv 联合通道+空间（NIC）
  - 与 Family 6 的区别：F8 是**局部**因果窗，F6 是**全局** Transformer
  - 分布从单 Gaussian（mbt2018/TIC/Informer）到 GMM K=3（LBHIC/NIC）到 GLL 三族混合（GLLMM）
  - **全部 hand-written，无 `LatentCodec` 类**；compress 实现参差（mbt2018/Cheng2020/TIC/GLLMM/Informer 已实现；LBHIC NotImpl；NIC 根本没有 compress/decompress）
- **Family 9（非高斯 / VQ）⭐ 补充调研新增**：NVTC
  - entropy-constrained VQ + 学习条件类别先验（`DiscreteConditionalEntropyModel`）；无 hyperprior、无连续高斯、无 mean/scale
  - **仓里唯一非高斯范式**；compress/decompress 均 NotImpl（仅 rate 估计）
- **Family 0（无 AR mean-scale baseline，参照原点）**：ScaleHyperprior (bmshj2018-hyperprior), MeanScaleHyperprior (mbt2018-mean), TBTC `-hyperprior`, gained 系列
  - 不切片、无空间 AR，纯 hyperprior；是 Family 1/8 的共同出发点

### 3.2 各 family 的 forward loop 结构性不同，不能强行合并

- 1-pass channel-slice (F1) vs 2-pass checkerboard (F2) vs 12-pass multi-resolution (F3) vs 2×顺序 1-pass wavelet (F4) vs 4+2+2+1 sub-stage (F5) vs 全分辨率 Transformer-AR (F6) vs cascade 参考搜索 GMM (F7) vs **空间 raster-scan AR (F8)** vs **VQ 残差多级 (F9)** ——是真实的结构差异
- mean+scale **联合 vs 分离**影响 `cc_*_transforms` 的 ModuleList 结构和 state_dict key
- F6/F7 把 hyperprior 和 / 或 entropy head 完全 collapse 到单个 nn.Module，跟 F1-F5 的"codec 收 (means, scales)"模型主拥有的设计哲学不同
- **F8 是 channel-slice 出现前的范式**（逐像素/逐块空间 AR），与 F1–F2 的「按通道切」正交；强塞进 channel-slice 容器无意义
- **F9 根本不是高斯 VAE**——无连续 latent、无 mean/scale，现有 `LatentCodec` / `GaussianConditional` 体系完全不适用
- F4-F9 跟 F1 的差异是**不可参数化的架构选择**（wavelet 切分 / 联合-vs-分离 / 切片-vs-不切片 / 空间-AR-vs-通道-AR / 单 Gaussian vs GMM vs GLL混合 vs VQ）
- **F1 内部 FTIC 变体 / F6 内部 ContextFormer 变体**只是 context 模块换 transformer，forward loop 形态仍是各自 family 标准

**`per-model codec class` 是 maintainer 的明确方向**：fork `script` 上 17 个走 LatentCodec 接口的 codec 类覆盖 Family 1–7；FTIC、ContextFormer、以及全部 Family 8/9 模型当前都以 hand-written `nn.Module` + entropy_model 直连形式落地（容器化/归口为 follow-up，见 §4）。

### 3.3 Family 1 内部的真实痛点（`script` 现状）

> 以下痛点描述 fork `script` 全量主干现状。**上游 migration 线已由 H+G 容器化重构解决**（删除 `ChannelSliceLatentCodec` + `SliceEntropyCompressionModel`，改用 upstream codec 嵌套）——详见 [exec-plan 设计依据段](../exec-plans/completed/codec-containerization-h-g-refactor.md#design-rationale)。

- `SliceEntropyCompressionModel._init_slice_entropy` 把 `widths = (224, 176, 128, 64)` 写死，TCM 不能用
- `ChannelSliceLatentCodec.forward` 用 `y.chunk(num_slices)`，CCA-main 不能用（要变长）
- `ChannelSliceLatentCodec` 没有 skip-most-recent support 模式，CCA-aux 不能用
- `ChannelSliceLatentCodec` 没有「跳过 LRP」开关，无 LRP 的变体不能用
- `_bases/dictionary_entropy.py` 与 `_bases/slice_entropy.py` 90% 重合——要么吸收为可选 hook，要么承认其作为兄弟基类

**Family 4 (WeConvene) 和 MambaIC 都已经在 fork `script` 调用 `_bases` 的 helper**（`slice_support_channels` / `lrp_support_channels` / `make_entropy_transform`），证明 widths 参数化已经不只是 Family 1 内部需求——抽象设计广泛地服务于多个 codec 类。

### 3.4 fork `script` 现存 17 个 codec 类完整分类

| Family | Codec 类 | 用于 | 循环 | 分布 | 关键特征 |
|---|---|---|---|---|---|
| 1 | `ChannelSliceLatentCodec` | STF, WACNN, TCM, DCAE, **SAAF**, CCA-main, CCA-aux, MambaVC | 1-pass per slice | Gaussian | 分离 mean/scale，可选 LRP |
| 2 | `ChannelGroupsLatentCodec` | ELIC, **GLIC**, **CMIC**, MLIC++（外层）| per-group → checkerboard 2-pass | Gaussian | mean+scale 联合（ELIC/GLIC/CMIC）；分离（MLIC++ via sibling leaf）；groups 大小列表 |
| 2 | `MultiContextCheckerboardLatentCodec`（sibling leaf，与 `CheckerboardLatentCodec` 同位）| MLIC++（MambaIC 候选评估后未复用）| per-slice anchor→nonanchor 2-pass | Gaussian | 双 EP head（anchor/nonanchor）+ 可插拔 spatial/channel intra-context + 可选 per-pass LRP |
| 2 | `MambaICLatentCodec` | MambaIC（**评估后保留 dedicated** —— 分离 mean/scale head + 跨层 slice 装配与 `MultiContextCheckerboardLatentCodec` 单 head 契约冲突）| per-slice anchor→nonanchor 2-pass | Gaussian | 分离 mean/scale + VSS (Mamba SSM) 上下文 |
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

**总：17 个 codec 类 = 7 功能 family + 4 wrapper/legacy/gain 桶**。FTIC 不在此 17 类中——它在 fork `script` 上以 `TCAEntropyModel` (`nn.Module`) + `GsnConditionalLocScaleShift`（**indexed Gaussian + integer mean shift trick，等价于 2-参数 `GaussianConditional`**）直接调用形式落地，所以既不在表中、也不算"新 codec 类"。**结构上属于 Family 1 channel-slice AR 的 transformer-context 变体**（见 §2.7、§3.1），容器化为 `ChannelGroupsLatentCodec(channel_context={...TCA adapter...}, latent_codec={LRPGaussianLatentCodec ×K})` 是 follow-up（见 §4）。

> **补充调研模型（Family 8/9 + ContextFormer/TBTC）均不在这 17 个 codec 类中**：ContextFormer / TIC / GLLMM / Informer / LBHIC / NIC / NVTC / TBTC 全部 **hand-written**（在 model `forward`/`compress`/`decompress` 里直连 entropy_model，不走 `LatentCodec`），所以既不在上表、也不算新 codec 类。其中：
> - **Legacy `RasterScanLatentCodec`** 已经是 Minnen2018 风格逐像素 mask-conv AR 的容器化封装——它是 **Family 8 的天然归口**（mbt2018/Cheng2020/Informer/TIC 的 forward loop 与它同构）；若日后要把 F8 容器化，复用/扩展 `RasterScanLatentCodec`（加 mixture-leaf、3D-mask、attention-context 变体）比新建更合理。
> - **Family 9（NVTC）无对应 codec 类**——非高斯 VQ 与 `LatentCodec`（围绕 `GaussianConditional`/`EntropyBottleneck` 设计）契约根本不兼容；若要纳入需另起 VQ-codec 体系。

> 注：MLIC++ 的 `MultiContextCheckerboardLatentCodec` 是 pr-mlicpp 把原 `MLICPlusPlusLatentCodec` monolith 拆出的 sibling leaf（外层用 upstream `ChannelGroupsLatentCodec`）——见 [`family2-roadmap.md`](../exec-plans/active/family2-roadmap.md) + [`mlicpp-latent-codec-refactor.md`](../exec-plans/completed/mlicpp-latent-codec-refactor.md)。

---

## 4. 容器化适用边界

> H+G 容器化的候选方向对比（A–H）、推荐方案、详细 API/state_dict 设计、LoC 估算已实施完毕，整体记录在 [`codec-containerization-h-g-refactor.md`](../exec-plans/completed/codec-containerization-h-g-refactor.md#design-rationale)「设计依据」段。本节只保留**哪些能容器化 / 哪些不能**的边界结论，供后续模型迁入决策。

**能容器化（沿 ELIC `HyperpriorLatentCodec` + `ChannelGroupsLatentCodec` pattern）**：
- **Family 1**（STF/WACNN/TCM/CCA-main/CCA-aux/DCAE/SAAF/MambaVC + FTIC transformer-context 变体）——共享 1-pass channel-slice forward loop，差异收敛在 5 个可参数化轴。STF/WACNN/TCM/CCA 已在上游容器化；DCAE/SAAF/MambaVC 的容器化为 follow-up。
- **Family 2 的 MambaIC**——仅 G（把 monolithic codec 包进 `HyperpriorLatentCodec`）。
- GLIC / CMIC 与 ELIC 一样**已经**直接用 upstream `ChannelGroupsLatentCodec`，无需进一步容器化。

**保持兄弟 codec 类 / 不容器化进 channel-slice 体系**：
- **Family 2 dedicated codec（MLIC++/MambaIC）**——联合 vs 分离 mean/scale 让 state_dict 不兼容，强行合并代价大于收益。MLIC++ 已拆出 sibling leaf `MultiContextCheckerboardLatentCodec`（外层仍复用 upstream 容器）。Direction F（三家合一 + `intra_slice_context_factory`）留给第 4 个 dedicated-codec 用户出现时再评估（ELIC/GLIC/CMIC 共用 upstream，不计入 dedupe 阈值）。
- **Family 3–7（HPCM/WeConvene/TinyLIC/ShiftLIC/Entroformer/RefBasedAR + ContextFormer）**——forward loop 各有不可参数化的架构差异（multi-resolution mask AR / wavelet 两分支 / 4+2+2+1 sub-stage / 全分辨率 Transformer / cascade GMM / spatio-channel Transformer），硬塞 channel-slice 容器只增间接性、无抽象收益。
- **Family 8（空间 raster-scan AR：TIC/GLLMM/Informer/LBHIC/NIC + mbt2018/Cheng2020 祖先）**——这是 channel-slice 出现**之前**的范式（逐像素/逐块空间 AR），与 channel-slice 容器正交。**潜在归口是 legacy `RasterScanLatentCodec`**（mbt2018 forward loop 与它同构），而非 `ChannelGroupsLatentCodec`；要容器化得给 `RasterScanLatentCodec` 加 mixture-leaf / 3D-mask / attention-context 变体——单独评估，本文不展开。
- **Family 9（NVTC，非高斯 VQ）**——与围绕 `GaussianConditional`/`EntropyBottleneck` 设计的 `LatentCodec` 体系根本不兼容（无连续 latent / 无 mean-scale）。不纳入；若要支持需另起 VQ-codec 体系。

**FTIC 容器化是可达的 follow-up（roadmap Phase 8，非 hard blocker）**——scope 切分三步：(a) `GsnConditionalLocScaleShift` 升格到 `compressai/entropy_models/`（或扩 `GaussianConditional` 加 indexed-CDF 模式）；(b) 写 `TCAChannelContext` adapter 把 TCA 输出的全 K-slice 参数按 slice 切给 `ChannelGroupsLatentCodec.channel_context`；(c) 替换 `compressai/models/ftic.py` 的 `forward / compress / decompress` 改成调 `latent_codec`。

**遗留 follow-up**（不在本调研 scope）：
- bitstream `compress`/`decompress` 仍 `NotImplementedError` 或缺失的模型：Entroformer / RefBasedAR / HPCM（首轮）+ **ContextFormer / LBHIC / NVTC（补充调研）**；**NIC 连 compress/decompress 方法都没有**（仅 forward / rate 估计）。
- 训练脚本侧对新 codec 接口的适配。

> **GMM leaf 的 fast-path**（Family 7 / 未来 GMM 模型相关）：RefBasedAR 的 K=3 GMM 走 `GaussianMixtureConditional`，其 bitstream 路径的瓶颈在 Python 端逐 symbol 构 CDF table。[`flash-gmm-integration-plan.md`](../exec-plans/active/flash-gmm-integration-plan.md) 计划把 FlashGMM 作为 `GaussianMixtureConditional` 的 C++ rANS fast backend 集成，并新增可复用 leaf `GaussianMixtureConditionalLatentCodec`（GMM 版的 `GaussianConditionalLatentCodec`，可供 checkerboard/channel-group 容器复用）——是 RefBasedAR bitstream follow-up 的前置基础设施。

---

## 5. 引用

- [`plan/exec-plans/completed/codec-containerization-h-g-refactor.md`](../exec-plans/completed/codec-containerization-h-g-refactor.md) — **H+G 容器化重构执行计划 + 设计依据**（候选方向、推荐方案、详细 API/state_dict 设计，原本属本文 §4–§10）
- [`plan/exec-plans/active/family2-roadmap.md`](../exec-plans/active/family2-roadmap.md) — Family 2 上游迁入路线图（MLIC++ sibling leaf 抽象、MambaIC dedicated 保留决策）
- [`plan/exec-plans/completed/mlicpp-latent-codec-refactor.md`](../exec-plans/completed/mlicpp-latent-codec-refactor.md) — MLIC++ `MultiContextCheckerboardLatentCodec` 拆解
- [`plan/design-docs/cca-cross-model-extension.md`](cca-cross-model-extension.md) — CCA 作为跨模型插件的方向（互补文档）
- [`plan/product-specs/lic-migration-roadmap.md`](../product-specs/lic-migration-roadmap.md) — 整体 LIC 模型迁入路线图（FTIC = Phase 8、CMIC = Phase 2）
- [`plan/exec-plans/active/flash-gmm-integration-plan.md`](../exec-plans/active/flash-gmm-integration-plan.md) — FlashGMM C++ rANS fast backend + `GaussianMixtureConditionalLatentCodec` leaf（Family 7 RefBasedAR GMM 路径的前置基础设施）
- `compressai/latent_codecs/{channel_slice,channel_groups,mambaic,weconvene,weconvene_support,multistage_checkerboard,transformer_ar,ref_autoregressive,hpcm,mlicpp,hyperprior,hyper,checkerboard,gaussian_conditional,entropy_bottleneck}.py` — fork `script` 上 17 个 codec 类
- `compressai/models/_bases/{slice_entropy,dictionary_entropy}.py`（fork `script`）— Family 1 共享基类
- `compressai/models/{stf,tcm,cca,sensetime,mlicpp,mambaic,mambavc,weconvene,tinylic,shiftlic,damo,hpcm,dcae,glic,cmic,ftic}.py` — Family 1–7 的 entropy head 实现
- `compressai/models/{contextformer,tic,gllmm,informer,lbhic,nic,nvtc,nvtc_support,tbtc}.py` — 补充调研的新增 entropy head（Family 6 变体 / Family 8 / Family 9 / F1 ChARM）；helper 层在 `compressai/layers/attn/{contextformer,cam}.py`（cam = TIC 因果注意力窗）、`compressai/layers/lic/{lbhic,nvtc}.py`
- `compressai/models/{google,waseda}.py` — Family 8 祖先 + Family 0 baseline（`JointAutoregressiveHierarchicalPriors` / `Cheng2020*` / `MeanScaleHyperprior` / `ScaleHyperprior`）
- `candidate/FTIC/models/{flic,tca,entropy_models}.py` — FTIC 原作者代码（§2.7 复核基础）
