# CompressAI LIC 迁移路线图

## 上游迁入进度（`Yiozolm/CompressAI` master）

> 本路线图是候选池总盘 + 推荐顺序；逐 PR 的细节状态在 `plan/exec-plans/` 各 plan 里。已合并到上游 master 的模型：
>
> - **STF / WACNN**（PR #3 系），**LIC_TCM / CCA**（pr-tcm-cca），**DCAE / SAAF**（pr-dcae-saaf-auxt），**MLIC ×4**（pr-mlicpp），**GLIC**（PR #4），**MambaIC / CMIC**（PR #5），**TinyLIC / ShiftLIC small/middle/large**（PR #6，2026-06-03，merge `2f453cb`，共享 `MultistageCheckerboardLatentCodec`），**WeConvene**（PR #7，2026-06-03，merge `0cbe7cf`，新 `WeChARMLatentCodec` + `[wavelet]`），**ICLR2024-FTIC**（PR #8，2026-06-03，merge `36b48b7`，新 `GsnConditionalLocScaleShift`），**InvCompress**（PR #9，2026-06-03，merge `43d47eb`，新 `[invcompress]`/FrEIA extra）。
> - **不迁入**：`2024-MambaVC`（仅 arXiv 预印本，未同行评审，见 §E.14 / Phase 11）。
> - **剩余候选**：无独立 model 待迁（WeConvene/FTIC/InvCompress 已于 PR #7/#8/#9 全部迁入）；`AuxT`（共享层，已随 pr-dcae-saaf-auxt 落地）、`CCA`（已落地）为方法/插件类,不单独注册 entry。

## 本轮勘察结论

- 当前图像模型的稳定落点在 `compressai/models/`，基础抽象在 `compressai/models/base.py`。
- 当前最现代、最接近候选实现的主线是 `compressai/models/sensetime.py` 中的 `Elic2022Official`：
  - 使用 `SimpleVAECompressionModel`
  - 使用 `HyperpriorLatentCodec` / `ChannelGroupsLatentCodec` / `CheckerboardLatentCodec`
  - 已经具备 checkerboard + uneven channel groups 的现代组织方式
- 可复用基础层主要在 `compressai/layers/`，其中已有：
  - `AttentionBlock`
  - `ResidualBlock*`
  - `CheckerboardMaskedConv2d`
  - `sequential_channel_ramp`
- 模型注册入口是 `compressai/registry/torch.py` 的 `@register_model`。
- zoo 暴露入口是 `compressai/zoo/__init__.py` 和 `compressai/zoo/image.py`。
- 现有 smoke test 主要集中在 `tests/test_models.py` 和 `tests/test_layers.py`。
- **候选池（共 16 个 model entry，分布在 14 个目录）**：
  - Elic2022 主线：`GLIC`、`CMIC`
  - 老式主流 baseline：`STF`（含 **WACNN 与 SymmetricalTransFormer 两个模型**）、`LIC_TCM`、`MLIC++`、`DCAE`、`SAAF`、`WeConvene`、`TinyLIC`、`ShiftLIC`（含 small / middle / large 三 variant，与 TinyLIC 共用 `MultistageCheckerboardLatentCodec`）
  - novelty 模块型：`ICLR2024-FTIC`（含自定义 entropy model）、`InvCompress`（含可逆层）
  - 方法 / 插件：`AuxT`、`CCA`
  - SSM 高风险：`MambaIC`、`2024-MambaVC`
- **候选里尚未在现仓存在、但被多处消费的共享层**：
  - Swin 家族（`WMSA` / `SwinBlock` / `ConvTransBlock` / `SWAtten` / `Win_noShift_Attention`）——`STF` / `WACNN` / `LIC_TCM` / `DCAE` / `SAAF` / `MambaVC` / `FTIC` 都自带了一份
  - Invertible 家族（`CouplingLayer` / `InvertibleConv1x1` / `SqueezeLayer`）——`InvCompress` 独占
  - 共享层抽取的收益面比单看 GLIC / CMIC 时大得多。

## candidate 分组

### A. Elic2022 主线（继承 `Elic2022Official` + `latent_codecs`）

1. `GLIC`
   - 直接继承 `Elic2022Official`
   - 已使用当前仓库的 `latent_codecs`
   - 主要新增 graph/wavelet 相关模块
   - 风险：`pywt`、graph 相关实现需要拆分
   - License：有

2. `CMIC`
   - 直接继承 `Elic2022Official`
   - 已使用当前仓库的 `latent_codecs`，候选中的 `FixedCheckerboardLatentCodec` 可直接改为现有 `CheckerboardLatentCodec`
   - 与 `GLIC` 共享 `GatedTransformCNN` / `OLP` / `WLS` / `iWLS`
   - 风险：`mamba_ssm`、`basicsr`、wavelet 依赖
   - License：**无**

### B. 主流老式 baseline（老式 `CompressionModel`，但结构良好、社区常用）

3. `STF` (Symmetrical Transformer, CVPR 2022)
   - `stf.py` 787 行（`SymmetricalTransFormer`）+ `cnn.py` 336 行（`WACNN`）
   - 依赖 `timm`；依赖 `Win_noShift_Attention`——**当前 compressai 不存在**，需随 STF 一并引入
   - 候选目录里自带一份 `compressai/` 旧 fork，仅 `models/{stf,cnn}.py` 有迁移价值，其它要忽略
   - License：Apache 2.0

4. `LIC_TCM` (Tian et al., CVPR 2023)
   - `tcm.py` 626 行（`TCM`）
   - Swin 风格 `WMSA` / `Block` / `ConvTransBlock` / `SwinBlock`，与 `DCAE` / `SAAF` / `MambaVC` 的 Swin 层**高度重合**
   - License：MIT

5. `MLIC++` (MLICPlusPlus)
   - `mlicpp.py` 389 行 + `modules/transform/` 498 行（analysis/synthesis/entropy/context/quantization 分文件）
   - **候选里已经是"一个 model + 拆好的 modules"**，天然对齐本 roadmap 的拆分原则，是成本最低的老式收敛样板
   - 核心贡献 `LinearGlobalInter/IntraContext` 可考虑实现为 `latent_codec` 变体
   - 依赖 `timm` / `einops`
   - License：Apache 2.0

6. `DCAE`
   - 仍是老式 `CompressionModel` 手写 `forward/compress/decompress`
   - 内部重复实现了 buffer update / bottleneck block / conv helper
   - 可迁移，但需要显式向当前风格收敛
   - License：有

7. `SAAF`
   - 与 `DCAE` 共享较多结构
   - 还引入 Adaptive Frequency / sparse attention / diffusion regularizer
   - 建议放在 `DCAE` 之后
   - License：有

8. `WeConvene`
   - wavelet-domain 卷积 + 双 entropy 分支
   - 属于老式大单文件实现，拆分成本不低

### C. 带有独立贡献点的模型（novelty 在 entropy/invertible 模块）

9. `ICLR2024-FTIC` (FrequencyAwareTransFormer)
   - `flic.py` 584 行 + `tca.py` 225 行 + `entropy_models.py` 450 行
   - 两个真正的新贡献：`GsnConditionalLocScaleShift`（**indexed Gaussian + integer mean shift trick**——分布上等价于 `GaussianConditional`，只是用了一组预算好的 CDF 表 + 整数 mean 移位优化；2026-05-11 复核更正，原误判为 loc/scale/shift 三参数 Gaussian）+ `TCA_EntropyModel`（masked-channel transformer，结构上是 Family 1 channel-slice AR 的 transformer-context 变体，详见 `plan/design-docs/channel-slice-codec-redesign.md` §2.7）
   - 使用外部 `range_coder` 做算术编码（当前 compressai 用自带 ANS），**新依赖**
   - License：**无**
   - 迁移路径：可在 `compressai/entropy_models/` 新增 `IndexedGaussianConditional`（或扩 `GaussianConditional` 加 indexed-CDF 模式），TCA 写一个 channel-context adapter 后用 `ChannelGroupsLatentCodec` 容器包装；不再视为接口对齐难题

10. `InvCompress` (Xie et al., ACM MM 2021)
    - `ours.py` 140 行（`InvCompress(Cheng2020Anchor)`）+ `our_utils.py` 383 行（`CouplingLayer` / `InvertibleConv1x1` / `SqueezeLayer` / `DenseBlock`）
    - **陷阱**：候选目录里还有 `priors.py` / `waseda.py` 重新定义了一份旧版 `CompressionModel` / `ScaleHyperprior` / `JointAutoregressiveHierarchicalPriors`，**绝对不能整文件搬**，只取可逆相关层
    - 正确做法：从现仓的 `compressai/models/waseda.py::Cheng2020Anchor` 继承；把可逆层放入 `compressai/layers/lic/invertible.py`
    - License：Apache 2.0

### D. 方法/插件（不单独注册 model entry）

11. `AuxT`
    - 更适合作为共享模块来源
    - `WLS` / `iWLS` / `OLP` / `GatedTransformCNN` 已被 `CMIC`、`GLIC` 复用
    - 落点：仅 Phase 0 共享层，不走 `@register_model`

12. `CCA`
    - 更偏 loss / auxiliary entropy model 方法
    - 落点：`compressai/losses/` 或 `compressai/entropy_models/` 扩展，不走 `@register_model`

### E. 高风险候选（SSM / CUDA kernel）

13. `MambaIC`
    - 依赖 `SS2D` / Triton / VMamba 风格 selective scan
    - 文件巨大，外部依赖重
    - License：**无**

14. `2024-MambaVC` — ❌ **不迁入（won't-do，2026-06-03 决策）**
    - **原因：仅 arXiv 预印本，未经同行评审正式发表**，不纳入上游迁入范围
    - `MambaVC.py` **1618 行**（最大单文件候选），`csm_triton.py` 338 行
    - 同时依赖 `selective_scan_cuda_oflex` / `selective_scan_cuda_core` / `selective_scan_cuda` 三个 CUDA 扩展（代码里自带 try/except 回退）
    - License：**无**
    - 与 `MambaIC` 的 SSM 路径高度重叠；MambaIC（CVPR 2025）已迁入并落地共享 `compressai/layers/ssm/`，SSM 家族需求已被覆盖，MambaVC 不再有迁入价值

## 推荐迁移顺序

### Phase 0：先整理共享层 + latent codec + 依赖策略

目标：不要把 candidate 代码整块复制进 `compressai/models/`。在动第一个模型之前把基础设施、latent codec 差异、依赖分档这三件事一次性收敛。

#### 0.1 共享层抽取

优先抽出的共享件：

- `ResidualBottleneckBlock`
  - 当前在 `compressai/models/sensetime.py:549`
  - `DCAE` / `SAAF` 也各自重复实现了一份
- `LayerNorm2d`
- `GatedTransformCNN`
- `OLP`
- `WLS`
- `iWLS`
- **Swin 家族共享层**（新增）
  - `WMSA`（Window Multi-head Self-Attention，不带 shift）
  - `SwinBlock` / `Block`（含 mlp + residual）
  - `ConvTransBlock`（Conv + Trans 并联）
  - `SWAtten`（`AttentionBlock` + swin；`LIC_TCM` / `DCAE` / `MambaVC` 各自重复写了一遍）
  - `Win_noShift_Attention`（`STF::WACNN` 显式引用，但**当前 compressai 不存在**，随 STF 一并抽出）
  - 主要消费者：`STF` / `WACNN` / `LIC_TCM` / `DCAE` / `SAAF` / `MambaVC` / `FTIC`
- **Invertible 家族共享层**（新增，`InvCompress` 专用）
  - `CouplingLayer` / `InvertibleConv1x1` / `SqueezeLayer` / `DenseBlock`

建议做法：

- 在 `compressai/layers/` 下按主题拆 sub-package：`attn/`（Swin 家族）、`graph/`（GLIC）、`ssm/`（Mamba/VSS）、`wave/`（DWT/IDWT），`lic/` 留给 LIC-niche 杂项（`blocks.py` / `dcae.py` / `saaf.py` / `invertible.py` / `stf.py` / `mlic/`）
- `compressai/layers/__init__.py` 顶层 `from .{attn,graph,ssm,wave,lic} import *`，保持 `from compressai.layers import X` 既有写法不破
- 通用 `conv` / `deconv` 工具的规范位置已落在 `compressai/layers/layers.py`（与 `conv1x1` / `conv3x3` 同处）；`compressai/models/utils.py` 仅做 backward-compat re-export
- `compressai/models/sensetime.py`、未来新模型都从共享层导入
- 历史：原计划全部塞进 `compressai/layers/lic/` 单包；2026-04 已按主题再分拆，详见 candidate/TODO.md "Shared Foundations"

#### 0.2 Latent codec / entropy model 差异补齐

- `CMIC_AuxT` 候选代码里使用 `FixedCheckerboardLatentCodec`，但该类只覆盖 `_y_ctx_zero`；当前仓库 `CheckerboardLatentCodec` 已等价地对 anchor context 置零
  - 推荐做法：不新增 alias / 不扩展 API；迁移 `CMIC` 时直接把 `FixedCheckerboardLatentCodec` 改为 `CheckerboardLatentCodec`
  - Phase 2 CMIC 不再把该类名视为硬阻塞项
- `FTIC` 自带 `GsnConditionalLocScaleShift`（**indexed Gaussian + integer mean shift trick，等价于 2-参数 `GaussianConditional`**；2026-05-11 复核更正）和 `TCA_EntropyModel`（masked-channel transformer）
  - 推荐做法：在 `compressai/entropy_models/` 下新增 `IndexedGaussianConditional`（或在现有 `GaussianConditional` 上加 indexed-CDF 模式），`TCA` 在 `compressai/models/ftic.py` 内部用 + 写一个 channel-context adapter 后用 `ChannelGroupsLatentCodec` 容器（结构上是 Family 1 channel-slice AR 的 transformer-context 变体，详见 `plan/design-docs/channel-slice-codec-redesign.md` §2.7、§3.1）
  - Phase 8 FTIC 的工作项（**已不再视为硬阻塞**：分布兼容、context 模式归 Family 1）
- **`ChannelSliceLatentCodec`（已抽出）**：Minnen2020-style 通道自回归熵模型在 STF / WACNN / MLIC++ / 后续 LIC_TCM / DCAE / SAAF 里反复出现。已在 `compressai/latent_codecs/channel_slice.py` 落地：
  - 等大切片 + 双头 (`cc_mean_transforms` / `cc_scale_transforms`) + `max_support_slices` 截断 + 内置 LRP + 单 RANS string
  - 与现有 `ChannelGroupsLatentCodec`（不等大 group + 外部 channel_context + 委托内层 codec）互为亲缘类，docstring 互相点明
  - STF / WACNN 已迁过去（`compressai/models/stf_support.py::SliceEntropyCompressionModel` 退化为薄壳）
  - 后续凡是"按通道等大切片 + 双头 EP + LRP"的模型应直接复用，**不要再手写 `chunk(num_slices)` 循环**
- `MLIC++` 的 `LinearGlobalInter/IntraContext` 在 `latent_codec` 体系下更自然，可作为新的 context-aware latent codec 候选；当前先以"内部模块"落地，未来重构时建议把"双头 anchor/nonanchor checkerboard + 多 context + 双 LRP"抽成独立内层 codec，再用 `ChannelSliceLatentCodec` / `ChannelGroupsLatentCodec` 做外层切片循环

#### 0.3 依赖三档分级

不再笼统说"统一可选依赖策略"，分三档处理：

- **必装**：`einops`（已在 `pyproject.toml`）；建议把 `timm` 升格为必装，因为 `DCAE` / `SAAF` / `STF` / `WACNN` / `LIC_TCM` / `MLIC++` / `FTIC` / `MambaVC` 都硬依赖 `timm.models.layers` 的 `trunc_normal_` / `DropPath` / `to_2tuple`
- **软可选**：
  - `pytorch_wavelets`（PyPI）：封装在 `compressai/layers/wave/wavelet.py` 里 lazy import，缺失时 `GLIC` / `CMIC` / `WeConvene` 注册跳过；`pywt` 由 `pytorch_wavelets` 作为下游依赖自动带入，compressai 不直接 import
- **严格可选（缺失则跳过 `@register_model`）**：
  - `mamba_ssm`（PyPI，Linux+CUDA wheel）：`CMIC` + `compressai/layers/ssm/ssm.py` 的 1D fallback
  - `selective_scan_cuda_oflex` / `_core` / `_cuda` / `triton`：`MambaIC` / `MambaVC` 的加速路径（非必须，`ssm.py` 有纯 PyTorch `selective_scan_ref` fallback）
  - `FrEIA`（PyPI）：`InvCompress` 的可逆层
  - `range_coder`（PyPI）：`FTIC` 的 `compress/decompress`；`forward` / 训练 smoke test 不需要，缺失时仅跳过编解码 smoke test
- **可直接内联替换、不作为依赖**：`basicsr`——只用到 `to_2tuple` / `trunc_normal_`，迁移时改为 `timm.layers` 或内联实现

文档构建依赖：

- `[doc]` extras 已升到 `sphinx>=7.4,<9` + `sphinx-book-theme>=1.1`（原 `sphinx==4.3.0` 在 Python 3.10+ 装不上），并加 `python_version >= '3.9'` marker 让 Py 3.8 解析跳过；本机 `uv sync --group doc` + `uv run sphinx-build -b html docs/source docs/_build/html` 已通过

验收标志：在没装 `pytorch_wavelets` / `mamba_ssm` / `selective_scan_cuda*` / `triton` / `FrEIA` / `range_coder` 的干净环境里 `import compressai` 不报错，`compressai.models` / `compressai.zoo` 里对应模型缺席但其它模型可用。

#### 0.4 第三方库复用策略（"能用现成的绝不自写"）

调研结论：以下层家族在 PyPI 已有可直接调用的成熟库，应取代候选里各自重写的版本。

| 层家族 | 候选现状 | 推荐 | 处理方式 |
|---|---|---|---|
| Swin 辅助（Mlp / DropPath / PatchEmbed / trunc_normal_ / to_2tuple / window_partition） | 全部候选已在用 `timm.layers` 的子集 | **`timm.layers`** | **复用**，作为必装依赖（0.3 已规定） |
| Swin 主块（`WMSA` / `SwinBlock` / `Win_noShift_Attention`） | 7 份重写 | `timm.models.swin_transformer.SwinTransformerBlock` 锁死 `input_resolution`，不能用；`torchvision` 的 `WindowAttention` 非公开 API | **自写**（`compressai/layers/attn/swin.py` + `swin_attention.py`），支持动态分辨率；`Mlp` / `DropPath` 等从 `timm.layers` 导 |
| Wavelet DWT/IDWT | 裸 `pywt` + 手写卷积（无 GPU/autograd） | **`pytorch_wavelets`**（`DWTForward` / `DWTInverse`，PyPI，GPU + autograd） | **复用**，`compressai/layers/wave/wavelet.py` 基于 `pytorch_wavelets` 封装 `WLS` / `iWLS` / `DWT2D` / `IDWT2D`；`pywt` 不再被 compressai 直接 import，只通过 `pytorch_wavelets` 依赖链带入 |
| Invertible（CouplingLayer / 1x1 conv / Squeeze） | InvCompress 自写 ~380 行 | **`FrEIA.modules`**（vislearn/FrEIA，PyPI，GLOW/RNVP/NICE 耦合块 + 可逆 1x1 conv） | **复用**，`compressai/layers/lic/invertible.py` 只保留 `EnhBlock` / `DenseBlock`（InvCompress 独有）的 dense 封装，其它从 FrEIA 组合 |
| Mamba 1D selective scan | CMIC 直接用 `mamba_ssm.selective_scan_fn` | **`mamba_ssm`**（官方，PyPI 有 Linux+CUDA wheel） | **复用**（严格可选） |
| Mamba 2D selective scan (`SS2D`) | MambaIC / MambaVC 各 vendor 一份 VMamba kernel | 无干净的 PyPI 包；VMamba / Mamba2D / MambaVision 不适合作为依赖 | **vendor**（`compressai/layers/ssm/ssm.py` + `ssm_ops.py`），回退链：`selective_scan_cuda*` → `mamba_ssm.selective_scan_fn` → 纯 PyTorch `selective_scan_ref` |
| Graph attention / neighborhood agg | GLIC 自写 ~500 行，操作的是特征图上的局部邻域 | `torch_geometric` / `DGL` 面向稀疏大图、依赖树重，收益不对称 | **自写**（`compressai/layers/graph/{graph,graph_gfa,graph_ops}.py`） |
| 算术编码 | FTIC 用 `range_coder` | `range_coder`（PyPI） | **复用**（严格可选），仅 `compress/decompress` 需要；现仓 ANS 不受影响 |

新增依赖落到 `pyproject.toml` 的位置：

- **必装**：`timm`、`einops`
- **软可选**：`pytorch_wavelets`（缺失则 `GLIC` / `CMIC` / `WeConvene` 注册跳过；导入 `compressai` 不受影响）
- **严格可选**：`mamba_ssm`（CMIC / SSM fallback）、`FrEIA`（InvCompress）、`range_coder`（FTIC 编解码）、`selective_scan_cuda*` / `triton`（MambaIC / MambaVC CUDA 加速）

不加的依赖：`torch_geometric` / `DGL`（GLIC 用不上那么重）、`VMamba` / `MambaVision`（非 PyPI 干净包，直接 vendor 更干净）。

### Phase 1：迁移 `GLIC`

理由：

- 主干沿 `Elic2022Official` + `latent_codecs`，不需要先重写整套 entropy interface
- 可以顺带沉淀 graph / wavelet 相关层的拆分模板

体量提醒：不要把 GLIC 视作"薄包装 Elic"。候选 `glic_model/` 里 `layers/graph_*.py`（~500 行）+ `wavelet_layers.py` + `utils/graph_*.py` + `utils/wavelet.py` 合计 ~800 行的 graph / wavelet 子系统在当前仓库无任何对应实现，Phase 1 实际工作接近"Elic 主干 + 新增一个子系统"，测试与 review 预算按后者计。

目标落点：

- `compressai/layers/lic/`：`graph.py`、`wavelet.py`、`gating.py` 等共享层
- `compressai/models/`：`glic.py`
- `compressai/models/__init__.py`：导出
- `compressai/zoo/image.py`：注册 `glic` 架构，登记 `model_urls` 占位（即使暂无预训练权重也先留位）
- `tests/test_models.py`：smoke + 数值等价测试（见"测试与验收"）

### Phase 2：迁移 `CMIC`

理由：

- 与 `GLIC` 共用较多 AuxT-style 组件
- 同样走 `Elic2022Official` 路线

前置依赖：

- Phase 0.2 已确认无需新增 `FixedCheckerboardLatentCodec`；`CMIC_AuxT` 的 `latent_codec` 构造直接复用 `CheckerboardLatentCodec`

额外关注：

- `mamba_ssm` 按 Phase 0.3"严格可选"策略处理：缺失则跳过 `CMIC` 的 `@register_model`
- `basicsr` 直接在迁移时替换为 `timm.layers` / 内联，**不引入**为依赖
- `pywt` 按软可选处理

### Phase 3：迁移 `MLIC++`

理由：

- 候选代码**已经拆好**了 `modules/transform/{analysis,synthesis,context,entropy,quantization}.py` 和 `modules/layers/{attention,conv,res_blk}.py`，和本 roadmap 的落地风格天然对齐
- 依赖最轻（`timm` + `einops`），无 SSM / wavelet / graph 依赖
- 选它作为"老式 `CompressionModel` 收敛到现仓风格"的第一个样板，比 `DCAE` 成本低得多

迁移原则：

- `compressai/models/mlicpp.py` 只保留模型类 + `from_state_dict` + registry glue
- `modules/transform/*` 内容搬入 `compressai/layers/lic/mlic/` 或按能力拆（context/entropy/transform）
- `LinearGlobalInter/IntraContext` 先作为内部模块；未来 latent codec 再重构时考虑把"双头 anchor/nonanchor checkerboard + 多 context + 双 LRP"抽成独立内层 codec，再用 `ChannelGroupsLatentCodec` / `ChannelSliceLatentCodec` 统一外层切片循环
- `utils.func.update_registered_buffers` / `utils.ckbd.*` 替换为现仓已有实现

### Phase 4：迁移 `STF`（两个模型）

`STF` 目录同时提供 **两个** 独立模型，必须分别迁移和注册：

- **4a. `WACNN`**（`cnn.py`，~336 行）
  - 纯卷积（+ `Win_noShift_Attention`）baseline
  - 依赖 `Win_noShift_Attention`——**当前 compressai 不存在**，Phase 0.1 的 Swin 家族共享层已把它列为需一并抽出的目标
  - 作为 STF 迁移里"依赖最少"的一个，先做
  - 落点：`compressai/models/stf.py::WACNN` + `compressai/zoo/image.py` 注册 `wacnn`

- **4b. `SymmetricalTransFormer`**（`stf.py`，~787 行）
  - 带 Swin 注意力的主 baseline；`PatchMerging` / `PatchSplit` / `BasicLayer` / `SwinTransformerBlock`
  - 复用 4a 抽出的 Swin 层
  - 落点：`compressai/models/stf.py::SymmetricalTransFormer` + `compressai/zoo/image.py` 注册 `stf`

注意事项：

- 候选目录自带旧版 `compressai/` fork，**只抽 `models/{stf,cnn}.py`**，其它（`setup.py` / `third_party/` / 旧版 `compressai/layers`）全部忽略
- STF 的 `base.py::CompressionModel` 是老版本重定义，直接改为继承现仓 `compressai.models.base.CompressionModel`
- **共享 channel-slice 熵模型**：WACNN 与 SymmetricalTransFormer 共用 `compressai/models/stf_support.py::SliceEntropyCompressionModel`，它本身是 `compressai/latent_codecs/channel_slice.py::ChannelSliceLatentCodec` 的薄壳。`cc_mean_transforms` / `cc_scale_transforms` / `lrp_transforms` / `gaussian_conditional` 都活在 codec 内部，state_dict 键路径为 `latent_codec.cc_mean_transforms.*` 等。后续 LIC_TCM / DCAE / SAAF / MLIC++ 凡是"按通道等大切片 + 双头 EP + LRP"的也走这条路

### Phase 5：迁移 `LIC_TCM`

理由：

- Swin 家族共享层在 Phase 4 已抽出（现位于 `compressai/layers/attn/`），`LIC_TCM` 的 `WMSA` / `Block` / `ConvTransBlock` / `SwinBlock` 直接复用
- `TCM` 主体只剩一个模型类 + 几个 Conv/Trans 并联结构，文件会显著瘦身

### Phase 6：迁移 `DCAE`

理由：

- Swin 层、`ResidualBottleneckBlock`、`timm` 相关依赖在 Phase 0 + Phase 4-5 已经沉淀好
- `DCAE` 的独特贡献（`MutiScaleDictionaryCrossAttentionGLU` 等）拆进 `compressai/layers/lic/dcae.py`

迁移原则：

- 不沿用候选里的重复 helper（`ResidualBottleneckBlock` 改从 `compressai/layers/lic/blocks.py` 取；`conv` / `deconv` 改从 `compressai/layers` 取，`compressai/models/utils` 仍可用但仅是 re-export）
- 统一复用 `compressai.models.base` 和 `compressai.models.utils`
- 尽量拆成：共享层 / entropy/context block / model 主体

### Phase 7：迁移 `SAAF`

理由：

- 可以直接复用 `DCAE` 迁移后沉淀的 bottleneck / attention / context 组件
- 再叠加 frequency-related 模块

### Phase 8：迁移 `FTIC` ✅ 已迁入（PR #8）

> **已完成 2026-06-03，PR #8，merge `36b48b7`**（详见 `plan/exec-plans/completed/ftic-integration.md`）。新移位高斯 `GsnConditionalLocScaleShift` 入 `compressai/entropy_models/`；复用 master `pad_to_window_multiple`（`swin.py`,`swin_attention.py` 未迁）；convert-to-examples + deep-import-only；T-CA 熵模型内联在 `models/ftic.py`。**未用 `range_coder`**（自带 ANS 即可）。

前置依赖（历史记录）:

- Phase 0.2 的 `GsnConditionalLocScaleShift` 合入 `compressai/entropy_models/`（或作为新类型）
- `range_coder` 按严格可选处理：`forward` smoke 不需要，`compress/decompress` smoke 在缺失时跳过

迁移要点：

- `TCA_EntropyModel` 进 `compressai/layers/attn/`（与 Swin 主块并列）或新建 `compressai/layers/attn/tca.py`
- `FrequencyAwareTransFormer` 本体进 `compressai/models/ftic.py`
- License 缺失，合入前需要先确认

### Phase 9：迁移 `InvCompress` ✅ 已迁入（PR #9）

> **已完成 2026-06-03，PR #9，merge `43d47eb`**（详见 `plan/exec-plans/completed/invcompress-integration.md`）。可逆层用 `FrEIA.modules`（`GLOWCouplingBlock` / `Fixed1x1Conv` / `IRevNetDownsampling`）组装,全部内联在 `compressai/models/invcompress.py`；`InvCompress(Cheng2020Anchor)` 保留继承；新增 `[invcompress]`/FrEIA extra（CI `--all-extras` 真跑）；convert-to-examples + deep-import-only。修了 script 版 `_maybe_register_model` 配套 `TypeVar(..., bound=type[nn.Module])` 在 py3.8 import 崩的问题（改纯 `@register_model`）。

迁移要点（历史记录）:

- **坑**：候选目录下 `priors.py` / `waseda.py` 是旧版 `CompressionModel` / `ScaleHyperprior` / `JointAutoregressiveHierarchicalPriors` 的整体重写，**绝对不搬**。只搬 `ours.py::InvCompress` 和 `our_utils.py` 里 InvCompress 独有的部分
- 可逆层**优先复用 `FrEIA.modules`**：`GLOWCouplingBlock` / `RNVPCouplingBlock` / `InvertibleConv1x1` / `PermuteRandom`；只有 `EnhBlock`（InvCompress 独有的 dense-conv 封装）进 `compressai/layers/lic/invertible.py`
- 主体 `InvCompress(Cheng2020Anchor)` 保留原继承关系，使用现仓 `compressai/models/waseda.py::Cheng2020Anchor`
- `FrEIA` 走严格可选：缺失则 `InvCompress` 跳过 `@register_model`

### Phase 10：评估 `WeConvene`、`CCA`、`AuxT`

- `WeConvene` ✅ **已迁入（PR #7，2026-06-03，merge `0cbe7cf`）**：复用 Phase 0 抽出的 `compressai/layers/wave/`（`pytorch_wavelets`,`[wavelet]` extra）;新增 `WeChARMLatentCodec`（wavelet-domain channel-AR）入 `compressai/latent_codecs/`;详见 `plan/exec-plans/completed/weconvene-integration.md`
- `CCA`：定位为 **loss / auxiliary entropy 方法**而非独立模型 zoo entry；落点 `compressai/losses/` 或 `compressai/entropy_models/` 的扩展，不走 `@register_model`
- `AuxT`：定位为 **共享模块来源**；`WLS` / `iWLS` / `OLP` / `GatedTransformCNN` 已在 Phase 0 抽入 `compressai/layers/lic/`，不单独注册 model entry

### Phase 11：`MambaIC` ✅ 已迁入；`MambaVC` ❌ 不迁入

**MambaIC**（CVPR 2025）已于 2026-06-03 迁入 upstream（Family-2 PR #5），共享层 `compressai/layers/ssm/` 已落地（`SS2D` / `VSSBlock` / `selective_scan` 三档回退）。

**MambaVC**（2024，arXiv 预印本）**决策不迁入**（2026-06-03）：
- **核心原因**：仅 arXiv 预印本，未经同行评审正式发表，不纳入上游迁入范围
- SSM 家族需求已由 MambaIC 落地的 `compressai/layers/ssm/` 覆盖，MambaVC 与其 selective-scan 路径高度重叠，无额外迁入价值
- 1618 行最大单文件 + 三个 CUDA 扩展依赖 + License 缺失，成本/收益不成比例

## 代码组织建议

### 共享层拆分原则

- 单文件控制在 200-400 行
- 按能力拆，不按论文拆
- 候选里的超大文件不要原样搬运
- LIC-niche 层走 `compressai/layers/lic/` 子包，不要平铺进 `compressai/layers/` 根

建议结构：

- `compressai/layers/layers.py`
  - 通用 `conv` / `deconv` / `conv1x1` / `conv3x3` / `subpel_conv3x3` 工具与 `MaskedConv2d` / `CheckerboardMaskedConv2d` / `ResidualBlock*` / `AttentionBlock` 等历史层（`models/utils.py` 仅 re-export `conv` / `deconv`）
- `compressai/layers/lic/blocks.py`
  - `ResidualBottleneckBlock`
  - `LayerNorm2d`
  - `GatedTransformCNN`
  - `OLP`（统一定义；`wave/wavelet.py` / `lic/saaf.py` 跨包引用，不再各自重写）
  - `GatedFFN` / `DepthwiseConv5x5`
- `compressai/layers/attn/swin.py` + `compressai/layers/attn/swin_attention.py`
  - `WMSA` / `SwinBlock` / `Block` / `ConvTransBlock` / `SWAtten`
  - `WinNoShiftAttention`（alias `Win_noShift_Attention`）
  - `PatchMerging` / `PatchSplit`（STF 专用）
  - `WindowAttention` / `window_partition` / `window_reverse` / `build_window_attention_mask`
  - **辅助工具从 `timm.layers` 导**（`Mlp` / `DropPath` / `PatchEmbed` / `trunc_normal_` / `to_2tuple`），不重复实现
  - **主块自写**（`timm.models.swin_transformer.SwinTransformerBlock` 锁死 `input_resolution`，不能复用）
  - 供 `STF` / `WACNN` / `LIC_TCM` / `DCAE` / `SAAF` / `MambaVC` / `FTIC` 共同消费
- `compressai/layers/wave/wavelet.py`
  - `WLS` / `iWLS` / `DWT2D` / `IDWT2D` 封装；底层调 **`pytorch_wavelets.DWTForward` / `DWTInverse`**（PyPI，GPU + autograd），不再手搓 `pywt` 卷积
- `compressai/layers/graph/{graph,graph_gfa,graph_ops}.py`
  - `GraphAttentionLayer` / `GFA` / `MGB` / `GraphLayerStack` / `FeatureReshape` / `FeatureRestore`
  - 自写（`torch_geometric` / `DGL` 不引入）
- `compressai/layers/lic/invertible.py`
  - `EnhBlock` / `DenseBlock`（InvCompress 独有）
  - `CouplingLayer` / `InvertibleConv1x1` / `SqueezeLayer` 通过 **`FrEIA.modules`** 包装层组装，不重写
- `compressai/layers/ssm/{ssm,ssm_ops}.py`
  - `SS2D` / `VSSBlock` + selective-scan 封装；回退链：`selective_scan_cuda_oflex` / `_core` / `_cuda` → `mamba_ssm.selective_scan_fn` → 纯 PyTorch `selective_scan_ref`
  - 仅当决定接纳 `CMIC` / `MambaIC` / `MambaVC` 时维护 vendor 实现（无干净 PyPI 包）
- `compressai/layers/lic/dcae.py` / `saaf.py` / `stf.py` / `mlic/`
  - 各模型独有的中重型组件落点，不与上面通用子包混在一起
- `compressai/layers/__init__.py`
  - 顶层 `from .{attn,graph,ssm,wave,lic} import *`，保持 `from compressai.layers import X` 既有写法不破

### 模型文件拆分原则

- 每个 model 文件只保留：
  - model class
  - `from_state_dict`
  - 少量 glue code
- 大块 building blocks 尽量下沉到 `compressai/layers/`

## 测试与验收

最低验收线：

1. `forward` smoke test
2. 输出字典包含 `x_hat`、`likelihoods`
3. `x_hat.shape == x.shape`
4. `from_state_dict` 可恢复
5. 如模型支持熵编解码，再补 `compress/decompress` smoke test
6. **数值等价测试**：同一 seed 构造权重（或加载候选 checkpoint），对同一输入分别跑候选实现和迁移后实现，`x_hat` / `likelihoods` 用 `torch.allclose` 在合理 tolerance 下对齐。这是"语义不变"的硬证据，不能只靠 shape/key 检查。

建议补充：

- 可选依赖缺失时的 import test：在不装 `pywt` / `mamba_ssm` / `triton` 的环境里，`import compressai` / `import compressai.models` / `import compressai.zoo` 不报错；对应模型从 registry 缺席
- registry test：已注册模型名能通过 `compressai.zoo` 的公开入口解析并构造

## Zoo / Registry 暴露清单

每迁移一个模型，除了 `@register_model`，还要同步：

- `compressai/zoo/__init__.py`：显式 re-export 模型构造函数
- `compressai/zoo/image.py`：
  - `model_architectures` 增加该模型的 factory
  - `model_urls` 预留占位（即使暂无公开权重，也保留 key，方便后续填 URL）
  - 如果候选提供了 quality → 配置的映射表，一并搬运
- 文档：在对应 `docs/zoo.rst`（如存在）或 `README` 表格里补一行模型名

## 当前主要风险

- License 缺失：`CMIC` / `MambaIC` / `FTIC`（均已迁入；合入前已确认按既定政策不阻塞，`MambaVC` 已剔除）
- **SSM 路径重叠**：`MambaIC` 和 `MambaVC` 有大量重叠的 selective-scan 核心，应合并进 `compressai/layers/ssm/ssm.py` 后再分别包 model 类，避免两套 kernel 胶水代码
- `MambaIC` / `MambaVC` 依赖 Triton / `selective_scan_cuda*` kernel，维护成本最高
- `CMIC` 依赖 `mamba_ssm.selective_scan_fn`，没有纯 PyTorch fallback；`basicsr` 的依赖仅是 `to_2tuple`/`trunc_normal_`，可直接替换掉，不应错误地视为核心依赖
- `Win_noShift_Attention` 当前仓库不存在，是 STF 的硬阻塞项；`FixedCheckerboardLatentCodec` 不新增，CMIC 迁移时改用 `CheckerboardLatentCodec`；`GsnConditionalLocScaleShift` 是 FTIC 的工作项但**不再视为硬阻塞**（实质是 indexed Gaussian + integer mean shift trick，等价于 `GaussianConditional`，2026-05-11 复核更正——见 §C.9 / §兼容性 FTIC 段）
- `InvCompress` 候选目录自带老版 `CompressionModel` / `ScaleHyperprior` 重定义，**禁止整包搬运**
- 多个候选文件超过 600 行（`MambaVC.py` 1618、`WeConvene/tcm_wave_...py` 1074、`MambaIC.py` 906、`dcae.py` 827、`stf.py` 787、`VSS_module.py` 735、`cmic_utils.py` 682、`saaf.py` 639、`LIC_TCM/tcm.py` 626、`AuxT/tcm.py` 626、`CCA/vae.py` 605），必须主动拆分，不能直接复制
- `AuxT`、`CCA` 的落点已在本 roadmap 明确（前者 shared-layers-only，后者 loss/aux-entropy），需在 PR 时坚持该定位，避免偷偷变成独立 model entry

## 我建议的下一步

1. Phase 0：共享层抽取（blocks + swin + invertible）+ `GsnConditionalLocScaleShift` 合入 + `FixedCheckerboardLatentCodec` 改名策略确认 + 依赖三档落地（`timm` 升格必装、`pywt` 软可选、`mamba_ssm` / `selective_scan_cuda*` / `triton` / `range_coder` 严格可选）
2. Elic 主线第一批：`GLIC`（Phase 1）→ `CMIC`（Phase 2）
3. 老式 baseline 收敛：`MLIC++`（Phase 3，最轻）→ `WACNN`（4a）→ `SymmetricalTransFormer`（4b）→ `LIC_TCM`（Phase 5）
4. 旧风格复杂模型：`DCAE`（Phase 6）→ `SAAF`（Phase 7）
5. novelty 模型：`FTIC` ✅ 已迁入（PR #8，新 `GsnConditionalLocScaleShift`）→ `InvCompress` ✅ 已迁入（PR #9，FrEIA 可逆层）
6. Phase 10：`WeConvene` ✅ 已迁入（PR #7，新 `WeChARMLatentCodec`）；`CCA`（→ losses/aux）、`AuxT`（→ 仅共享层）定位不变
7. Phase 11：`MambaIC` ✅ 已迁入（PR #5，共享 `ssm.py`）；`MambaVC` ❌ 不迁入（仅 arXiv 预印本）
