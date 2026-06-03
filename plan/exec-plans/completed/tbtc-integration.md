# TBTC 集成计划

> 候选：`candidate/TBTC/`，Transformer-based Transform Coding, ICLR 2022。
> 上游在 vendored CompressAI fork 中提供 4 个 image model：`zyc2022-conv-hyperprior`、`zyc2022-conv-charm`、`zyc2022-swint-hyperprior`、`zyc2022-swint-charm`。目标是迁入现仓 CompressAI 风格，保留 `forward` / `compress` / `decompress` 语义，补齐 registry、zoo、测试与 checkpoint 转换脚本。
> 当前状态（2026-05-13）：completed（runtime 集成、zoo 注册、smoke tests 与 API docs 回填已完成；实现 commit `89a1723`）。checkpoint 转换脚本与真实 ckpt 数值对齐仍作为后续项。

## 一、上游范围

### 1.1 核心文件

- `candidate/TBTC/compressai/models/qualcomm.py`
  - `ConvHyperprior(MeanScaleHyperprior)`：4-stage Conv+GDN analysis/synthesis，hyperprior 输出 `2 * main_dim` 的 scale/mean 参数。
  - `ConvChARM(ConvHyperprior)`：在 Conv-Hyperprior 上增加 10-slice channel-wise autoregressive entropy model。每 slice 大小固定为 32，因此上游实际要求 `main_dim = 320` 才能覆盖 `M` 配置；`S/L` cfg 在 zoo 里存在，但 ChARM head 写死 32 通道。
  - `SwinTHyperprior(ConvHyperprior)`：用 SwinT analysis/synthesis/hyper transforms 替换 Conv transforms。
  - `SwinTChARM(ConvChARM)`：SwinT transform + 10-slice ChARM entropy。
- `candidate/TBTC/compressai/layers/swin.py`
  - `PatchEmbed` / `PatchMerging` / `PatchSplitting` / `WindowAttention` / `SwinTransformerBlock` / `BasicLayer`。
  - 全部采用 BHWC layout，`BasicLayer` 内部交替 window / shifted-window attention。
- `candidate/TBTC/compressai/zoo/image.py`
  - 4 个 `zyc2022-*` zoo entry。
  - Conv 模型 cfg：`S=(main=192, hyper=128)`、`M=(320,192)`、`L=(448,256)`。
  - SwinT 模型 cfg：`S/M/L` 完整 dict，README 公开的 pretrained checkpoint 只覆盖 `M`、`lambda=0.01`。

### 1.2 非目标

- 不迁 Lightning 训练外壳：`lit_train.py`、`lit_data.py`、`lit_model.py`、`lit_config.py`。
- 不迁 vendored CompressAI runtime、datasets、video、sadl codec、C++ extension、third_party。
- 不新增训练依赖：`lightning`、`tensorboard` 等只用于上游训练脚本，主库集成不需要。

### 1.3 License / 权重

- 本地候选没有根目录 `LICENSE` 文件；`setup.py` 声明 `BSD 3-Clause Clear License`，多数 vendored 源文件是 CompressAI 版权头。计划按现有候选迁移政策处理为非阻塞，但 PR 文档要明确来源与 license 依据。
- README 给出 4 个 Google Drive checkpoint 链接，仓库内未包含 checkpoint。转换脚本需要支持用户手动下载后的本地路径。

## 二、落点设计

### 2.1 模型落点

新增 `compressai/models/tbtc.py`，包含：

- `TBTCConvHyperprior`
  - `@register_model("zyc2022-conv-hyperprior")`
  - 构造参数：`main_dim: int = 320`、`hyper_dim: int = 192`。
  - 保持上游 key：`g_a.0.weight` / `h_a.0.weight` 等，减少 checkpoint key 转换。
- `TBTCConvChARM`
  - `@register_model("zyc2022-conv-charm")`
  - 继承 `TBTCConvHyperprior`，新增 10-slice ChARM latent codec。
  - 默认只开放 `main_dim=320` 的 M 配置；若保留 S/L factory，需要先把 slice width 改为 `main_dim // num_slices` 并确认 checkpoint/key 兼容。计划先以 M 为主线，S/L 只作为 no-pretrained experimental kwargs。
- `TBTCSwinTHyperprior`
  - `@register_model("zyc2022-swint-hyperprior")`
  - 构造参数直接接收 `g_a/g_s/h_a/h_s` config dict，默认使用 M 配置。
- `TBTCSwinTChARM`
  - `@register_model("zyc2022-swint-charm")`
  - SwinT transforms + ChARM entropy，默认 M 配置。

`compressai/models/__init__.py` re-export `tbtc.py`。

### 2.2 SwinT transform 落点

新增 `compressai/layers/lic/tbtc.py`，仅放 TBTC 风格 BHWC transform：

- `TBTCPatchEmbed`
- `TBTCPatchMerging`
- `TBTCPatchSplitting`
- `TBTCSwinTransformerBlock`
- `TBTCBasicLayer`
- `TBTCAnalysisTransform`
- `TBTCSynthesisTransform`
- `TBTCHyperAnalysisTransform`
- `TBTCHyperSynthesisTransform`

设计原则：

- 复用 `compressai.layers.attn.swin_attention.WindowAttention`、`window_partition`、`window_reverse`、`build_window_attention_mask`、`pad_to_window_multiple`，避免复制 attention 低层实现。
- 保持上游模块命名和参数名尽量一致，尤其是 `layers.{i}.blocks.{j}.attn.qkv.*`、`norm1/norm2/mlp.*`、`downsample.reduction.*` 等 checkpoint keys。
- 不直接使用 `compressai.layers.attn.swin.PatchMerging/PatchSplit` 顶层类，因为现仓版本是 token-major `(B,L,C)` + 显式 H/W，TBTC 上游是 BHWC 且 `out_dim` 不一定等于 `2 * dim`。强行复用会导致 key shape 和 forward path 都漂移。

### 2.3 ChARM entropy 落点

优先复用现仓 `compressai.latent_codecs.ChannelSliceLatentCodec`，但要注意 TBTC 与 STF/TCM-style channel-slice 的一个差异：

- 现有 `ChannelSliceLatentCodec` 默认把完整 `latent_means` / `latent_scales` 与历史 slices 拼接。
- TBTC 上游每个 slice 只拼 `means_hat_slices[slice_index]` / `scales_hat_slices[slice_index]` 与历史 decoded slices。

计划做法：

1. 新增轻量 support transform 或 TBTC 专用 latent codec wrapper，把完整 `means_hat/scales_hat` 切成当前 slice 后再送入 `ChARMBlockHalf`。
2. 若改现有 `ChannelSliceLatentCodec`，只加可选 `latent_mean_transforms` / `latent_scale_transforms` 或 `slice_latent_params=True`，默认行为不变，避免影响 STF/TCM/MambaVC 等现有模型。
3. `lrp_transforms=None`，保持 TBTC 原始 ChARM 没有 LRP。

### 2.4 Zoo 暴露

更新：

- `compressai/zoo/image.py`
  - imports 新增 4 个 TBTC 类。
  - `candidate_model_architectures` 新增 4 个 key。
  - `candidate_model_urls` 先保持 `{}`，`pretrained=True` 抛 `Pre-trained model not yet available`。
  - 新增 factory：`zyc2022_conv_hyperprior`、`zyc2022_conv_charm`、`zyc2022_swint_hyperprior`、`zyc2022_swint_charm`。
- `compressai/zoo/__init__.py`
  - re-export 4 个 factory。
  - `image_models` 新增 4 个 key。

## 三、执行分解

| Phase | 内容 | 输出 | 估时 |
|---|---|---|---|
| 0 | 建 TBTC TODO 条目与测试基线；确认 4 个 checkpoint 下载路径约定 | `candidate/TODO.md` 待办项、plan 状态更新 | 0.25 d |
| 1 | 迁 `TBTCConvHyperprior` | `compressai/models/tbtc.py` 基础类、forward/compress/decompress、zoo entry、tests | 0.75 d |
| 2 | 接入 `TBTCConvChARM` | ChARM block + channel-slice wrapper、single-string RANS compress/decompress、tests | 1.0 d |
| 3 | 迁 TBTC SwinT layers | `compressai/layers/lic/tbtc.py` + isolated layer parity tests | 1.5 d |
| 4 | 接入 `TBTCSwinTHyperprior` / `TBTCSwinTChARM` | 2 个模型类、M cfg factory、forward smoke、state_dict roundtrip | 1.0 d |
| 5 | checkpoint 转换与真实 ckpt smoke | `examples/convert_tbtc_checkpoint.py`，4 variant forward diff / Kodak JSON 对齐入口 | 1.0 d |
| 6 | 文档、TODO、回归收尾 | docs/zoo 文档、`candidate/TODO.md` 勾选、targeted pytest | 0.5 d |

总计约 6 人日。建议先做 Conv 两个模型，再做 SwinT，因为 Conv-Hyperprior/Conv-ChARM 能先验证 entropy path 和 zoo/test glue；SwinT 的主要风险集中在 BHWC transform 与 checkpoint key parity。

## 四、关键设计决策

| 项 | 选择 | 理由 |
|---|---|---|
| 文件命名 | `compressai/models/tbtc.py`，类名加 `TBTC` 前缀 | 上游 `qualcomm.py` 命名过宽；`TBTCConvHyperprior` 等名称能和 zoo arch `zyc2022-*` 对齐 |
| Conv-ChARM slice 宽度 | 主线锁定 M 配置 `main_dim=320`、`num_slices=10`、`slice_channels=32` | 上游 ChARM head 写死 32 通道，README pretrained 也只给 M。S/L zoo cfg 存在但与 hardcoded ChARM head 不自然，先不承诺 pretrained |
| SwinT 公共层 | 新增 TBTC 专用 BHWC layers，底层复用 `WindowAttention` 工具 | 现仓已有 Swin/RSTB/STF 变体，但接口和 shape 约束不同；专用层能保 state_dict 和数值等价 |
| Entropy codec | Hyperprior 走现有 `EntropyBottleneck` + `GaussianConditional`；ChARM 尽量走 `ChannelSliceLatentCodec` 小扩展 | 避免再复制手写 slice 循环；但 TBTC 当前-slice hyper params 需要一个小 adapter |
| Zoo 权重 | 先 URL 占空，不直接写 Google Drive 链接 | 现仓 pretrained 走稳定 URL/hash；Google Drive 适合转换脚本输入，不适合内置自动下载 |
| Lightning | 不迁 | 训练外壳不是 CompressAI runtime 必需项，且会引入额外依赖 |
| Pretrained quality | factory 默认 M；S/L 只作为显式 kwargs / experimental | README 公开结果和 checkpoint 只覆盖 M，先保证可验证路径 |

## 五、验证清单

### 5.1 单元测试

- `tests/test_models.py::TestModels::test_tbtc_conv_hyperprior`
  - 小 config forward：`main_dim=32`、`hyper_dim=16`、输入 `64x64`。
  - state_dict roundtrip。
  - `compress` / `decompress` smoke。
- `tests/test_models.py::TestModels::test_tbtc_conv_charm`
  - M-compatible config 或测试专用 `slice_channels` 参数。
  - forward 输出 `x_hat`、`likelihoods["y"]`、`likelihoods["z"]`。
  - RANS single-string roundtrip。
- `tests/test_models.py::TestModels::test_tbtc_swint_hyperprior`
  - 缩小版 SwinT cfg，输入尺寸必须是 downsampling factor 与 window multiple 可处理的尺寸。
  - forward + state_dict roundtrip。
- `tests/test_models.py::TestModels::test_tbtc_swint_charm`
  - 缩小版或 M cfg forward smoke；如耗时明显，标 `slow` 或只保留 shape smoke。
- `tests/test_zoo.py::TestCandidateModels::test_tbtc_*`
  - 4 个 zoo factory smoke。
  - `pretrained=True` gate 抛 `RuntimeError("Pre-trained model not yet available")`。

### 5.2 数值等价

转换脚本 `examples/convert_tbtc_checkpoint.py`：

- `--variant {conv-hyperprior,conv-charm,swint-hyperprior,swint-charm}`
- `--checkpoint path/to/*.pth.tar`
- 剥离常见 `state_dict` / `net` / `model` 容器和 `module.` 前缀。
- 对同一输入执行 upstream candidate model 与 migrated model：
  - `x_hat` max diff = 0.0 或 FP32 误差级；
  - `likelihoods["y"]` / `likelihoods["z"]` 对齐；
  - ChARM 模型额外验证 `compress` / `decompress` roundtrip 后 `x_hat` shape 与范围。

若无法自动加载 upstream vendored package（与当前工作区 `compressai` 包名冲突），转换脚本可用 subprocess + 临时 `PYTHONPATH=candidate/TBTC` 运行上游侧结果，或先做 state_dict strict load + migrated forward smoke。

### 5.3 回归命令

```bash
.venv/bin/python -m pytest tests/test_models.py -k "tbtc or channel_slice" -q
.venv/bin/python -m pytest tests/test_zoo.py -k "tbtc" -q
.venv/bin/python -m pytest tests/test_models.py -m "not pretrained" -q
```

## 六、风险与开放点

1. **S/L ChARM 配置是否真实可用**
   - 上游 zoo 给了 `S/L`，但 `ConvChARM` 的 transform head 写死 10 个 32-channel slices。需要在 Phase 2 明确是只注册 M，还是参数化为 `slice_channels = main_dim // 10` 并验证 S/L 随机权重路径。

2. **SwinT checkpoint key parity**
   - 现仓 `WindowAttention` buffer `relative_position_index` 可能是 persistent/non-persistent 差异。转换脚本应允许丢弃/重建该 buffer，但 trainable weights 必须 strict。

3. **输入尺寸 padding**
   - 上游 SwinT layers 假设 H/W 能被 window/downsample 整除，现仓测试需要明确最小输入尺寸。若要支持任意尺寸，应该在 TBTC transform 内 pad/unpad，而不是让 attention mask 隐式失败。

4. **上游 package 名冲突**
   - `candidate/TBTC` 自带完整 `compressai` 包，不能在同一 Python process 中同时 import 当前仓和候选仓。数值对齐脚本要隔离 import。

5. **License 表述**
   - PR 描述需写清：本地候选没有独立 LICENSE 文件，但 `setup.py` 声明 BSD 3-Clause Clear，迁入代码尽量只保留 TBTC 独有模型/layers，避免携带无关 vendored 文件。
