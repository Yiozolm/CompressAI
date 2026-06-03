# Task Plan: WeConvene 上游迁入（PR #7）

## Goal

把 `WeConvene`（H. Fu, J. Liang, Z. Fang, J. Han, F. Liang, G. Zhang: *"WeConvene: Learned Image Compression with Wavelet-Domain Convolution and Entropy Model"*, ECCV 2024；arXiv 2407.09983）从 `script` 主干迁入上游 `Yiozolm/CompressAI` master，沿用 DCAE/SAAF 的"容器化模型 + convert-to-examples + deep-import-only"约定。

WeConvene 把**小波域卷积**引入 analysis/synthesis transform，并用 wavelet-domain channel-AR 熵模型（双熵分支）编码 latent。

## 关键发现

- **依赖面**：复用 pr-dcae-saaf-auxt 已落地的 `[wavelet]` extra（`pytorch_wavelets`）和 `compressai/layers/wave/` 子包——**无新依赖**。WeConvene 的小波 transform 落在 `compressai/layers/wave/weconv.py`。
- **新 latent codec**：`WeChARMLatentCodec`（wavelet-domain channel-AR），是本 PR 唯一真·新 codec 类,落 `compressai/latent_codecs/weconvene.py`，`@register_module("WeChARMLatentCodec")`。
- **Swin block 变体**：upstream 自带一份与 master `SwinBlock` 略异的 residual swin block，迁入版用 `_ResidualSwinBlock` 在 model 文件内匹配 upstream 行为（保 state_dict fidelity）。

## 上游元数据

| 项 | 值 |
|---|---|
| Paper | Fu et al., ECCV 2024（arXiv 2407.09983） |
| License | 按既定政策不阻塞 |
| 预训练权重 | 候选目录无可用 `.pth`；数值等价校验不可做 |
| 依赖 | `[wavelet]`（`pytorch_wavelets`,软可选,pr-dcae-saaf-auxt 已落地) |

## 与 script 版的对齐改造

1. **wavelet transform 入 `layers/wave/`**：`WeConveneAnalysisTransform` / `WeConveneSynthesisTransform` / `WeConveneHyperAnalysisTransform` / `WeConveneHyperSynthesisTransform` + `WaveletResidualBlockWithStride` / `WaveletResidualBlockUpsample` 落 `compressai/layers/wave/weconv.py`,`wave/__init__.py` 加 export。
2. **新 codec 入 `latent_codecs/`**：`WeChARMLatentCodec` 落 `compressai/latent_codecs/weconvene.py`,`latent_codecs/__init__.py` 加 export。
3. **convert-to-examples**：upstream→compressai key remap 移进 `examples/convert_weconvene_checkpoint.py::convert_upstream_weconvene_state_dict`；model `from_state_dict` 纯 native shape 推断。
4. **纯 `@register_model("weconvene")`** + **deep-import-only**（不改 `models/__init__.py`,zoo 走 `_LazyImport`；`pytorch_wavelets` 缺失时构造期报错）。

## Phases

- [X] **Phase 0**：建分支 `pr-weconvene`（base master,含 TinyLIC/ShiftLIC PR #6）
- [X] **Phase 1**：`compressai/layers/wave/weconv.py`（小波 transform + residual blocks）+ `wave/__init__` export
- [X] **Phase 2**：`compressai/latent_codecs/weconvene.py::WeChARMLatentCodec` + `latent_codecs/__init__` export
- [X] **Phase 3**：`compressai/models/weconvene.py::WeConvene`（纯 `@register_model` + 纯 native `from_state_dict` + `_ResidualSwinBlock`）
- [X] **Phase 4**：`examples/convert_weconvene_checkpoint.py`（`convert_upstream_weconvene_state_dict`）
- [X] **Phase 5**：zoo 接线（`_LazyImport` + `weconvene()` factory）
- [X] **Phase 6**：`tests/test_models.py` + `tests/test_layers.py`（WeConvene model + layer 覆盖）
- [X] **Phase 7**：全量验证 + commit + PR

## Status

**Done** —— 已上游迁入 `Yiozolm/CompressAI` master,**PR #7（2026-06-03,merge commit `0cbe7cf`）**。

> 注：PR #7 合并时新 CI workflow 尚未生效（"no checks reported"）；CI 在 PR #8（FTIC）首次真正运行,并照出两个 master 上潜伏的 pre-existing 问题（旧文件 ruff 格式 + mamba-ssm 编译）——那两个在 PR #8 里一并修了,与 WeConvene 本身无关。

- 模型 `compressai/models/weconvene.py::WeConvene`；新 codec `compressai/latent_codecs/weconvene.py::WeChARMLatentCodec`；小波层 `compressai/layers/wave/weconv.py`；1 个 zoo entry `weconvene`
- convert 脚本 `examples/convert_weconvene_checkpoint.py::convert_upstream_weconvene_state_dict`
- 复用 `[wavelet]` extra(`pytorch_wavelets`,无新依赖)
- 测试 `tests/test_models.py`（WeConvene）+ `tests/test_layers.py`（小波层）
- commits：`97498ac`（layers）、`867f109`（latent_codecs WeChARMLatentCodec）、`2cb0609`（model）、`4482b9f`（zoo+examples）、`c2157b2`（test）
