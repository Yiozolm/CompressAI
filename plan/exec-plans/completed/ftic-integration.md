# Task Plan: FTIC 上游迁入（PR #8）

## Goal

把 `FTIC`（H. Li, S. Li, W. Dai, C. Li, J. Zou, H. Xiong: *"Frequency-Aware Transformer for Learned Image Compression"*, ICLR 2024；OpenReview `HKGQDDTuvZ`；上游 `qingshi9974/ICLR2024-FTIC`）从 `script` 主干迁入上游 `Yiozolm/CompressAI` master，沿用 DCAE/SAAF/WeConvene 的"容器化模型 + convert-to-examples + deep-import-only"约定。

FTIC 在 analysis/synthesis transform 里插入**频率分解窗口注意力（FDWA）**和**频率调制 FFN**，并用基于 transformer 的**通道自回归（T-CA）熵模型**配合移位高斯（rounded-location shifted Gaussian）编码 latent。

## 关键发现

- **依赖面比 WeConvene 还小**：FTIC 只从 script `compressai/layers/attn/swin_attention.py` import 一个函数 `pad_to_window_multiple`，而该函数在 master `compressai/layers/attn/swin.py` 里**逐字相同**（DCAE/SAAF PR 落地的 BHWC-aware 版本）。→ `swin_attention.py` 整个文件冗余，**不迁入**，只改 import 路径。
- FTIC 自己的窗口/注意力/变换/TCA 全部 helper（`BranchWindowAttention`/`WindowFrequencyModulation`/`SwinFDWA`/`FATBlock`/`TCA*`）都在 `models/ftic.py` 内部。→ **单文件模型**。
- 唯一真·新文件：`compressai/entropy_models/gaussian_conditional_shifted.py`（`GsnConditionalLocScaleShift` + `Scaler`），**完全自洽**（只 import `torch` + `.entropy_models` 的 `EntropyModel`/`GaussianConditional`）。

## 上游元数据

| 项 | 值 |
|---|---|
| Paper | Li et al., ICLR 2024（OpenReview HKGQDDTuvZ） |
| 上游 repo | `qingshi9974/ICLR2024-FTIC` |
| License | **无**（合入前确认；按既定政策不阻塞） |
| 预训练权重 | 候选目录无 `.pth`；数值等价校验不可做 |

## 与 script 版的对齐改造

1. **新移位高斯熵模型自洽**：`gaussian_conditional_shifted.py` 逐字搬，`entropy_models/__init__.py` 加 export。`entropy_models` 是 `import compressai` 的 eager 核心子模块,加它不引入 timm。
2. **`pad_to_window_multiple` 路径改写**：`from ...attn.swin_attention import` → `from ...attn.swin import`（dcae/saaf 先例）。
3. **convert-to-examples**：从 model 删除 `_is_upstream_state_dict` + 全部 `_rename_*` + `convert_upstream_state_dict`,搬进 `examples/convert_ftic_checkpoint.py` 作 `convert_upstream_ftic_state_dict` 自由函数。`from_state_dict` 改纯 native shape 推断（`feature_dims`/`M`/`config`/`num_heads`/`num_slices`/`tca_*` + `_prior_mean`/`_prior_scale`/`scale_table` 处理）。
4. **纯 `@register_model("ftic")`** + **deep-import-only**（不改 `models/__init__.py`,zoo 走 `_LazyImport`）。

## convert 逻辑（upstream → compressai）

- flat `nn.Sequential`（`g_a`/`g_s`/`h_a`/`h_mean_s`/`h_scale_s`）→ 命名子阶段（`input_block`/`stage{1,2,3}.blocks.{i}`/`stage*.tail`）
- FAT_Block 内：`conv1_1`/`conv1_2` → `conv1`/`conv2`；`trans_block` → `frequency_attention`（`attns`→`branch_attentions`,`fm`→`frequency_modulation`）
- `tca.TCA.*` → `tca.tca.*`（`q1`/`k1`/`v1`→`q_proj`/`k_proj`/`v_proj`,`cpe.0`→`positional_encoding`），丢弃未用的 `start_token`,`hyper_trans` Linear 权重 reshape 成 1×1 Conv2d
- 剥 `gaussian_conditional._entropy_model.` 包裹 + 丢 upstream-only buffers

## 实现注意点（数值敏感，逐字保留）

- `SwinFDWA.forward` 的 `-split_size // 4` 运算符优先级 quirk：`(-split_size)//4 ≠ -(split_size//4)`(split_size<4 时),是训练权重 baked-in 行为,保留 + 注释。
- `from_state_dict` 返回 training 态 → 测试需 `.eval()` 才能 `torch.allclose`。

## Phases

- [X] **Phase 0**：建分支 `pr-ftic`（base master `0cbe7cf`,含 WeConvene PR #7）
- [X] **Phase 1**：`compressai/entropy_models/gaussian_conditional_shifted.py`（逐字 + `__init__` export）
- [X] **Phase 2**：`compressai/models/ftic.py`（搬 + `swin` import + 删 convert + 纯 native `from_state_dict`）
- [X] **Phase 3**：`examples/convert_ftic_checkpoint.py`（内联 `convert_upstream_ftic_state_dict`）
- [X] **Phase 4**：zoo 接线（`_LazyImport` + `ftic()` factory）
- [X] **Phase 5**：`tests/test_models.py::TestFTIC`（forward / native round-trip / compress / 合成 upstream conversion）
- [X] **Phase 6**：全量验证
- [X] **Phase 7**：commit + PR
- [X] **Phase 8（CI 修复）**：见下

## Errors Encountered（实施过程中的坑）

1. **CI static_analysis 失败（pre-existing）**：ruff 0.8.6 嫌两个**非本 PR 改动**的旧文件——`multi_context_slice.py`（末尾多空行）+ `graph/__init__.py`（import 没排序）。PR #8 是第一个真正跑这套新 CI workflow 的（PR #7 合并时 "no checks reported"），把 master 上潜伏的旧问题照了出来。顺手在本 PR 修了（commit `6bba044`）。
2. **CI tests 失败(mamba-ssm 编译)**：`uv sync --all-extras` 试图从源码编译 `mamba-ssm==2.2.2`，但 CPU runner 无 nvcc → `NameError: bare_metal_version`,装依赖阶段就崩。用户定"CI 不管 ssm"→ `pytest.yml` 改 `--all-extras --no-extra ssm`（commit `f53e631`）。MambaIC/CMIC 走纯 PyTorch fallback,测试照跑。

## Status

**Done** —— 已上游迁入 `Yiozolm/CompressAI` master,**PR #8（2026-06-03,merge commit `36b48b7`）**,16 CI checks 全绿。

- 模型 `compressai/models/ftic.py::FrequencyAwareTransFormer`；移位高斯 `compressai/entropy_models/gaussian_conditional_shifted.py::GsnConditionalLocScaleShift`；1 个 zoo entry `ftic`
- convert 脚本 `examples/convert_ftic_checkpoint.py::convert_upstream_ftic_state_dict`
- 复用 master `pad_to_window_multiple`（`swin.py`,无新 layer 文件）；`[attn]` extra（timm,无新依赖）
- 测试 `tests/test_models.py::TestFTIC`（3 个）
- commits：`2763fa0`（entropy_models）、`a85407c`（model）、`f52c181`（zoo+examples）、`51d3e22`（test）、`6bba044`（pre-existing ruff 修复）、`f53e631`（CI skip ssm）
