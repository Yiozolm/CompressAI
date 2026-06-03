# Task Plan: InvCompress 上游迁入（PR #9）

## Goal

把 `InvCompress`（Y. Xie, K.L. Cheng, Q. Chen: *"Enhanced Invertible Encoding for Learned Image Compression"*, ACMMM 2021；arXiv 2108.03690；上游 `xyq7/InvCompress`）从 `script` 主干迁入上游 `Yiozolm/CompressAI` master，沿用 DCAE/SAAF/WeConvene/FTIC 的"容器化模型 + convert-to-examples + deep-import-only"约定。

InvCompress 用**可逆神经网络（INN）**＋注意力通道压缩＋特征增强模块替换 `Cheng2020Anchor` 的自编码变换,熵模型保留 joint autoregressive + hyperprior。

## 关键发现

- **依赖面**：唯一真·新依赖是 `FrEIA`（Framework for Easily Invertible Architectures）——master 的 `pyproject.toml`/`uv.lock` 此前都没有。INN 变换由 FrEIA 的 `GLOWCouplingBlock` / `Fixed1x1Conv` / `IRevNetDownsampling` 组装。
- **FrEIA 可控**：PyPI 仅 `FrEIA 0.2`（2022，sdist-only，但 `setup.py` 只 import setuptools → 隔离构建无需 torch；运行期 deps `numpy/scipy/torch` 已在 lock）。0.2 的 API 恰好支持 wrapper 所需的 `clamp_activation="SIGMOID"` / `Fixed1x1Conv(M=...)` / `legacy_backend=True`。
- **single-file 模型**：FrEIA wrapper（`_CouplingLayer`/`_InvertibleConv1x1`/`_SqueezeLayer` + `build_freia_*`）、blocks（`_DenseBlock`/`_EnhBlock`）、`_InvertibleCompressionTransform`、`InvCompress` 全在 `compressai/models/invcompress.py` 内部，不依赖 layers 子包。
- **`import compressai` 不被污染**：FrEIA 全走 `_load_freia_modules()` 函数内 lazy import；模块顶层只 `from importlib import import_module, util`。

## 上游元数据

| 项 | 值 |
|---|---|
| Paper | Xie, Cheng, Chen, ACMMM 2021（arXiv 2108.03690） |
| 上游 repo | `xyq7/InvCompress` |
| License | **Apache 2.0**（model 文件保留上游 attribution 段 + InterDigital BSD-3-Clear header） |
| 预训练权重 | 候选目录无 `.pth`；数值等价校验不可做（同 ShiftLIC/WeConvene/FTIC） |

## 与 script 版的对齐改造

1. **convert-to-examples**：script `invcompress.py` 把 `convert_upstream_state_dict` / `_is_upstream_state_dict` / `_block_diagonal_conv` 内嵌在 model 里,且 `from_state_dict` 顶部无条件调 convert。迁入版**全部搬进** `examples/convert_invcompress_checkpoint.py` 作 `convert_upstream_invcompress_state_dict` 自由函数；model `from_state_dict` 改纯 native（`N = state_dict["h_a.0.weight"].size(0)` → `cls(N=N)` → `load_state_dict`）。

2. **纯 `@register_model` + 修 py3.8 崩溃**：script 用 `@_maybe_register_model("invcompress")`（FrEIA 缺失退化 identity）。配套的 `_ModelType = TypeVar(..., bound=type[nn.Module])` 是**运行期实参**（`from __future__ import annotations` 救不了），`type[...]` 下标在 **py3.8 import 期直接崩**。迁入版改纯 `@register_model("invcompress")`（FrEIA 缺失时由 `__init__` 的 `_require_freia()` 在构造期报错），并删除 `_identity_decorator`/`_maybe_register_model`/`_ModelType`。

3. **deep-import-only**：不改 `compressai/models/__init__.py`；zoo 走 `_LazyImport`（仿 dcae/ftic）。

4. **新 extra + lock 重生成**：`pyproject.toml` 加 `invcompress = ["FrEIA"]`，`uv lock` 重生成（新增 freia 0.2 节点）；CI `--all-extras` 真装真跑 `TestInvCompress`（不像 ssm 那样 `--no-extra`）。

## convert 逻辑（upstream → compressai）

- 可训练的 upstream `inv.operations.{i}.weight`（`[C,C]` 矩阵）→ FrEIA `Fixed1x1Conv` 的 `M`/`M_inv`（float64 求逆）/`logDetM` 三 buffer。
- 四并联耦合子瓶颈 `G1`/`G2`/`H1`/`H2` 融合成两个 `GLOWCouplingBlock` subnet：`conv1` 沿输出维行拼 `[scale; shift]`，`conv2`/`conv3` 放块对角（零跨块），bias 拼接——块对角 + 逐元素 LeakyReLU = 两半互不串扰，数值 bit-for-bit 等价。

## Phases

- [X] **Phase 0**：建分支 `pr-invcompress`（base master `36b48b7`，含 FTIC PR #8）
- [X] **Phase 1**：`compressai/models/invcompress.py`（搬 + 删 convert/registration、纯 native `from_state_dict`、纯 `@register_model`、`__all__=["InvCompress"]`、清 unused import）
- [X] **Phase 2**：`examples/convert_invcompress_checkpoint.py`（内联 `convert_upstream_invcompress_state_dict` + `_is_upstream_invcompress_state_dict` + `_block_diagonal_conv`）
- [X] **Phase 3**：`pyproject.toml` 加 `invcompress = ["FrEIA"]` + `uv lock` 重生成
- [X] **Phase 4**：zoo 接线（`_LazyImport` + `invcompress()` factory）
- [X] **Phase 5**：`tests/test_models.py::TestInvCompress`（forward / native round-trip / compress / 合成 upstream conversion）
- [X] **Phase 6**：全量验证（import 卫生、py3.8 type[] 扫、ruff、lock check、全量 pytest）
- [X] **Phase 7**：commit + PR

## Errors Encountered（实施过程中的坑）

1. **numpy 2.x 撞老 torch（CI 27 fail）**：为"放宽 scipy/numpy 版本"我加了 `fork-strategy = "requires-python"`，结果把 numpy 2.4.6 拉到 py3.8–3.11，但那里 torch 锁的是 2.2.2（torch 2.3 才支持 numpy 2.0 ABI）→ CI `RuntimeError: Numpy is not available`（27 个测试 fail）。本机 venv 是 py3.13 + torch 2.9，吃得下 numpy 2.x，**掩盖了**这个矩阵不兼容——只有 CI 照出来。
   - **修法**：`numpy >= 1.24.4, < 2; python_version < '3.12'` + `numpy >= 1.24.4; python_version >= '3.12'`。scipy 仍升上去（1.13.1 → 1.15.3 (3.10/3.11) → 1.17.1 (3.12)），因为 scipy 1.17.1 接受 `numpy>=1.26.4,<2.7`，cap 下 py3.11 退到 scipy 1.15.3。
   - **教训**:torch/numpy ABI 这类只有 CI 矩阵能照出来，本机单 Python 版本会掩盖。

2. **本机 py3.13 scipy 源码构建**：`uv sync` 一度试图从源码编译 scipy 1.13.1（py3.13 无 cp313 wheel）——同样由 `fork-strategy` 解决（py3.13 取 scipy 1.17.1，有 cp313 wheel）。

## Status

**Done** —— 已上游迁入 `Yiozolm/CompressAI` master，**PR #9（2026-06-03，merge commit `43d47eb`）**，16 CI checks 全绿（FrEIA 经 `--all-extras` 真装，TestInvCompress 在 py3.8–3.12 真跑）。

- 模型 `compressai/models/invcompress.py::InvCompress`（FrEIA-backed INN + `_AttentionModule` + `_EnhancementModule`）；1 个 zoo entry `invcompress`
- convert 脚本 `examples/convert_invcompress_checkpoint.py::convert_upstream_invcompress_state_dict`
- 新 extra `invcompress = ["FrEIA"]` + `uv.lock` 刷新 + `fork-strategy = "requires-python"` + numpy<2 (py<3.12) cap
- 测试 `tests/test_models.py::TestInvCompress`（3 个，CI 真跑）
- commits：`c5d560e`（model）、`f744f9b`（zoo+examples）、`d77d318`（build：extra+lock）、`c5c35ba`（test）、`7861bb6`（build：numpy<2 cap 修 CI）
