# `compressai/layers/` 抽象分层评估与重构计划

**评估日期：** 2026-04-30
**评估对象：** `compressai/layers/`（约 12,448 行，~55 个 .py 文件）
**背景：** 随着 2022 年后新模型陆续集成（TinyLIC、ShiftLIC、TIC、Informer、GainedVAE、Entroformer、RefBasedAR、MLIC、FTIC、TCA、DCAE、CMIC、STF…），`layers/` 持续膨胀，分层和职责边界开始模糊。本文档记录现状评估与重构方向。

---

## 1. 总体结论

> **分类原则不一致；`lic/` 把模型级整块塞进了 layers，导致命名空间污染、基础层（如 ResidualBlock、window padding）多处重复实现。**

`attn/`、`ssm/`、`wave/`、`graph/`、`pointcloud/` 都按"算子/数据类型"分类，是真正的可复用层库；唯独 `lic/` 按"模型族"分类（ftic、tca、cmic、stf、dcae、mlic…），实际放的是模型私有的 Analysis/Synthesis/Hyper Transform 与 EntropyModel——它们应该属于 `compressai/models/`。

---

## 2. 现状数据

| 子目录 | 文件数 | 行数 | 性质 |
|---|---|---|---|
| 顶层（basic.py / gdn.py / layers.py） | 3 | 736 | 通用算子（混入了应用特定层） |
| `attn/`（含 `natten/`） | 11 | ~2,400 | 通用注意力算子 ✅ |
| `ssm/` | 5 | 860 | 通用 SSM 算子 ✅ |
| `wave/` | 3 | 413 | 通用小波算子 ✅ |
| `graph/` | 4 | 869 | 通用图算子 ✅ |
| `pointcloud/` | 5 | 1,711 | 点云算子 ✅ |
| **`lic/`（含 `mlic/`）** | **19** | **~4,800** | **模型族整块 ❌** |

`lic/__init__.py` 一口气导出 79 个名字（参见 `compressai/layers/lic/__init__.py:106-185`），其中大量是 `*AnalysisTransform` / `*SynthesisTransform` / `*EntropyModel`。

---

## 3. 五大问题（带证据）

### 问题 1：分类原则不一致

- 其他子目录按算子类型分类，`lic/` 按模型族分类，混合分类导致层次错乱。
- `compressai/layers/lic/ftic.py:440-480` 的 `FTICAnalysisTransform` 仅被 `compressai/models/ftic.py:11-17` 使用一次，是典型的模型私有代码，不应在 layers 中暴露。
- 类似的还有 `lic/tca.py`、`lic/dcae.py`、`lic/stf.py`、`lic/cmic.py`、`lic/cmic_stage.py`、`lic/mlic/transforms.py` 等。

### 问题 2：`ResidualBlock` 散落 13+ 变种

| 位置 | 形式 |
|---|---|
| `compressai/layers/layers.py:261-292` | `ResidualBlock`（LeakyReLU + GDN） |
| `compressai/layers/layers.py:295-330` | `ResidualBlockWithStride`、`ResidualBlockUpsample` |
| `compressai/layers/lic/mlic/transforms.py:22-71` | `GeluResidualBlock*`（仅激活换 GELU，结构相同） |
| `compressai/layers/lic/hpcm.py` | `DWConvResBlock`、`PConvResBlock` |
| `compressai/layers/lic/shift.py` | `ResidualBlockShift` |

应统一为参数化基类 `ResidualBlock(act=, norm=, conv=)`。

### 问题 3：window / padding 工具至少重复 3 份

| 位置 | 函数 |
|---|---|
| `compressai/layers/attn/swin_attention.py` | `window_partition` / `_pad_to_window_size` |
| `compressai/layers/lic/ftic.py:43-53` | `_pad_bhwc` |
| `compressai/layers/lic/tca.py:21-30` | `_pad_to_window_multiple` |
| `compressai/layers/lic/stf.py:13-19` | ✅ 复用 attn/swin_attention 的版本（说明意识到了问题，但仅部分共用） |

### 问题 4：顶层 `__init__.py` 过度暴露

- 根 `compressai/layers/__init__.py:30-37` 全部使用 `from .X import *`，盲扫所有子目录。
- `from compressai.layers import AnalysisTransform` 至少能匹配到 4 个不同的类（mlic、cmic、ftic、…），有命名冲突风险。
- `lic/__init__.py:185` 行声明 79 项 `__all__`，远超其他子目录（attn ~37，ssm ~14，wave ~13，graph ~17）。

### 问题 5：`layers.py`（496 行）边界不清

- 通用算子（`ResidualBlock`、`AttentionBlock`、卷积工厂）与应用特定层（`MaskedConv2d`、`CheckerboardMaskedConv2d`，仅自回归熵模型使用）混在同一文件。
- 标准包装器（`Lambda`、`Reshape`、`Transpose`、`Interleave`）按性质应归 `basic.py` 或 utils。

---

## 4. 目标目录结构

```
compressai/layers/                  # 通用层库（仅基础算子和工具）
├── __init__.py                     # 精选导出（~30 项）
├── basic.py
├── gdn.py
├── conv.py                         # [新] 从 layers.py 抽取：卷积工厂 + 统一 ResidualBlock 基类
├── attn/      （现状保留）
├── ssm/       （现状保留）
├── wave/      （现状保留）
├── graph/     （现状保留）
├── pointcloud/（现状保留）
└── lic_blocks/                     # [新] LIC 真正可复用的基础块
    ├── __init__.py
    ├── residual.py                 # 统一 ResidualBlock 基类（参数化 act/norm）
    ├── window.py                   # [新] 集中 padding / window_partition / window_reverse
    ├── invertible.py               # ← 现 lic/invertible.py
    ├── saaf.py                     # ← 现 lic/saaf.py
    └── ...                         # 其他确实跨模型复用的块

compressai/models/
├── ftic.py     ← 合并现 layers/lic/ftic.py
├── tca.py      ← 合并现 layers/lic/tca.py
├── dcae.py     ← 合并现 layers/lic/dcae.py
├── stf.py      ← 合并现 layers/lic/stf.py
├── cmic.py     ← 合并现 layers/lic/{cmic,cmic_stage,cmic_context}.py
├── mlic.py     ← 合并现 layers/lic/mlic/
└── _transforms/                    # 可选：若单文件过大，模型私有 Transform 放这里
```

---

## 5. 重构优先级

| 优先级 | 动作 | 工作量 | 收益 |
|---|---|---|---|
| **P0** | 把 `lic/{ftic,tca,dcae,stf,cmic,cmic_stage,cmic_context,mlic/}` 从 `layers/` 迁移到 `models/`，更新 `models/*.py` 的导入路径 | 中 | 立即解耦，`layers/` 减约 4-5k 行 |
| **P0** | 抽 `ResidualBlock(act=, norm=, conv=)` 统一基类到 `layers/conv.py`，删除 ~200 行重复实现（mlic/transforms.py、hpcm.py、shift.py 等） | 中 | 维护成本骤降 |
| **P1** | 新增 `layers/lic_blocks/window.py`，集中 padding / window_partition；`attn/swin_attention.py` 与原 ftic/tca 全部改用此版本 | 中 | 消除 3 处重复 |
| **P1** | 收紧 `layers/__init__.py` 与 `lic_blocks/__init__.py`：禁用 `import *`，模型级类不再 re-export | 低 | 命名空间清洁，API 意图明确 |
| **P2** | `MaskedConv2d` / `CheckerboardMaskedConv2d` 从 `layers.py` 移到 `models/utils.py` 或对应模型文件 | 低 | 边界清晰 |

---

## 6. 风险与兼容性

- 现有 `from compressai.layers import X` 的下游导入会断。重构期可在 `layers/__init__.py` 留过渡兼容层（`from compressai.models._transforms.ftic import FTICAnalysisTransform as _ft  # 兼容旧路径`），并打 `DeprecationWarning`，下个版本删除。
- `pytest`（不含 zoo pretrained 部分，按 AGENTS.md 第 7 条）需保持全绿；建议 P0 每迁移一个模型独立提交一次 commit 便于回滚。
- `candidate/` 目录内待集成模型可能依赖现有路径，重构前应确认 `candidate/TODO.md` 中已迁移项不再引用旧路径。

---

## 7. 后续步骤

1. 在本 PR 中**仅落地评估文档**，不动代码。
2. 由人工确认目录方案 → 创建 P0 子任务（每个模型一个 commit）。
3. P0 完成后再启动 P1 的 ResidualBlock 与 window 工具统一。

---

## 8. 执行进度

### P0 步骤 1：FTIC 迁移示范（2026-04-30 完成）

将 `compressai/layers/lic/ftic.py`（591 行）和 `compressai/layers/lic/tca.py`（343 行）合并到 `compressai/models/ftic.py`，理由是 `tca.py` 的所有类（含 `TCAEntropyModel`）实际上只服务于 FTIC 模型（无独立 TCA 模型存在）。

**变更：**
- `compressai/models/ftic.py`：605 → 1508 行（合并 transforms + tca + 主类）
- `compressai/layers/lic/ftic.py`：删除
- `compressai/layers/lic/tca.py`：删除
- `compressai/layers/lic/__init__.py`：删除 `from .ftic import (...)` / `from .tca import (...)` 与对应 13 项 `__all__`，导出从 79 项缩到 65 项。
- 未保留兼容 re-export：grep 确认 `models/ftic.py` 是唯一外部使用方，无需 DeprecationWarning 过渡层。

**验证：**
- `pytest tests/test_models.py tests/test_layers.py tests/test_init.py`: 71 passed, 4 skipped
- 直接 import + forward + state-dict 往返均通过

**经验：**
- "因文件长度限制而拆出去"的旧理由不再成立（vbr.py 已 982 行；ftic.py 现为 1508 行也可接受）。
- 单文件方案的代价是缺乏主类与 transforms 的视觉分离；迁移过程中通过 `# ----` 分隔注释保留章节边界。
- 对于"transforms + 主类 + state-dict 转换器"三段式模型，合并到单文件后阅读路径更短，不再需要在 layers/ 与 models/ 间跳转。

### P0 步骤 2：CMIC 系列迁移（2026-04-30 完成）

将 `compressai/layers/lic/{cmic,cmic_stage,cmic_context}.py`（共 175+328+134 = 637 行）合并到 `compressai/models/cmic.py`。三文件依赖完全自闭——`cmic_context.py` 的两个 importer 都是同系列内部（`cmic.py` 与 `cmic_stage.py`），不存在真正跨模型共享。

**变更：**
- `compressai/models/cmic.py`：449 → 1057 行
- `compressai/layers/lic/{cmic,cmic_stage,cmic_context}.py`：删除
- `compressai/layers/lic/__init__.py`：删除 `from .cmic import (...)` 与 5 项 `__all__`，导出从 65 项缩到 60 项。
- 未保留兼容 re-export：grep 确认 `models/cmic.py` 与 `examples/convert_cmic_checkpoint.py` 是仅有的两处使用方，且后者只引用了顶层 `CMIC` 类（仍通过 `compressai.models` 暴露）。

**验证：**
- `pytest tests/test_models.py tests/test_layers.py tests/test_init.py`: 71 passed, 4 skipped（含 `test_cmic`、`test_cmic_missing_dependency`）

**累计进度：** layers/lic/ 文件数 19 → 14；__all__ 导出 79 → 60。

### P0 步骤 3：invertible 迁移（2026-04-30 完成）

将 `compressai/layers/lic/invertible.py`（329 行）合并到 `compressai/models/invcompress.py`。`invertible.py` 仅被 InvCompress 使用，且 `is_freia_available` 函数也属于该模型的能力探测，整体迁出最干净。

**变更：**
- `compressai/models/invcompress.py`：543 → 862 行
- `compressai/layers/lic/invertible.py`：删除
- `compressai/layers/lic/__init__.py`：删除 `from .invertible import (...)` 与 9 项 `__all__`，导出从 60 项缩到 51 项。
- `compressai/zoo/image.py`、`tests/test_models.py`、`tests/test_zoo.py`：把 `is_freia_available` 的导入路径从 `compressai.layers` 改到 `compressai.models`（伴随 invertible 一起迁移）。
- 私有化收紧：`CouplingLayer`/`InvertibleConv1x1`/`SqueezeLayer`/`DenseBlock`/`EnhBlock` 加 `_` 前缀，确认没有外部用户。`build_freia_*` 三个工厂函数和 `_initialize_weights*` 也成为模型私有。
- `is_freia_available` 仍在 `__all__` 中暴露，作为模型可用性探测的公开 API。

**验证：**
- `pytest tests/test_models.py tests/test_layers.py tests/test_init.py`: 71 passed, 4 skipped（含 `test_invcompress`、`test_invcompress_missing_dependency`）

**累计进度：** layers/lic/ 文件数 14 → 13；__all__ 导出 60 → 51。

### P0 步骤 4：SFT 迁移（2026-04-30 完成）

将 `compressai/layers/lic/sft.py`（51 行，仅 1 个 `SFT` 类）合并到 `compressai/models/gained.py`，私有化为 `_SFT`。`SFT` 之前唯一的使用方是 `gained.py` 里的 `SCGainedMSHyperprior`，没有别的下游用户。

**变更：**
- `compressai/models/gained.py`：562 → 596 行
- `compressai/layers/lic/sft.py`：删除
- `compressai/layers/lic/__init__.py`：删除 `from .sft import SFT` 与 `"SFT"`，导出从 51 项缩到 50 项。
- `gained.py` 内的 `SFT(...)` 实例化（7 处）改为 `_SFT(...)`；属性名 `ga_SFT*` / `gs_SFT*` 保持不变以兼容现有 state-dict 键。
- 文档字符串引用 `:class:`~compressai.layers.SFT`` 改为 `:class:`_SFT``。

**验证：**
- `pytest tests/test_models.py tests/test_layers.py tests/test_init.py`: 71 passed, 4 skipped

**累计进度：** layers/lic/ 文件数 13 → 12；__all__ 导出 51 → 50。

### P0 步骤 5：GSDN 迁移（2026-04-30 完成）

将 `compressai/layers/lic/gsdn.py`（129 行，仅 1 个 `GSDN` 类）合并到 `compressai/models/qian2021ref.py`，私有化为 `_GSDN`。`GSDN` 仅作为 `RefBasedAR` 的可选 norm 实现使用，没有别的下游用户（其他文件出现 "GSDN" 都是字符串字面量 `norm="GSDN"` 或文档字符串）。

**变更：**
- `compressai/models/qian2021ref.py`：217 → 317 行
- `compressai/layers/lic/gsdn.py`：删除
- `compressai/layers/lic/__init__.py`：删除 `from .gsdn import GSDN` 与 `"GSDN"`，导出从 50 项缩到 49 项。
- `qian2021ref.py` 内部新增 `import torch.nn.functional as F` 与 `NonNegativeParametrizer`，`norm_cls = GSDN ...` 改为 `norm_cls = _GSDN ...`。
- 字符串字面量 `norm="GSDN"`（config 选项）保留不变；公开 API 不受影响。

**验证：**
- `pytest tests/test_models.py tests/test_layers.py tests/test_init.py`: 71 passed, 4 skipped（含 `test_qian2021_ref` GSDN/GDN 两条路径）

**累计进度：** layers/lic/ 文件数 12 → 11；__all__ 导出 50 → 49。

### P0 步骤 6：shift 迁移（2026-04-30 完成）

将 `compressai/layers/lic/shift.py`（214 行，4 个公开类 + 1 个 helper 函数）合并到 `compressai/models/shiftlic.py`。`shift.py` 仅被 `models/shiftlic.py` 与 `tests/test_layers.py` 用；`CheapChannelV1` 和 `channel_shuffle` 是完全的内部 helper（外部无引用，连测试都不测）。

**变更：**
- `compressai/models/shiftlic.py`：360 → 546 行
- `compressai/layers/lic/shift.py`：删除
- `compressai/layers/lic/__init__.py`：删除 `from .shift import (...)` 与 6 项 `__all__`，导出从 49 项缩到 43 项。
- 全部私有化：`Shift4`/`ResidualBlockShift`/`channel_shuffle`/`CheapChannelV1`/`CheapCS1`/`ResidualShiftStack` 加 `_` 前缀；属性名 `self.CheapChannel`/`self.CheapSpatial` 保持不变以兼容 state-dict。
- `tests/test_layers.py`：把对 4 个测试类的 import 从 `compressai.layers` 改到 `compressai.models.shiftlic`，用 `as` 别名映射回原名（保持测试代码体不变）。
- `latent_codecs/multistage_checkerboard.py`：更新一处过期 docstring 引用。

**验证：**
- `pytest tests/test_models.py tests/test_layers.py tests/test_init.py`: 71 passed, 4 skipped（含 `test_shiftlic_*` 与 `TestShift` 全部 4 个子测试）

**累计进度：** layers/lic/ 文件数 11 → 10；__all__ 导出 49 → 43。

### P1 步骤 1：ResidualBlock 统一（2026-04-30 完成）

把 `compressai/layers/layers.py` 的 `ResidualBlock` / `ResidualBlockWithStride` / `ResidualBlockUpsample` 加上 `act` 关键字参数（默认 `nn.LeakyReLU(inplace=True)`），并删除 `compressai/layers/lic/mlic/transforms.py` 里完全等价的 3 个 `Gelu*` 复制类。这是初始审计 §3 第 2 条问题的核心去重。

**变更：**
- `compressai/layers/layers.py`：496 → 525 行
  - 3 个 `ResidualBlock*.__init__` 新增 `*, act: Optional[nn.Module] = None`；属性 `self.leaky_relu` 重命名为 `self.act`
  - 默认行为完全不变（None → `nn.LeakyReLU(inplace=True)`）
- `compressai/layers/lic/mlic/transforms.py`：190 → 144 行
  - 删除 `GeluResidualBlock` / `GeluResidualBlockWithStride` / `GeluResidualBlockUpsample` 三个类
  - 14 处 `GeluResidualBlock*(...)` 改为 `ResidualBlock*(..., act=nn.GELU())`
  - `from compressai.layers import GDN, conv1x1, conv3x3, subpel_conv3x3` 改为 `import ResidualBlock, ResidualBlockUpsample, ResidualBlockWithStride, conv3x3, subpel_conv3x3`（不再需要 `GDN`/`conv1x1`，由基类内部使用）
- `examples/convert_mlicpp_checkpoint.py`：更新过期 docstring 引用。

**State-dict 兼容性验证：**
- `nn.LeakyReLU` 与 `nn.GELU` 都无可学习参数 → state_dict 不包含 `act.*` 键
- 其余子模块名（`conv1`/`conv2`/`gdn`/`skip`/`subpel_conv`/`upsample`/`igdn`）保持一致
- 已训练的 LeakyReLU 检查点（compressai 老模型）和 GELU 检查点（MLIC 新模型）均可继续加载，无需 state-dict 重命名
- 烟雾测试：默认 `b.act` 是 LeakyReLU；`act=nn.GELU()` 后是 GELU；二者 state_dict 键完全相同

**验证：**
- `pytest tests/test_models.py tests/test_layers.py tests/test_init.py tests/test_scripting.py`: 74 passed, 4 skipped
- 包括 `test_mlicpp` 和 `test_scripting`（torchscript 编译路径）

**累计进度：** lic/mlic/transforms.py 减重 24%；ResidualBlock 实现从 6 类合并为 3 类（减少一半）。

### P1 步骤 2：window padding 工具集中（2026-04-30 完成）

把 4 个分散的 padding helper 统一为 `compressai.layers.attn.swin_attention.pad_to_window_multiple`，支持 BCHW / BHWC 两种 layout 与 `int` / `(h, w)` 两种窗口形状。这是初始审计 §3 第 3 条问题的最终解。

**统一前的 4 个实现：**

| 位置 | 名称 | layout | window | 调用点 |
|---|---|---|---|---|
| `attn/swin_attention.py:90` | `_pad_to_window_size` | BHWC | square | swin.py / stf.py×2 / cmic.py |
| `lic/dcae.py:34` | `_pad_to_window_multiple` | BCHW | square | dcae.py / saaf.py |
| `models/ftic.py:44` | `_pad_bhwc` | BHWC | **rect** | ftic 内部 ×4 |
| `models/ftic.py:600` | `_pad_to_window_multiple` | BCHW | square | ftic TCA ×1 |

**统一后：**
```python
pad_to_window_multiple(
    x: Tensor,
    window_size: Union[int, Tuple[int, int]],
    *,
    layout: str = "BCHW",  # 或 "BHWC"
) -> Tuple[Tensor, int, int]
```

**变更：**
- `compressai/layers/attn/swin_attention.py`：新增 `pad_to_window_multiple`（41 行），删除 `_pad_to_window_size`；`__all__` / `attn/__init__.py` 同步更新
- `compressai/layers/lic/dcae.py`：删除局部 `_pad_to_window_multiple`（11 行），改 import；`saaf.py` 跟随更新
- `compressai/models/ftic.py`：删除两个局部 helper（`_pad_bhwc` 14 行 + `_pad_to_window_multiple` 11 行），同时去掉不再需要的 `import torch.nn.functional as F`
- 8 个调用点更新到统一函数；`_pad_bhwc` 的 `permute → pad → permute` 优化为直接 BHWC `F.pad`，省两次 `.contiguous()`

**验证：**
- `pytest tests/test_models.py tests/test_layers.py tests/test_init.py tests/test_scripting.py`: 74 passed, 4 skipped
- 包含 `test_ftic`（FTIC 完整 forward + state-dict round-trip）、`test_stf`、`test_cmic`、`test_dcae`、`test_saaf`

**累计进度：** padding helper 实现 4 → 1（减少 75%）；删除 ~36 行重复代码 + 一个不再需要的 import。

### P0 步骤 7：DAMO 系列合并（2026-04-30 完成）

将 DAMO Academy 团队（Yichen Qian, Ming Lin, Xiuyu Sun, Zhiyu Tan, Rong Jin 等）的两个模型及其私有 transforms 整合到 `compressai/models/damo.py`。这两个模型都是同一团队的工作（Entroformer ICLR 2022 + RefBasedAR ICLR 2021），共用 Ballé-style transforms，放一起符合学术上下文。

**变更：**
- **新建** `compressai/models/damo.py`（744 行）：
  - 内联 `_Balle2Encoder` / `_Balle2Decoder` / `_Balle2Upsample`（来自 `layers/lic/balle2.py`）
  - 内联 `_RefHyperEncoder` / `_RefHyperDecoder`（来自 `layers/lic/ref_hyper.py`）
  - `_GSDN` （从 `models/qian2021ref.py` 带过来，已 P0 时私有化）
  - `Entroformer` 完整模型（含 `from_state_dict`）
  - `RefBasedAR` 完整模型
- **删除** 5 个文件：`layers/lic/{balle2,ref_hyper,ref_search}.py` + `models/{entroformer,qian2021ref}.py`
- **`compressai/latent_codecs/ref_autoregressive.py`**：内联 `_Conv2dUnfold` / `_SearchTransfer`（从 `layers/lic/ref_search.py` 搬入），私有化。**关键决策**：放 latent_codec 而非 damo.py 是为了避免 `latent_codecs → models → latent_codecs` 循环 import。
- `compressai/models/__init__.py`：用 `from .damo import *` 替换 `from .entroformer import *` 和 `from .qian2021ref import *`
- `compressai/layers/lic/__init__.py`：删除 5 个项（Balle2Encoder/Decoder/Upsample, Conv2dUnfold, SearchTransfer, RefHyperEncoder/Decoder），导出从 43 项缩到 36 项

**公开 API 不变**：
- `from compressai.models import Entroformer, RefBasedAR` 仍工作
- `examples/convert_{entroformer,qian2021ref}_checkpoint.py` 不需要改
- zoo 注册（`@register_model("entroformer")` / `@register_model("qian2021-ref")`）保留

**验证：**
- `pytest tests/test_models.py tests/test_layers.py tests/test_init.py tests/test_scripting.py`: 74 passed, 4 skipped（含 `test_entroformer`、`test_qian2021_ref` 两个 norm 路径、torchscript）
- 直接 forward 验证：Entroformer + RefBasedAR 两模型 build + forward 通过

**累计进度：** layers/lic/ 文件数 10 → 7（减少 30%）；__all__ 导出 43 → 36（减少 16%）；models 文件数 23 → 22（合并 2 → 1）。

### P0 步骤 8：CCA NAF 块迁移（2026-04-30 完成）

将 `compressai/layers/lic/cca.py`（85 行）的 `SimpleGate` / `NAFBlock` / `NAFTransform` 搬到 `compressai/entropy_models/cca.py`，私有化为 `_SimpleGate` / `_NAFBlock` / `_NAFTransform`。CCA 系列两个真实使用方都收到了：

- `compressai/entropy_models/cca.py` 自身用 `_NAFTransform`（CCA 辅助熵模型的支持变换）
- `compressai/models/cca.py` 用 `_NAFBlock` + `_NAFTransform`（g_a/g_s + 5 切片熵模型）

**关键决策**：放 `entropy_models/cca.py`（依赖链底部）而不是 `models/cca.py`。因为 `models/cca.py` 已经依赖 `entropy_models`（`from compressai.entropy_models import EntropyBottleneck, GaussianConditional`），如果把 NAF 放 models/，反过来 `entropy_models/cca.py` 引用 `models/cca.py` 会形成循环 import。

**变更：**
- `compressai/entropy_models/cca.py`：201 → 293 行（+ 92 行 NAF 三个类）
- `compressai/models/cca.py`：导入路径改为 `from compressai.entropy_models.cca import _NAFBlock, _NAFTransform`；调用点 4 处 NAFBlock 与 4 处 NAFTransform 加 `_` 前缀
- `compressai/layers/lic/cca.py`：删除
- `compressai/layers/lic/__init__.py`：无变更（cca 本来就没在 lic/__init__.py 暴露）
- `examples/convert_cca_checkpoint.py`：更新过期 docstring 引用

**验证：**
- `pytest tests/test_models.py tests/test_layers.py tests/test_init.py tests/test_scripting.py`: 74 passed, 4 skipped（含 `test_cca` 完整 forward + state-dict round-trip）

**累计进度：** layers/lic/ 文件数 7 → 6（包含 mlic/ 子目录共 6 + 3 = 9 .py）；剩余 5 个独立 .py（blocks/dcae/saaf/stf/hpcm）。


### P0 步骤 9：SAAF 迁移（2026-05-02 完成）

将 `compressai/layers/lic/saaf.py`（285 行，6 个公开类 + 1 个 `_group_count` helper）合并到 `compressai/models/saaf.py`，全部私有化（`_AdaptiveFrequencyBlock` / `_InverseAdaptiveFrequencyBlock` / `_DenoisingAsRegularizer` / `_CrossSparseWindowAttention` / `_SpatialAttentionLayer` / `_SpatialAttentionBlock`）。`saaf.py` 6 个类的唯一外部使用方就是 `models/saaf.py`，与 step 4-6 的单 importer 模板完全一致。

**变更：**
- `compressai/models/saaf.py`：314 → 587 行（新增 inlined blocks + 删去过时 import）
  - 新增 import：`einops.rearrange`、`timm.layers.DropPath`、`compressai.layers.attn.swin_attention.pad_to_window_multiple`、`compressai.layers.lic.blocks.ResidualBottleneckBlock`、`compressai.layers.lic.dcae.{ConvolutionalGLU, Scale}`
  - 6 个调用点 `AdaptiveFrequencyBlock(...)` / `InverseAdaptiveFrequencyBlock(...)` / `DenoisingAsRegularizer(...)` / `SpatialAttentionLayer` / `SpatialAttentionBlock` 全部加 `_` 前缀
- `compressai/layers/lic/saaf.py`：删除
- `compressai/layers/lic/__init__.py`：删除 `from .saaf import (...)` 与 6 项 `__all__`，导出从 36 项缩到 30 项
- `docs/source/layers.rst`：删除 6 个 `autoclass`（`SpatialAttentionBlock` / `SpatialAttentionLayer` / `AdaptiveFrequencyBlock` / `InverseAdaptiveFrequencyBlock` / `DenoisingAsRegularizer` / `CrossSparseWindowAttention`）

**State-dict 兼容性验证：**
- 私有化只改类名，不动属性名（`olp` / `freq_attn` / `freq_weights` / `noise_predictor` / `condition_encoder` / `time_embed` / `embedding_layer` / `global_tokens` / `global_kv` / `relative_position_bias_table` / `linear` / `ln1` / `ln2` / `mlp` / `msa` / `res_scale_1` / `res_scale_2` / `layers` / `conv` 全部保持），state_dict 键完全不变
- 烟雾测试：构造小尺寸 SAAF → state_dict round-trip → forward 对比 `x_hat diff = 0.0`

**验证：**
- `pytest tests/test_models.py tests/test_layers.py tests/test_init.py`: 71 passed, 4 skipped（含 `test_saaf` 完整 forward + state-dict round-trip）

**累计进度：** layers/lic/ 文件数 6 → 5（不含 mlic/ 子目录）；__all__ 导出 36 → 30（减少 17%）。剩余单 importer：`hpcm.py`、`stf.py`、`mlic/`。

### P0 步骤 10：HPCM blocks 迁移（2026-05-02 完成）

将 `compressai/layers/lic/hpcm.py`（100 行，3 个公开类）合并到 `compressai/models/hpcm.py`，全部私有化（`_PartialConv3x3` / `_DWConvResBlock` / `_PConvResBlock`）。`models/hpcm.py` 是仅有的外部使用方；`PartialConv3x3` 实际仅在 `_PConvResBlock` 内部使用，连模型也没直接 import。

**变更：**
- `compressai/models/hpcm.py`：547 → 651 行（+104，inlined 三个类 + 章节分隔注释）
  - 删除 `from compressai.layers.lic import DWConvResBlock, PConvResBlock`
  - 5 处构造调用全部加 `_` 前缀（`g_a` 384-ch stage、`g_s` 384/640-ch stage、`y_prior` branch1/branch2 中的 `_DWConvResBlock` 列表）
  - 3 处状态推断辅助注释里的概念名同步更新
- `compressai/layers/lic/hpcm.py`：删除
- `compressai/layers/lic/__init__.py`：删除 `from .hpcm import (...)` 与 3 项 `__all__`，导出从 30 项缩到 27 项

**State-dict 兼容性：**
- 仅类名改变；属性名（`pconv` / `branch`）保持，state_dict 键完全不变
- 直接构造 + `load_state_dict` 烟雾测试通过（216 keys）

**验证：**
- `pytest tests/test_models.py tests/test_layers.py tests/test_init.py`: 71 passed, 4 skipped（含 `test_hpcm`）

**累计进度：** layers/lic/ 文件数 5 → 4（独立 .py：blocks/dcae/stf 加 mlic/ 子目录）；__all__ 导出 30 → 27（减少 10%）。剩余单 importer：`stf.py`、`mlic/`。

### P0 步骤 11：STF building blocks 迁移（2026-05-02 完成）

将 `compressai/layers/lic/stf.py`（395 行，7 个类：`_STFResidualUnit` / `STFMLP` / `STFWinBasedAttention` / `STFWinNoShiftAttention` / `STFSwinTransformerBlock` / `STFBasicLayer` / `PatchEmbed`）合并到 `compressai/models/stf.py`，全部私有化为 `_` 前缀。`models/stf.py` 是唯一外部使用方，且 `lic/__init__.py` 本来就没 re-export STF（只能通过 `compressai.layers.lic.stf` 完整路径访问，零外部依赖风险）。

**变更：**
- `compressai/models/stf.py`：454 → 808 行（+354；inlined 7 个类 + 章节分隔注释）
  - 新增 import：`timm.layers.DropPath`、`compressai.layers.attn.swin_attention.{WindowAttention, build_window_attention_mask, pad_to_window_multiple, window_partition, window_reverse}`、`Type`
  - 现有 `from compressai.layers import GDN, conv3x3, subpel_conv3x3` 添加 `conv1x1`
  - 删除 `from compressai.layers.lic.stf import (...)`
  - 7 处 `STFWinNoShiftAttention(...)` / `PatchEmbed(...)` / `STFBasicLayer(...)` 调用全部加 `_` 前缀；docstring 中 `STFWinNoShiftAttention` 引用同步更新
- `compressai/layers/lic/stf.py`：删除
- `compressai/layers/lic/__init__.py`：无变更（STF 类本来就没在 lic re-export 列表里）

**State-dict 兼容性：**
- 仅类名加 `_`；属性名（`conv` / `relu` / `fc1` / `act` / `fc2` / `drop` / `attn` / `drop_path` / `norm1` / `norm2` / `mlp` / `blocks` / `downsample` / `proj` / `norm` / `conv_a` / `conv_b`）保持，state_dict 键完全不变
- 烟雾测试：
  - `WACNN`：405 keys round-trip → forward `x_hat diff = 0.0`
  - `SymmetricalTransFormer`：315 keys round-trip → forward `x_hat diff = 0.0`

**验证：**
- `pytest tests/test_models.py tests/test_layers.py tests/test_init.py`: 71 passed, 4 skipped（含 `test_stf` / `test_wacnn` / `test_stf_upstream_state_dict` 等）

**累计进度：** layers/lic/ 独立 .py 文件 4 → 3（blocks/dcae 加 mlic/ 子目录）。剩余单 importer 仅 `mlic/`。

### P1 步骤 3：mlic/ 内部去重 R1-R4（2026-05-02 完成）

不动 `mlic/` 子目录的物理位置，先在原地把四类重复消掉。`MLICPlusPlus` 模型的 g_a/g_s + 整个 latent codec 的 forward/compress/decompress 全路径用 state-dict round-trip + bitstream round-trip 验证。

**R1 — `keys/queries/values` 工厂化（`context.py`）：**
- 新增模块级 `_pointwise_then_dwconv(dim) -> nn.Sequential`，封装 `Conv1x1 + DepthwiseConv3x3` 两层
- `LinearGlobalIntraContext` / `LinearGlobalInterContext` 各自的 `keys/queries/values` 6 处构造（每处 4 行）替换为 `self.keys = _pointwise_then_dwconv(dim)` 等
- state_dict 键完全不变（工厂返回 `nn.Sequential`，仍然是 `keys.0.weight` / `keys.1.weight`）
- 节省 ~24 行

**R2 — `compress_*` / `decompress_*` 4 个薄包装去除（`utils.py`）：**
- `_compress_symbols` / `_decompress_symbols` 重命名为公开 `compress_symbols` / `decompress_symbols`
- 删除 `compress_anchor_symbols` / `compress_nonanchor_symbols` / `decompress_anchor_symbols` / `decompress_nonanchor_symbols` 4 个 wrapper（共 ~85 行）
- 调用方直接传 `squeeze_anchor` / `unsqueeze_anchor` / `squeeze_nonanchor` / `unsqueeze_nonanchor` 函数引用
- 仅函数级 refactor，不影响 state_dict
- `utils.py`：234 → 147 行（−37%）

**R3 — `mlicpp_support.py` 合并回 `latent_codecs/mlicpp.py`：**
- `mlicpp_support.py`（161 行）整体内联：
  - `select_num_heads` → 模块级私有 helper `_select_num_heads`
  - `entropy_coder_state(codec)` → 私有方法 `MLICPlusPlusLatentCodec._entropy_coder_state(self)`
  - `compress_single(codec, ...)` → 私有方法 `MLICPlusPlusLatentCodec._compress_single(self, ...)`
  - `decompress_single(codec, ...)` → 私有方法 `MLICPlusPlusLatentCodec._decompress_single(self, ...)`
- 删除 `from .mlicpp_support import compress_single, decompress_single, select_num_heads` 与文件中 `if TYPE_CHECKING:` 反向引用 dance
- `latent_codecs/mlicpp.py`：332 → 474 行（+142；吸收 4 个函数 + 6 个新 import）
- `latent_codecs/mlicpp_support.py`：删除（−161）
- 同步应用 R2：4 处 `compress_anchor_symbols(...)` / 4 处 `compress_nonanchor_symbols(...)` / `decompress_*` 改写为 `compress_symbols(..., squeeze_anchor, unsqueeze_anchor, ...)` 等

**R4 — `mlic/__init__.py` 反映 R2 新签名：**
- 删除 4 项 `compress_anchor_symbols` / `compress_nonanchor_symbols` / `decompress_anchor_symbols` / `decompress_nonanchor_symbols`
- 新增 6 项 `compress_symbols` / `decompress_symbols` / `squeeze_anchor` / `squeeze_nonanchor` / `unsqueeze_anchor` / `unsqueeze_nonanchor`
- 暴露条目从 18 项变为 20 项（数量微增是因为 R2 把 squeeze/unsqueeze 提升为公开 API；逻辑上仍是去重）

**验证：**
- `pytest tests/test_models.py tests/test_layers.py tests/test_init.py`: 71 passed, 4 skipped
- `MLICPlusPlus` state_dict round-trip：471 keys → reload → forward `x_hat diff = 0.0`
- `MLICPlusPlus` 完整 compress → decompress 流水（R2 entropy coding 路径）：256×256 输入正常往返
- `keys.0.weight` / `keys.1.weight` 等 state-dict 键名不变，已训练 MLIC++ 检查点继续可加载

**净减重统计：**
- `mlic/utils.py`：234 → 147 行（−87）
- `latent_codecs/mlicpp_support.py`：161 → 0（−161）
- `latent_codecs/mlicpp.py`：+142
- `mlic/context.py`：381 → 372（−9，R1 因抽工厂只省了重复部分）
- `mlic/__init__.py`：45 → 49（+4）
- 净计：-72 行（且消除了 1 个文件、1 个 TYPE_CHECKING 反向 import 反模式、6 处 keys/queries/values 重复模板）

### P1 步骤 4：根 layers/__init__.py 显式化（2026-05-02 完成）

把根 `compressai/layers/__init__.py` 的 8 个 `from .X import *` 全部改为显式枚举 import + 显式根级 `__all__`，作为审计 §3 第 4 条问题的最终解（"顶层 __init__.py 过度暴露 / 盲扫"）。配套清理 `lic/__init__.py` 中已无外部使用方的 mlic re-export。

**前置验证（命名冲突扫描）：**
- 8 个子包 + 3 个顶层文件 (`basic`/`gdn`/`layers`) 的 `__all__` 总计 131 个名字，**零冲突**
- 之前评估文档里说的 "AnalysisTransform 同时匹配 mlic/cmic/ftic 4 个类" 在 P0 阶段已经消失（cmic/ftic 都已迁出 layers）
- mlic 的 10 项 re-export（`AnalysisTransform` / `ChannelContext` / `EntropyParameters` / `HyperAnalysis` / `HyperSynthesis` / `LatentResidualPrediction` / `LinearGlobalInterContext` / `LinearGlobalIntraContext` / `LocalContext` / `SynthesisTransform`）外部 0 个使用方（grep 整个 repo 包括 examples/docs/scripts/tests）—— 全是死代码，删除安全

**变更：**
- `compressai/layers/__init__.py`：8 行 wildcard → 121 项显式 import + 121 项 `__all__`
  - basic: 6 项
  - gdn: 2 项
  - layers: 16 项
  - attn: 35 项
  - graph: 17 项
  - ssm: 14 项
  - wave: 13 项
  - lic: 17 项（去掉 mlic 10 项后只剩 blocks + dcae）
- `compressai/layers/lic/__init__.py`：删除 `from .mlic import (...)` 与 10 项 `__all__`，从 27 项缩到 17 项；mlic/ 子目录文件保持不动（其内容仍由 `compressai.layers.lic.mlic` 完整路径访问，mlicpp 模型 / latent_codec 不受影响）

**验证：**
- `compressai.layers.__all__` 121 项全部 `hasattr` 通过
- `pytest tests/test_models.py tests/test_layers.py tests/test_init.py`: 71 passed, 4 skipped
- `tests/test_layers.py:32-45` 的 13 项 import 全部可解析（GDN/GDN1/AttentionBlock/MaskedConv2d/MultistageMaskedConv2d/QReLU/GatedTransformCNN/LayerNorm2d/ResidualBlock/ResidualBottleneckBlock/ResidualBlockUpsample/ResidualBlockWithStride）
- `tests/test_models.py` 与 `tests/test_zoo.py` 的 `is_pytorch_wavelets_available`、`tests/test_scripting.py` 的 `GDN/GDN1/MaskedConv2d`、`docs/source/tutorials/tutorial_custom.rst` 引用的 `GDN`/`conv`/`deconv` 全部仍在 `__all__` 中

**收益：**
- 根 `compressai.layers` 的命名空间从隐式（盲扫子包）变为显式（一目了然）
- 删除 mlic 10 项死 re-export，命名空间净减 10 名
- IDE 跳转、类型检查、grep 友好；新增/删除子包 export 时强制更新一处

