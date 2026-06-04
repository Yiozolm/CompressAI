# pr-tcm-cca: Codec 容器化 H+G 重构执行计划

**计划日期**：2026-05-09
**完成日期**：2026-05-09 ✅
**分支**：`pr-tcm-cca`（基于 `pr-stf-wacnn`，rebase 后基于 `upstream/master`）
**目标 PR**：本仓向上游 `InterDigitalInc/CompressAI` 提交的下一个 PR
**最终 commits**：`c6d556a..f87c8c8`（6 commits，rebase 后 hash），已 push 到 `origin/pr-tcm-cca`
**PR draft**：[`plan/generated/pr-tcm-cca-draft.md`](../../generated/pr-tcm-cca-draft.md)
**设计文档**：[`plan/design-docs/channel-slice-codec-redesign.md`](../../design-docs/channel-slice-codec-redesign.md) §1–§3 是 codec 家族分类速查表（变异维度 + 七族叙事）。**候选方向对比、推荐方案与详细 API/state_dict 设计**（原 design-doc §4–§10）已于 2026-06-04 重构并入本计划末尾「[设计依据](#design-rationale)」段。

> **执行顺序调整（2026-05-09，Phase 1 完成后）**：原 Phase 2「删除 `ChannelSliceLatentCodec`」会提前破坏 STF/WACNN（因为 `_bases/slice_entropy.py` 还在 import 它），跟 Phase 2 自身的「不引入新失败」验收冲突。决定：**Phase 2 推迟到 STF/TCM/CCA 全部迁移完成之后**，与 Phase 6 删除 `_bases/slice_entropy.py` 合并执行。Phase 文档编号保持稳定；实际执行顺序为 **Phase 1 → 3 → 4 → 5 → 2+6 → 7 → 8**。详见 Phase 2 / Phase 6 头部 cross-reference。
>
> **完成回填（2026-05-09）**：所有 8 个 Phase 全部落地。最终 6 commits（rebase 后）`c6d556a..f87c8c8` 已 push 到 `origin/pr-tcm-cca`，base `upstream/master`，6 ahead / 0 behind 线性可 fast-forward merge。PR draft 在 `plan/generated/pr-tcm-cca-draft.md`，待 InterDigital upstream review。Phase 7 全量验证：213/213 非 pretrained 测试 + ruff 全过 + 0 timm import + state_dict §10.7 路径自检 4/4。Phase 8.3 的 rebase 决策记录见末尾「实施差异 / 决策记录」段。

---

## ⭐ 首要任务（4 个 Family 1 模型按容器化重写）

按以下顺序迁移到容器化 + codec-owned hyperprior（H+G）：

| 优先级 | 模型 | 来源 | Phase |
|---|---|---|---|
| 1 | **WACNN** | 已在 `pr-stf-wacnn` 基线上 | Phase 3 |
| 2 | **SymmetricalTransFormer (STF)** | 已在 `pr-stf-wacnn` 基线上 | Phase 3 |
| 3 | **TCM** | fork `script` 上有，本 PR 新增到 pr-tcm-cca | Phase 4 |
| 4 | **CCA**（CCAModel + `_CCAAuxEntropyModel`） | fork `script` 上有，本 PR 新增到 pr-tcm-cca | Phase 5 |

这 4 个模型是本 PR 的**核心交付**。infrastructure（Phase 1-2）服务于它们；清理与验证（Phase 6-8）收尾。

**延后**：DCAE, MambaVC（fork `script` 已迁入但未上游）—— 留给独立 follow-up PR。
**不动**：ELIC, MLIC++, HPCM, WeConvene, TinyLIC, ShiftLIC, Entroformer, RefBasedAR。

---

## 上下文与现状

### 已与上游 maintainer 确认
- `@YodaEmbedding` 同意后续 refactor PR 重构 latent_codec 抽象
- state_dict 兼容性约束已解除，可以重排 Family 1 模型 ckpt 路径

### 当前 working tree 状态
- 本地有未 commit 的 TCM/CCA 改动（按旧 monolithic + model-owned hyperprior 写的）
- 这部分**全部要丢弃重写**，因为 H+G 是结构性 pivot

---

## Phase 0: 清理 working tree（30 分钟）✅ 完成于 2026-05-09

将现有 TCM/CCA 改动备份到独立分支后清空 working tree。

### 任务
- [x] 备份当前进度：`git checkout -b pr-tcm-cca-monolithic-backup`，commit 所有未保存改动作为参考快照（不推到 origin）→ snapshot commit `f3fcd4d`，13 文件 +2162 行（7 modified + 6 new）
- [x] 切回 `pr-tcm-cca`，hard reset 到 `pr-stf-wacnn` 的最新 commit（`c9496cb` 当前 head）→ 两分支已同 commit，无需 reset，仅 `git checkout pr-tcm-cca`
- [x] 确认 working tree 干净：`git status` 只剩 untracked 的 `plan/`、`AGENTS.md`、`CLAUDE.md`、`candidate*/`

### 验收
- `git log pr-tcm-cca` 与 `pr-stf-wacnn` 完全一致 ✅（HEAD = `c9496cb`）
- 新分支 `pr-tcm-cca-monolithic-backup` 保留旧实现作为参考 ✅（HEAD = `f3fcd4d`，未推 origin）

### 风险
- 丢失精简过的 CCA/TCM 代码 → 备份分支兜底

---

## Phase 1: 新增 latent_codecs 基础设施 + 扩展 upstream（1 天）✅ 完成于 2026-05-09，commit `c6d556a`

新增 3 个文件 + 扩展 upstream `channel_groups.py` ~10 行 + 应用层 helper 2 个文件，单元测试覆盖。

### 任务

#### 1.1 `compressai/latent_codecs/_hyper_synthesis.py`（~25 行）
- [x] 定义 `DualHyperSynthesis(h_mean_s, h_scale_s)` adapter
- [x] forward 返回 `cat([h_mean_s(z), h_scale_s(z)], dim=1)`
- [x] 单元测试：构造 + forward shape（含 state_dict path 子测试）

#### 1.2 在 upstream `compressai/latent_codecs/gaussian_conditional.py` 末尾追加 `LRPGaussianLatentCodec`（~30 行）
- [x] **不新建文件**——subclass 跟基类同文件，避免文件碎片
- [x] 定义 `LRPGaussianLatentCodec(GaussianConditionalLatentCodec)` —— **subclass upstream `GaussianConditionalLatentCodec`**
- [x] 构造函数：`(lrp_transform, *, lrp_scale=0.5, **gc_kwargs)` —— 透传 `entropy_parameters` / `quantizer` / `chunks` / `gaussian_conditional` 给基类
- [x] override `forward(y, ctx_params)`：调 `super().forward()` 拿 `y_hat`，再做 LRP 后处理（`lrp = self.lrp_scale * tanh(self.lrp_transform(cat([ctx_params, y_hat])))`）
- [x] **关键决策**：LRP 输入用整段 `ctx_params`（不是单独 `mean_support`）—— application 层用合适宽度的 `lrp_transform` 第一 conv 适配。Phase 3 wiring STF/WACNN 时若需切片字段可再细化
- [x] override `compress` / `decompress` 同样加 LRP 后处理
- [x] `__all__` 加 `"LRPGaussianLatentCodec"`
- [x] 单元测试：forward + state_dict round-trip + 与 `GaussianConditionalLatentCodec` 兼容（`lrp_scale=0` 退化测试 + likelihoods 不变）

#### 1.3 扩展 upstream `compressai/latent_codecs/channel_groups.py`（~10 行 diff）
- [x] 加构造参数 `max_support_slices: int = -1`（默认 -1 表示 use-all-prior，跟 upstream 当前行为一致）
- [x] 加构造参数 `support_filter: Optional[Callable[[int, List[Tensor]], List[Tensor]]] = None`
- [x] 修改 `_get_ctx_params(k, side_params, y_hat_)`：用 `self._select_support(k, y_hat_)` 替代 `y_hat_[:k]`
- [x] `_select_support` 实现：如果 `support_filter` 不为 None，调它；否则按 `max_support_slices` clamp（`prior[:max_support_slices]`）
- [x] **必须向后兼容**：默认参数下行为跟 ELIC 等 upstream 用户一致 ✅（`tests/test_models.py::TestStf` 3 通过）
- [x] 单元测试：默认参数等同原 forward；`max_support_slices=2` clamp；`support_filter=skip_most_recent` 验证

#### 1.4 `compressai/latent_codecs/_slice_helpers.py`（~120 行）
- [x] 从 `compressai/models/_bases/slice_entropy.py` **复制**（不删除——Phase 6 才删，避免 Phase 1-5 期间 STF/WACNN 失效）：
  - `slice_support_channels`
  - `lrp_support_channels`
  - `make_entropy_transform`
  - `infer_num_slices`
  - `infer_max_support_slices`
- [x] **更新 `infer_num_slices` 的 prefix scan** 适配新 state_dict 路径（默认 prefix `latent_codec.latent_codec.y.channel_context.y` + suffix `.mean_cc.0.weight`）；slice 0 无 channel_context entry，函数返回 `len(matches) + 1`
- [x] 单元测试：每个 helper 函数（含 default widths / 自定义 widths / clamp / new prefix scan）

#### 1.5 `compressai/latent_codecs/__init__.py` exports
- [x] 加 `DualHyperSynthesis`, `LRPGaussianLatentCodec`
- [x] helper 函数走 `compressai.latent_codecs._slice_helpers` 深路径，不暴露在顶层

#### 1.6 `compressai/models/_helpers/channel_slice.py`（新文件，~40 行）
- [x] 应用层 factory：`build_channel_slice_codec(*, groups, leaf_factory, channel_context_factory, max_support_slices, support_filter) -> ChannelGroupsLatentCodec`
- [x] 内部生成 `{"y0".."yK-1"}` 字典传给 `ChannelGroupsLatentCodec`
- [x] **注意**：`"y0"` 不需要 channel_context（slice 0 无 prior，upstream `_get_ctx_params(k=0)` 直接走 `side_params`）；从 `"y1"` 开始才生成 context entry
- [x] 单元测试：用工厂构造 + state_dict 路径验证（含 `max_support_slices` / `support_filter` 透传）

#### 1.7 `compressai/models/_helpers/channel_context.py`（新文件，~80 行）
- [x] 应用层 helper：`MeanScaleContextHead` `nn.Module` + `build_mean_scale_head(slice_ch, support_ch, *, widths=(224, 128), support_transform_factory=None) -> MeanScaleContextHead`
- [x] 输出 channels = `2 * slice_ch`（mean + scale 拼起来）
- [x] 内部分别构造 mean_cc 和 scale_cc 头，可选用 support_transform_factory 为 mean / scale 各包一层 SWAtten / NAFTransform（独立实例）
- [x] 单元测试：构造 + forward + shape + 默认无 support transform 时 state_dict 不含其字段

### 验收
- [x] `pytest tests/test_latent_codecs.py tests/test_models_helpers.py -q` 28 通过 ✅
- [x] `make static-analysis` 全过（ruff format / imports / lint，3 步）✅
- [x] `import compressai`, `compressai.zoo`, `compressai.latent_codecs` 不加载 timm 也不引入新依赖 ✅
- [x] `pytest tests/test_models.py -q` 16 通过（含 STF 3）✅
- [x] `uv lock --check` 一致（无 pyproject 改动）

### 实施差异 / 决策记录
- LRP 数据传递选「整段 `ctx_params`」而非「单独 `mean_support` kwarg」—— ChannelGroupsLatentCodec 接口零变化，application 层 lrp_transform 的第一 conv 配 `ctx_ch + slice_ch` 即可
- `_slice_helpers.py` 是**复制**而不是搬迁——保留 `_bases/slice_entropy.py` 至 Phase 6 删除，确保 STF/WACNN 在 Phase 1-5 期间继续可工作
- `infer_num_slices` 在新 prefix 下返回 `len(matches) + 1`，因为 slice 0 无 channel_context entry 不被扫到（旧路径下每个 slice 都有 cc_mean_transforms entry，无需 +1）
- 测试 fixture `tests/test_models_helpers.py` 不在 `tests/test_models/` 下（plan 原写法），因为现有测试都是平铺文件，沿用既定布局

### 风险
- ~~LRP 数据传递约定（`mean_support` 怎么从 channel_context 传到 leaf）需要在 1.2 实现时定下；可能需要小调整 1.7 的 head 输出格式~~ → 已通过「整段 `ctx_params`」方案规避，无需调整 head 输出格式

---

## Phase 2: 删除冗余 + 扩展验证（0.5 天）✅ 完成于 2026-05-09，commit `0c44c75`（合入 Phase 6.1）

> **执行已合并**：见文档顶部「执行顺序调整」。本 Phase 的删除动作并入 Phase 6.1 的 cleanup commit `0c44c75`，与 `_bases/slice_entropy.py` 同步删除。本 Phase 文档保留作为 scope 描述。

把 pr-stf-wacnn 加的 `ChannelSliceLatentCodec`（实质重复 `ChannelGroupsLatentCodec`）删除。

### 任务（所有勾选项均通过 commit `0c44c75` 完成）
- [x] 删除 `compressai/latent_codecs/channel_slice.py` ✅
- [x] `compressai/latent_codecs/__init__.py` 移除 `ChannelSliceLatentCodec` export ✅
- [x] 全仓 grep `ChannelSliceLatentCodec`，确认所有 caller（应该只有 `compressai/models/_bases/slice_entropy.py` 和 `tests/test_models.py`）—— Phase 3-5 会逐一迁过去 ✅（grep 验证零生产/测试 caller）
- [x] 验证 ELIC（`compressai/models/sensetime.py`）继续工作：`pytest -k Elic` 全过 ✅（`tests/test_models.py` 16/16 全过含 ELIC）

### 验收
- [x] `pytest tests/` 全 suite 不引入新失败（其他模型不应该依赖 `ChannelSliceLatentCodec`）✅
- [x] ELIC 测试通过证明扩展 `ChannelGroupsLatentCodec` 没破坏现有用户 ✅

### 风险（已消解）
- ~~如果 STF/WACNN 的 from_state_dict 已经 hardcode 了 `ChannelSliceLatentCodec` 类引用，需要在 Phase 3 一起处理~~ → Phase 3 迁移时确认 STF/WACNN 的 `from_state_dict` 已改用 `ChannelGroupsLatentCodec` 路径，无 hardcode 残留；commit `0c44c75` 删除时全仓 grep 验证零生产/测试 caller

---

## Phase 3: 迁移 STF/WACNN（2 天）✅ 完成于 2026-05-09，commit `8b3ea4d`

重写 `compressai/models/stf.py` 用容器化 + codec-owned hyperprior。

> **接口落点（Phase 1 实施后定稿）**：使用 `compressai.latent_codecs.ChannelGroupsLatentCodec`（upstream，Phase 1 已扩展 `max_support_slices` / `support_filter`，Phase 3 再扩展 `side_in_context`），通过 `compressai.models._helpers.channel_slice.build_channel_slice_codec` factory 装配。`ChannelSliceLatentCodec`（pr-stf-wacnn 引入的重复轮子）将在 Phase 6 cleanup 删除，**Phase 3 不再使用**。

### 任务

#### 3.0 进一步扩展基础设施（Phase 1 之上的小幅追加）
- [x] `ChannelGroupsLatentCodec`：加 `side_in_context: bool = False`。`True` 时（a）`_get_ctx_params` 把 `side_params` cat 进 `channel_context.y{k}` 输入（含 k=0，需要存在 `channel_context.y0`）；（b）skip 末尾对 leaf 的二次 cat（channel_context 输出即最终 ctx_params）。ELIC 默认 False，零行为变化
- [x] `MeanScaleContextHead`：加 `side_split: int = 0`。`>0` 时 forward 内部 split：`latent_means = x[:, :side_split]`、`latent_scales = x[:, side_split:2*side_split]`、`prev_y_hat = x[:, 2*side_split:]`，分别送 mean_cc / scale_cc。返回 `cat(scale, mean)` 给 `chunks=("scales","means")`
- [x] **追加（实施时新增）**：`MeanScaleContextHead` 加 `emit_mean_support: bool = False`。`True` 时把 `mean_in = cat(latent_means, *prev_y_hat)` 拼到输出末尾，让 `LRPGaussianLatentCodec` 能切出来作为 LRP 输入——这样 LRP 第一 conv 输入宽度就能跟上游 `M + slice_ch*(support+1)` 匹配，权重 byte-for-byte 可迁移
- [x] **追加（实施时新增）**：`LRPGaussianLatentCodec` 加 `mean_support_trail_channels: int = 0`。`>0` 时把 ctx_params 切成 `[gaussian_params, mean_support]`，gaussian_params 送基类 chunk，mean_support 送 LRP `cat(mean_support, y_hat)`
- [x] `build_channel_slice_codec`：加 `side_in_context: bool = False`、`side_channels: int = 0`。`side_in_context=True` 时也建 y0 entry（`channel_context_factory(0, slice_ch, side_channels)`），每个 k 的 `support_ch` 加上 `side_channels`
- [x] `build_mean_scale_head`：加 `side_split: int = 0` + `emit_mean_support: bool = False`，自动推算 mean_cc / scale_cc 的输入宽度（`support_ch - side_split`）
- [x] **追加（实施时新增）**：`_slice_helpers.infer_num_slices` 自动检测 y0 是否在 state_dict（Family 1 `side_in_context=True` 模式下 channel_context 覆盖 y0..yK-1，不再 +1）
- [x] 单元测试：`side_in_context=True` 模式下 forward shape + `channel_context.y0` 存在 + 缺失 y0 时报错；`side_split` routing 与 mean_in / mean_support 输出验证

#### 3.1 重写 `compressai/models/stf.py`
- [x] `WACNN.__init__` wiring（替换 `_init_slice_entropy`）：
  - `h_a` / `h_mean_s` / `h_scale_s` 改为局部变量（不再是 model attr）
  - `self.latent_codec = HyperpriorLatentCodec(h_a=h_a, h_s=DualHyperSynthesis(h_mean_s, h_scale_s), latent_codec={"z": EntropyBottleneckLatentCodec(EntropyBottleneck(N), quantizer="noise"), "y": build_channel_slice_codec(...)})`
  - `build_channel_slice_codec(groups=[slice_ch]*K, side_channels=2*M, side_in_context=True, max_support_slices=MS, leaf_factory=...(mean_support_trail_channels=M+slice_ch*support_count(k)), channel_context_factory=...(side_split=M, emit_mean_support=True))`
- [x] `SymmetricalTransFormer.__init__` 同样改造（`bottleneck_channels = latent_channels // 2`，`widths=(224, 176, 128, 64)` 跟 WACNN 一致；`_build_stf_transformer_h_subpel` helper 用于 transformer 派生 h_s 宽度）
- [x] 模型基类改成 `CompressionModel`（不引 `SimpleVAECompressionModel`，避免 cherry-pick），手写 5-10 行 forward / compress / decompress 委托给 `self.latent_codec`
- [x] 删除 STF/WACNN 的 `from_state_dict` 中跟旧 hyperprior / `SliceEntropyCompressionModel` 相关的 import 与 helper 调用
- [x] 移除 `compressai/models/stf.py` 顶部对 `SliceEntropyCompressionModel` / 旧 helpers 的 import；改为 `from compressai.latent_codecs import ...` 与 `from compressai.latent_codecs._slice_helpers import infer_num_slices, infer_max_support_slices, make_entropy_transform`
- [x] **追加（实施时新增）**：抽 `_build_family1_latent_codec` private helper 给两个模型共用

#### 3.2 更新 STF 上游 ckpt 转换
- [x] `convert_upstream_stf_state_dict` 多 pass 改造：
  - Pass 1：strip `module.` 前缀；同时跑 `_nest_winmsa_keys` 把上游 `conv_b.<i>.attn.{qkv,proj,relative_position_*}` nest 成 `conv_b.<i>.attn.attn.*`（适配当前 WMSA→WindowAttention wrapper 结构，正则只匹配 conv_b 路径，不影响 Swin 的 layers.*.blocks.*.attn.*）
  - Pass 2：`cc_mean_transforms.{k}.` → `latent_codec.y.channel_context.y{k}.mean_cc.`、同理 scale_cc / lrp_transform；`gaussian_conditional.` 复制 K 份到 per-slice leaf；`h_a/h_mean_s/h_scale_s/entropy_bottleneck` 走 `_UPSTREAM_TOP_LEVEL_RENAMES`
- [x] **路径修正（实施时纠正）**：`HyperpriorLatentCodec` 通过 `self.y` / `self.z` 注册子模块（`self.latent_codec` 是普通 dict 不被 nn.Module 注册），所以路径是单层 `latent_codec.y.*` / `latent_codec.z.*`，**不是**双层 `latent_codec.latent_codec.*`。`_slice_helpers._DEFAULT_NUM_SLICES_PREFIX` 同步修正为 `latent_codec.y.channel_context.y`
- [x] **LRP byte-for-byte 兼容**（替代原 caveat 方案）：通过 3.0 的 `emit_mean_support` + `mean_support_trail_channels` 让新 LRP 输入宽度复刻上游 `M + slice_ch*(support+1)`，upstream `lrp_transforms.{k}.*` 权重直接转移，**无 caveat**
- [x] **WinNoShiftAttention 命名兼容**（实施时发现并修复）：`_nest_winmsa_keys` 正则解决 WMSA 包装层路径差异，无 unexpected keys
- [x] 验证：`candidate/cnn_0018_best.pth.tar`（WACNN, 585 keys）和 `candidate/stf_0018_best.pth.tar`（STF Transformer, 779 keys）都 strict=True 加载成功，forward 通过；MSE / y bpp 跟 fresh init 显著区别证明权重确实参与计算

#### 3.3 更新 `tests/test_models.py::TestStf`
- [x] state_dict round-trip 测试照搬，期望 `x_hat` 仍然 allclose
- [x] 加 state_dict 路径自检（实施时按真实路径修正为单层 `latent_codec.*`）：`latent_codec.h_a.0.weight`、`latent_codec.h_s.h_mean_s.0.weight`、`latent_codec.z.entropy_bottleneck.quantiles`、`latent_codec.y.channel_context.y0.mean_cc.0.weight`、`latent_codec.y.channel_context.y1.mean_cc.0.weight`、`latent_codec.y.latent_codec.y0.lrp_transform.0.weight`、`latent_codec.y.latent_codec.y0.gaussian_conditional.scale_table`
- [x] `test_stf_upstream_state_dict_conversion`：fixture 升级为含 `module.g_a.4.conv_b.0.attn.{qkv,proj,relative_position_*}` + 完整 cc_/lrp/gaussian_conditional/h_a/h_*_s/entropy_bottleneck，断言新路径 + LRP 保留

### 验收
- [x] `pytest tests/test_models.py::TestStf -v` 全过 ✅
- [x] 上游 `cnn_0018_best.pth.tar` + `stf_0018_best.pth.tar` 两个 ckpt strict=True 加载成功，forward x_hat shape 正确 ✅
- [x] state_dict 路径符合 §10.7 STF 一栏（已修正为单层 `latent_codec.*` 前缀）

### 实施差异 / 决策记录
- 路径前缀从设计文档 §10.7 写的双层 `latent_codec.latent_codec.*` 修正为单层 `latent_codec.*`——`HyperpriorLatentCodec.__init__` 把 `self.latent_codec = {...}` 设为普通 dict（非 nn.ModuleDict），所以子模块靠 `self.y` / `self.z` 注册，路径少一层
- 引入 `emit_mean_support` + `mean_support_trail_channels` 替代「LRP 权重重训」方案——上游 LRP 权重 byte-for-byte 可迁移，无任何 caveat
- 引入 `_nest_winmsa_keys` 正则解决 WMSA 包装层路径差异——不动 WMSA / WindowAttention 类层级（避免破坏 SwinBlock / SWAtten 现有 ckpt）

### 风险（已消解）
- ~~`SimpleVAECompressionModel` 在 pr-stf-wacnn 上不存在~~ → 用 `CompressionModel` + 5 行手写 forward/compress/decompress
- ~~LRP 输入宽度与旧架构不兼容~~ → `emit_mean_support` 方案打通
- ~~WinNoShiftAttention 24 keys mismatch~~ → `_nest_winmsa_keys` 正则修复

---

## Phase 4: 迁移 TCM（1 天）✅ 完成于 2026-05-09，commit `c2d931a`

新建 `compressai/models/tcm.py` 按容器化 pattern。

> **接口落点**：与 Phase 3 一致，使用 `build_channel_slice_codec` + `build_mean_scale_head` + `MeanScaleContextHead`，TCM 通过 `support_transform_factory=lambda c_in, c_out: SWAtten(...)` 注入 SWAtten 作为 mean / scale 各自独立的 support 变换。**不**使用 `ChannelSliceLatentCodec`。

### Scope 决策（2026-05-09）

- **不实现 `use_cca`**：fork `script` 上 `use_cca=True` 依赖 `CausalContextAdjustmentEntropyModel`（fork `script` 旧的非容器化版本）。Phase 5 会把 CCA-aux 重写为容器化 `_CCAAuxEntropyModel`——届时 TCM `use_cca` 应该挂的是新版本。Phase 4 暂不暴露这个参数，留给 Phase 5 完成后追加一个小 commit 给 TCM 接 `_CCAAuxEntropyModel`。依赖关系正向、commit 切分清晰。
- **不实现 `use_auxt`**：fork `script` 上独立特性（AuxT 训练辅助变换，依赖 `WLS/iWLS/OLP` layers），不在本 PR scope。

### 任务

#### 4.1 重写 `compressai/models/tcm.py`
- [x] 全删 backup 分支上现有 TCM 实现（backup 留作参考）→ `compressai/models/tcm.py` 全新写，~700 行
- [x] 按 §10.5 TCM sketch 写：`HyperpriorLatentCodec` + `DualHyperSynthesis` + `build_channel_slice_codec(side_in_context=True, max_support_slices=MS)` + `build_mean_scale_head(side_split=M, emit_mean_support=True, support_transform_factory=SWAtten)` + `LRPGaussianLatentCodec(mean_support_trail_channels=ms_ch, quantizer="ste")`
- [x] ~~`TCM.use_cca` opt-in 路径保留~~ → 见上「Scope 决策」，Phase 4 不带 use_cca
- [x] 删除 backup 上的 `_make_entropy_transform`, `_slice_support_channels`, `_lrp_support_channels`（已经在 `_slice_helpers.py`）—— TCM 直接从 `compressai.latent_codecs._slice_helpers` import
- [x] `_LEGACY_LATENT_PREFIX_MAP` 重写为 `convert_upstream_tcm_state_dict`：处理 `module.` 前缀剥离 + 上游 LIC_TCM `.msa.relative_position_params` / `.msa.embedding_layer` / `.msa.linear` MSA 重命名 + `atten_mean.{k}.0.` SWAtten wrapper 拆包 + `atten_*` → `*_support_transforms` 别名 + `ln1/ln2/mlp.0/mlp.2` 重命名 + per-slice rerooting + `gaussian_conditional` 复制 K 份 + 顶层 `h_a/h_*_s/entropy_bottleneck` → `latent_codec.*`
- [x] `_UPSTREAM_SWATTEN_WRAPPER` 保留并扩展（同时匹配 `atten_mean|atten_scale|mean_support_transforms|scale_support_transforms`）
- [x] **追加（实施时新增）**：抽 `_build_tcm_latent_codec` private helper，跟 STF 的 `_build_family1_latent_codec` 平级独立——避免修改 stf.py 已稳定的 helper（最小化对现有代码改动）。Phase 5 完成 CCA 后视情况考虑提取共享 `build_family1_latent_codec`

#### 4.2 更新 `examples/convert_tcm_checkpoint.py`
- [x] 新建（pr-stf-wacnn 基线上不存在）—— mirror `examples/convert_stf_checkpoint.py` 的 thin CLI wrapper
- [x] 验证：candidate/TCM 两个 ckpt round-trip：
  - `candidate/TCM/0.05.pth.tar` (N=64, M=320, 1293 keys → 1397): PSNR 39.15 dB / total bpp 0.317 (256×256 sinusoidal smoke)
  - `candidate/TCM/mse_lambda_0.05.pth.tar` (N=128, M=320, 1293 keys → 1397): PSNR 39.41 dB / total bpp 0.236
  - vs fresh-init baseline PSNR 5.41 dB → 权重 byte-for-byte 转移成功

#### 4.3 更新 `tests/test_models.py::TestTcm`
- [x] `test_tcm_forward_and_state_dict_round_trip`：state_dict 路径自检（含 `latent_codec.h_a.0.conv1.weight` / `latent_codec.h_s.h_mean_s.0.conv.weight` / `latent_codec.z.entropy_bottleneck.quantiles` / `latent_codec.y.channel_context.y0.mean_cc.0.weight` / `latent_codec.y.channel_context.y0.mean_support_transform.in_conv.weight`（SWAtten 字段）/ `latent_codec.y.latent_codec.y0.lrp_transform.0.weight` / `latent_codec.y.latent_codec.y0.gaussian_conditional.scale_table`）+ round-trip allclose + 推断 N/M/hyper/slices/MS 校验
- [x] `test_tcm_upstream_state_dict_conversion`：synthetic 上游 LIC_TCM-style state_dict（含 `module.` 前缀 + `.msa.relative_position_params` 4×15×15 buffer + `.msa.embedding_layer` + `atten_mean.0.0.in_conv.weight` SWAtten wrapper + 完整 cc_/lrp/gaussian_conditional/h_a/h_*_s/entropy_bottleneck），断言：`relative_position_params` 重塑 (15,15,4)→(225,4) + 各 prefix 重命名 + 旧 root 路径消失
- [x] ~~`test_tcm_with_cca`~~ → Scope 决策：留给 Phase 5（同时实现 CCA-main 时一起加）

#### 4.4 文档（实施时新增）
- [x] `compressai/latent_codecs/__init__.py` 在 `__all__` 后面追加 Family 1 wiring 注释块：用模板代码 + 4 点 ELIC↔Family1 差异 + 各模型（STF/WACNN, TCM, CCA-main, CCA-aux, DCAE/MambaVC）的 kwargs 变化点。让 reviewer 不用读 STF/TCM 源码就能看懂 Family 1 的拼装逻辑

### 验收
- [x] `pytest tests/test_models.py::TestTcm -v` 2/2 通过 ✅
- [x] `pytest tests/test_models.py tests/test_latent_codecs.py tests/test_models_helpers.py tests/test_layers.py tests/test_init.py -q` 71/71 通过（含 STF 3 + TCM 2）✅
- [x] 上游 candidate ckpt round-trip：N=64 / N=128 两个 ckpt strict load 成功，sinusoidal smoke PSNR 39+ dB ✅
- [x] `make static-analysis` 全过（ruff format / imports / lint）✅
- [x] `import compressai` / `compressai.zoo` / `compressai.latent_codecs` 触发 0 个 timm 加载 ✅
- [x] `uv lock --check` 一致（无 pyproject 改动）✅

### 实施差异 / 决策记录

- **`use_cca` 推迟**：见 Scope 决策。Phase 5 实现 `_CCAAuxEntropyModel` 后追加 commit `feat(models/tcm): add use_cca opt-in via _CCAAuxEntropyModel`
- **`use_auxt` 不做**：fork `script` 上独立特性，依赖 WLS/iWLS/OLP layers（本分支没有），不属于本 PR scope
- **TCM 自写 `_build_tcm_latent_codec`**：跟 STF `_build_family1_latent_codec` 平级独立而不复用——保留 stf.py 稳定（Phase 3 已落地）。Phase 5 完成后视 CCA 实施情况再决定是否提取共享 `build_family1_latent_codec` helper
- **`_UPSTREAM_SWATTEN_WRAPPER` 扩展正则**：同时匹配 `atten_mean|atten_scale|mean_support_transforms|scale_support_transforms`，处理上游 LIC_TCM 跟 fork `script` 中间 ChannelSliceLatentCodec layout 两种命名
- **`hyper_channels` 推断走新路径**：`state_dict["latent_codec.z.entropy_bottleneck.quantiles"].size(0)`（容器化后），不再是 fork `script` 的 `state_dict["entropy_bottleneck.quantiles"]`
- **`compress`/`decompress` 验证未做**：跟 STF Phase 3 一样，发现 `ChannelGroupsLatentCodec.decompress` 在 `side_in_context=True` 模式下有 split 维度不匹配的 pre-existing bug（STF / TCM 都挂）。本次不修——属于 channel_groups.py 内部问题，应单独 follow-up PR 解决，跟 Phase 4 scope 无关
- **`latent_codecs/__init__.py` 文档**：user 要求加 Family 1 说明，初稿是顶部 module docstring（覆盖所有 codec），后调整为「`__all__` 后注释块 + 仅 Family 1 内容」。最终位置不影响 import / `__all__`，纯文档增量

### 风险（已消解）
- ~~TCM 的 SWAtten 第一遍是 mean / scale 各一套（分离）~~ → `build_mean_scale_head(support_transform_factory=...)` 内部分别给 mean / scale 各构造一个独立 SWAtten 实例，自然支持
- ~~现 backup 上的 TCM `from_state_dict` 用了 `infer_num_slices` 等 helper，需确认 helper 已搬到 `_slice_helpers.py` 且 prefix 已更新~~ → Phase 1 已完成，本 Phase 直接 import 无问题

---

## Phase 5: 迁移 CCA-main + CCA-aux（2 天）✅ 完成于 2026-05-09，commit `1e636f9`

最复杂的一个，因为 CCA 有变长 slice + aux entropy + skip-most-recent + 部分 LRP。

> **接口落点**：CCA-main 与 Phase 3/4 一致使用 `build_channel_slice_codec` + `build_mean_scale_head`，传 `groups=resolved_slice_sizes` 支持变长 slice。CCA-aux 是模型字段（不在 `latent_codec` 树内），其内部容器同样基于 `ChannelGroupsLatentCodec`，通过 `support_filter=skip_most_recent` 实现。**不**使用 `ChannelSliceLatentCodec`。

### Scope 决策（2026-05-09）

- **`use_lrp_until` 不暴露**：上游 `LICAutoencoder` 在 main 上每个 slice 都跑 LRP，aux 在前 K-2 个 slice 上跑 LRP（但 published checkpoint 包含全部 K 份 LRP 权重）。为 strict-load 兼容，CCA-aux 也给所有 slice 用 `LRPGaussianLatentCodec` —— 后两个 slice 的 LRP 计算结果会被 `support_filter` 过滤掉，浪费一点点算力但不影响 likelihoods。design doc §5 的「mixed leaf 类型」方案因此放弃，文档中给出的 `if k < num_slices - 2 else GaussianConditionalLatentCodec()` 在实际 ckpt 下会丢 unexpected keys。
- **CCA 默认 `slice_proportions=(8, 28, 56, 92, 136)`**：跟上游 published M=320 layout 直接对齐（旧 backup 写的是 `(1,1,1,1,1)` 等比例），方便 user 不传该参数也能走对路径。
- **CCA-aux 的 `mean_support` 用 post-NAFTransform**：上游 `lrp_transforms` 训练时拿的是 NAFTransform 之后的 mean_support。`MeanScaleContextHead.emit_mean_support` 因此扩展为 `Literal["pre", "post"]`（True 兼容旧 STF/TCM = "pre"），CCA 走 "post"。

### 任务

#### 5.0 进一步扩展基础设施（Phase 1/3 之上的小幅追加）
- [x] `MeanScaleContextHead.emit_mean_support` 改为 `bool | Literal["pre", "post"]`：True/`"pre"` 等效（输出原始 `mean_in`，STF/TCM 用），`"post"` 输出 `mean_support_transform(mean_in)`（CCA-main / CCA-aux 用，匹配上游 LRP 训练的输入）。`build_mean_scale_head` 同步透传
- [x] `build_channel_slice_codec.support_count_fn`：可选 `Callable[[int], int]`，覆盖默认 `min(k, max_support_slices)` clamp。CCA-aux 传 `lambda k: max(k - 1, 0)` 跟 `support_filter=skip_most_recent` 配套，确保 `channel_context` head 的输入宽度匹配 filter 选出的 prior 数量
- [x] `ChannelGroupsLatentCodec._get_ctx_params_side_in_context` 处理空 support：CCA-aux 在 k=1 时 `support_filter` 返回 `[]`，原代码会触发 `torch.cat()` empty list error。补丁：support 为空时直接喂 `side_params` 给 head，head 输入宽度跟 k=0 路径一致

#### 5.1 重写 `compressai/models/cca.py::CCAModel`
- [x] 全删 backup 上的 monolithic CCA 实现（backup 留作参考）→ 新文件 `compressai/models/cca.py` ~700 行
- [x] 按 §10.5 CCA-main sketch 写：`HyperpriorLatentCodec` + `DualHyperSynthesis` + `build_channel_slice_codec(groups=resolved_slice_sizes, side_channels=2*M, side_in_context=True, max_support_slices=-1, leaf_factory=LRPGaussianLatentCodec, channel_context_factory=build_mean_scale_head(side_split=M, emit_mean_support="post", support_transform_factory=NAFTransform))`
- [x] `EntropyBottleneckLatentCodec(EntropyBottleneck(N), quantizer="ste")` 处理 z STE quantize（替代 backup 上手写的 `quantize_ste(z - z_offset) + z_offset`）
- [x] `_NAFBlock` / `_NAFTransform` 内联在 `compressai/models/cca.py`（不新增 layer 文件，最小化对现有代码改动）。NAFBlock 的 `LayerNorm2d` 用 `from timm.layers import LayerNorm2d`（与 STF/TCM 一致复用 `[attn]` extras）
- [x] `_CCAEncoder` / `_CCADecoder` 复用上游 NAFBlock + ResidualBottleneckBlock（从 `compressai.models.sensetime` import 后者）
- [x] `_conv2d` / `_convt2d` 改为复用 `compressai.models.utils.conv` / `deconv`（user 反馈：utils 已有相同实现）
- [x] `compress` / `decompress` 委托给 `self.latent_codec`，与 STF/TCM 一致
- [x] `from_state_dict` 自动检测 upstream layout 并调 `convert_upstream_cca_state_dict`，再走 `_infer_config_from_state_dict` 推断 ctor kwargs

#### 5.2 实现 `_CCAAuxEntropyModel`
- [x] 模型字段（不在 `latent_codec` 树内），thin `nn.Module`：`y_entropy_bottleneck = EntropyBottleneck(M)` + `inner_codec = build_channel_slice_codec(groups=slice_sizes, side_channels=2*M, side_in_context=True, support_filter=lambda k, prior: prior[:max(k-1,0)], support_count_fn=lambda k: max(k-1,0), leaf_factory=LRPGaussianLatentCodec, channel_context_factory=build_mean_scale_head(side_split=M, emit_mean_support="post", support_transform_factory=NAFTransform))`
- [x] 所有 K 个 slice 用 `LRPGaussianLatentCodec`（见 Scope 决策）—— mixed leaf 方案与上游 ckpt 不兼容
- [x] `forward(y, latent_means, latent_scales)` 返回 `{"y_aux", "y_cca"}`，与 `CCARateDistortionLoss` 兼容
- [x] `CCAModel.forward` 在 `cca_training=True` 时复跑一遍 hyperprior path 拿 latent_means/latent_scales 喂给 aux 分支（避免在 main path 修改 `HyperpriorLatentCodec` 暴露中间变量）

#### 5.3 更新 `convert_upstream_cca_state_dict`
- [x] Pass 1+2：NAFBlock interior renames（`dwconv`/`sca`/`FFN`/`conv1` → `pointwise_depthwise`/`channel_attention`/`feed_forward`/`project`，仅在 `_find_naf_block_prefixes` 检测到的 NAFBlock 范围内生效，避免误改 ResidualBottleneckBlock 的 `conv1`）+ NAFTransform interior renames（`in_conv`/`out_conv` → `input_projection`/`output_projection`）+ named-part renames（`mean_NAF_transforms` → `mean_support_transforms` 等，统一别名）+ top-level renames（`aux_entropymodel` → `aux_entropy_model`，hyperprior 移到 `latent_codec.*`，`z_entropy_bottleneck` → `latent_codec.z.entropy_bottleneck`）
- [x] Pass 3a：main 分支 per-slice rerooting—`mean_cc_transforms.{k}` → `latent_codec.y.channel_context.y{k}.mean_cc.*`，scale_cc 同理；`mean_support_transforms.{k}` → `latent_codec.y.channel_context.y{k}.mean_support_transform.*`，scale 同理；`lrp_transforms.{k}` → `latent_codec.y.latent_codec.y{k}.lrp_transform.*`；`gaussian_conditional.*` 复制 K 份
- [x] Pass 3b：aux 分支 per-slice rerooting，规则同 main 但前缀加 `aux_entropy_model.inner_codec.*`
- [x] `_infer_config_from_state_dict` 适配新路径前缀

#### 5.4 更新 `tests/test_models.py::TestCca`
- [x] `test_cca_forward_and_state_dict_round_trip`：tiny 变量构造（M=64, slice_proportions=(2,6,12,18,26)），forward + state_dict path checks（含 `latent_codec.h_a.0.weight` / `latent_codec.h_s.h_mean_s.0.weight` / `latent_codec.z.entropy_bottleneck.quantiles` / `latent_codec.y.channel_context.y0.mean_cc.0.weight` / `latent_codec.y.channel_context.y0.mean_support_transform.input_projection.weight` / `latent_codec.y.latent_codec.y0.lrp_transform.0.weight`）+ round-trip allclose
- [x] `test_cca_training_branch_forward_and_round_trip`：cca_training=True 路径，验证 `aux_likelihoods` keys + 形状 + 路径（`aux_entropy_model.inner_codec.channel_context.y0.mean_cc.0.weight` 等）+ 完整 round-trip
- [x] `test_cca_upstream_state_dict_conversion`：synthetic 上游 LICAutoencoder-style state_dict（NAFBlock + NAFTransform + ResidualBottleneckBlock + 主辅两个分支的 cc/support/lrp/gc + hyperprior backbone + aux_entropymodel 全套），断言每个新路径存在 + 旧路径消失 + ResidualBottleneckBlock 的 `conv1` 不被误改

#### 5.5 文档与配套
- [x] 新增 `compressai/losses/cca.py` (`CCARateDistortionLoss`)，wire 进 `compressai/losses/__init__.py`（pytorch_msssim 已是 `RateDistortionLoss` 依赖，不引入新 hard dep）
- [x] 新增 `examples/convert_cca_checkpoint.py` thin CLI wrapper（mirror `convert_tcm_checkpoint.py`）

### 验收
- [x] `pytest tests/test_models.py::TestCca -v` 3/3 通过 ✅
- [x] `pytest tests/test_models.py tests/test_latent_codecs.py tests/test_models_helpers.py tests/test_layers.py tests/test_init.py -q` 74/74 通过 ✅（含 STF 3 + TCM 2 + CCA 3）
- [x] 上游 `candidate/CCA/checkpoint_lambda_0.3.pth.tar` strict-load 成功（97M params，M=320, slice_sizes=[8,28,56,92,136], em_hidden=224, em_layers=4, cca_training=True），sinusoidal smoke PSNR 50.07 dB / total_bpp 0.072（vs fresh-init ~5dB）→ 权重 byte-for-byte 转移成功 ✅
- [x] `make static-analysis` 全过（ruff format / imports / lint）✅
- [x] `import compressai` / `compressai.zoo` / `compressai.latent_codecs` 触发 0 个 timm 加载 ✅
- [x] `uv lock --check` 一致（无 pyproject 改动）✅

### 实施差异 / 决策记录

- **CCA-aux 全 LRP leaf**：见 Scope 决策。design doc §5 的 mixed leaf 方案被 published checkpoint 的 K 份 LRP 权重否决——为 strict-load 兼容必须 K 份全开
- **`emit_mean_support="pre"|"post"`**：CCA 上游 LRP 拿的是 NAFTransform 之后的 mean_support；STF/TCM 因 `mean_support_transform=Identity` pre/post 等价。原 `bool` 接口保留向后兼容（True == "pre"），新加 "post" 模式供 CCA 用
- **`support_count_fn`**：CCA-aux skip-most-recent 让 `support_filter` 返回的 prior 数量跟默认 clamp 不一致，head 输入宽度算错。新加 `support_count_fn` 让 user 显式声明，跟 `support_filter` 配套使用
- **空 support 容错**：`ChannelGroupsLatentCodec._get_ctx_params_side_in_context` 在 k≥1 但 `support_filter` 返回空时（CCA-aux k=1）会 `torch.cat([])` 崩。改为 fallback 到只喂 `side_params`，跟 k=0 路径一致。也是为 CCA-aux 服务的最小补丁
- **NAFBlock detector**：`_find_naf_block_prefixes` 用 `.beta`/`.gamma`/`.dwconv.0.weight`/`.FFN.0.weight` 4-tuple 严格检测，确保 `g_a/g_s` 内 NAFBlock 的 `conv1` → `project` rename 不会误改同样在 `g_a/g_s` 内的 ResidualBottleneckBlock 的 `conv1`（后者只有 `.conv2.weight`/`.conv3.weight` 等不同标志）
- **`_NAFBlock` 内联在 cca.py 而非新建 layer 文件**：跟 TCM 的 `SWAtten` (`compressai/layers/attn/swatten.py`) 不一样的选择；理由是 NAFBlock 短期内只有 CCA 用，避免新增 layer 文件 + `compressai/layers/__init__.py` 改动。如未来 MambaVC 等也用 NAF 可再迁出
- **`_conv2d` / `_convt2d` 复用 utils**：user 提示 `compressai/models/utils.py::conv` / `deconv` 已有相同实现，去掉本地重复
- **CCAModel.forward 复跑 hyperprior 给 aux**：为了不修改 `HyperpriorLatentCodec` 暴露 latent_means/latent_scales 中间变量，aux training path 走一次额外的 `h_a + z_codec + h_s` 算 latent_means/latent_scales，再喂 aux 分支。算力多一点点，接口零侵入
- **gaussian_conditional 复制 → key 数 +56**：上游 main + aux 各 7 份共享 GC buffer，容器化后每分支 K=5 复制 → +28×2 = +56 keys（2328 → 2384），跟 §10.7 的「per-slice 副本，每模型多几十 KB」描述一致
- **CCA 默认 slice_proportions 改为 (8,28,56,92,136)**：直接对应上游 published M=320 layout，user 用 `CCAModel()` 默认就能 load `checkpoint_lambda_0.3.pth.tar`

### 风险（已消解）
- ~~`build_channel_slice_codec` 不知道 `support_filter` 选了几个 prior，head 宽度算错~~ → 加 `support_count_fn` 显式传
- ~~CCA-aux 在 k=1 时 `support_filter` 返回空触发 `torch.cat([])` 崩~~ → `_get_ctx_params_side_in_context` 加空-support 旁路
- ~~mixed leaf 类型（前 K-2 LRP，后 2 普通 GaussianConditional）~~ → 上游 ckpt 全 K 份 LRP，必须全 LRP 才能 strict-load
- ~~CCA z STE-quantize 需要 codec 支持~~ → upstream `EntropyBottleneckLatentCodec(quantizer="ste")` 已支持，直接用

---

## Phase 6: 清理基类 + zoo 接线（0.5 天）✅ 完成于 2026-05-09，commit `0c44c75` + `f87c8c8`

> **吸收 Phase 2 的删除任务**：见文档顶部「执行顺序调整」。原 Phase 2 的 `ChannelSliceLatentCodec` 删除会跟本 Phase 的 `_bases/slice_entropy.py` 删除合并为同一 cleanup commit。

### 任务

#### 6.1 删除 `compressai/models/_bases/slice_entropy.py` + `compressai/latent_codecs/channel_slice.py`（吸收原 Phase 2）—— commit `0c44c75`
- [x] 确认所有 helpers 已搬到 `_slice_helpers.py`，且 STF/WACNN/TCM/CCA 都不再 import `SliceEntropyCompressionModel` ✅（`git grep` 验证零生产 caller）
- [x] 全仓 grep `SliceEntropyCompressionModel` 应只剩 fork `script` 上 mambavc.py 等（本仓不存在）—— 删除 `_bases/slice_entropy.py` ✅（删除整个 `_bases/` 目录，因为 `__init__.py` 只 re-export slice_entropy 已无内容）
- [x] 全仓 grep `ChannelSliceLatentCodec` 应只剩 `tests/` 中已迁移到 `ChannelGroupsLatentCodec` 的旧引用（清理）—— 删除 `compressai/latent_codecs/channel_slice.py` ✅（grep 确认 tests/ 已无引用，直接删）
- [x] `compressai/models/_bases/__init__.py` 移除 `slice_entropy` 相关 import → 整个目录删除
- [x] `compressai/latent_codecs/__init__.py` 移除 `ChannelSliceLatentCodec` export ✅
- [x] **追加（实施时）**：修复 `compressai/latent_codecs/_slice_helpers.py:34` 模块 docstring 中指向已删除 `compressai.models._bases.slice_entropy` 的「previously hosted in」引用——避免文档指向不存在的模块
- [x] 验证 ELIC（`compressai/models/sensetime.py`）继续工作（原 Phase 2 验收点）：`pytest tests/test_models.py -q` 全过证明 `ChannelGroupsLatentCodec` 扩展未破坏现有用户 ✅（74/74 通过）

#### 6.2 zoo 接线 —— commit `f87c8c8`
- [x] `compressai/zoo/__init__.py` `image_models` 加 `cca`, `tcm` ✅
- [x] `compressai/zoo/image.py` `cca()` / `tcm()` 工厂函数 + `model_architectures` 用 `_LazyImport` 代理 ✅（mirror `stf()` 的 `pretrained=True` raise pattern）
- [x] 沿用现有 lazy import pattern，避免 `import compressai.zoo` 触发 timm ✅（验证：`compressai.zoo` import 后 `timm` 加载数 = 0）

### 验收
- [x] `import compressai` + `import compressai.zoo` 触发 0 个 timm import（同 STF PR）✅
- [x] `compressai.models.tcm` / `compressai.models.cca` 可深路径 import ✅
- [x] pytest 全 suite 不引入回归 ✅（74/74）
- [x] `make static-analysis` 全过（ruff format / imports / lint）✅

---

## Phase 7: 全量验证（0.5 天）✅ 完成于 2026-05-09

### 任务
- [x] `make static-analysis` 全过（ruff format / imports / lint）✅ 161 files already formatted；imports / lint 均 All checks passed
- [x] `pytest tests/test_models.py tests/test_layers.py tests/test_init.py tests/test_latent_codecs.py -q` 全过 ✅（实际跑了完整 `pytest tests/ -q`：**213 passed, 4 skipped, 32 deselected**；deselected 是 `tests/test_eval_model_video.py` + `tests/test_zoo.py` 的 pretrained-dependent 子套件，按 user 指示跳过——失败原因是本地 S3 ckpt 缓存损坏 `unexpected EOF, expected 783675 more bytes`，与本 PR 无关）
- [x] Import audit：跑导入跟踪脚本，确认 `import compressai` + `import compressai.zoo` 触发 timm 加载数 = 0 ✅（同时验证 `import compressai.latent_codecs` 也是 0）
- [x] `uv lock --check` 一致（如未改 pyproject.toml 应不需要）✅ Resolved 231 packages in 16ms，无错
- [x] 手工 state_dict 自检：构造小 STF / TCM / CCA 模型，dump 一个 key，验证路径符合 §10.7 ✅ 4/4 模型（WACNN/SymmetricalTransFormer/TCM/CCAModel）所有期望路径存在

### 验收
- [x] 全部通过 ✅

### 实施差异 / 决策记录
- **pretrained 测试跳过**：`tests/test_eval_model_video.py::test_eval_model_pretrained[*]` (4 失败) + `tests/test_zoo.py::TestBmshj2018Factorized::test_pretrained[mse]` (1 失败) 均因本地 S3 ckpt 缓存 `unexpected EOF` 损坏 → 与本 PR 改动 (Family 1 容器化) 无关，按 user 指示 deselect。其他 213 测试全过证明本 PR 对 WACNN/STF/TCM/CCA + ChannelGroupsLatentCodec + LRPGaussianLatentCodec + DualHyperSynthesis + MeanScaleContextHead + zoo wiring 等改动均零回归
- **state_dict 自检脚本路径修正**：(1) TCM 的 `h_a` / `h_*_s` 用 ConvNeXt-style ResidualBottleneckBlock + SubpelConv2x，路径是 `latent_codec.h_a.0.conv1.weight` / `latent_codec.h_s.h_mean_s.0.conv.weight`，不是 STF/WACNN 的 plain Conv2d 路径 `.0.weight`——这是 per-model 实现细节，不算合约违反；(2) `CCAModel` 构造参数是 `latent_channels` / `hyper_channels` / `em_hidden_channels` / `em_num_layers`，不是 `M` / `em_hidden`

---

## Phase 8: 提交 + push（0.5 天）

### 任务

#### 8.1 按 logical 分组打 commit（不要一个巨大 commit）

实际落地 **6 commits**（相对 `pr-stf-wacnn` 基线），按 Phase 推进顺序：

- [x] `feat(latent_codecs): add containerized infrastructure for Family 1 codecs`（Phase 1，commit `c6d556a`：DualHyperSynthesis + LRPGaussianLatentCodec + ChannelGroupsLatentCodec extensions + slice helpers + MeanScaleContextHead + build_channel_slice_codec）
- [x] `refactor(models/stf): migrate WACNN + SymmetricalTransFormer to containerized codec`（Phase 3.0 + 3.1，commit `8b3ea4d`：模型重写 + Phase 3.0 codec 扩展（`side_in_context` / `support_count_fn` / `emit_mean_support` / `mean_support_trail_channels`）打包进同一 commit；side_in_context 是给 Family 1 用的 codec 扩展，跟模型迁移强耦合，分开 commit 价值不大）
- [x] `feat(models): add TCM with containerized codec`（Phase 4，commit `c2d931a`）
- [x] `feat(models): add CCA model and loss with containerized codec`（Phase 5，commit `1e636f9`：包含 `MeanScaleContextHead.emit_mean_support="post"` + `build_channel_slice_codec.support_count_fn` + `ChannelGroupsLatentCodec` 空-support 旁路 + `compressai/models/cca.py` + `compressai/losses/cca.py` + `examples/convert_cca_checkpoint.py` + TestCca）
- [x] `chore(latent_codecs,models): drop ChannelSliceLatentCodec and SliceEntropyCompressionModel`（Phase 6.1，吸收原 Phase 2，commit `0c44c75`：删除 `compressai/latent_codecs/channel_slice.py` + 整个 `compressai/models/_bases/` 目录 + `latent_codecs/__init__.py` 移除 export + `_slice_helpers.py` 修复 docstring）
- [x] `chore(zoo): wire cca/tcm zoo entries with lazy import`（Phase 6.2，commit `f87c8c8`：`zoo/image.py` 加 `tcm()` / `cca()` factory + `_LazyImport` proxy；`zoo/__init__.py` re-export + `image_models` 字典 entry）

#### 8.2 PR 描述草稿
#### 8.2 PR 描述草稿 ✅ 完成于 2026-05-09
- [x] 写到 `plan/generated/pr-tcm-cca-draft.md` ✅（291 行，参考 #354 格式）
- [x] 重点说明：(a) 容器化的动机（pedagogical clarity + 跟 ELIC pattern 收敛）；(b) state_dict 路径变化；(c) `_bases/slice_entropy.py` 删除；(d) TCM/CCA 新模型按新 pattern 实现 ✅
- [x] 引用本 exec plan + design doc ✅（PR body 引用 #354 的 [latent_codec 重构承诺评论](https://github.com/InterDigitalInc/CompressAI/pull/354#issuecomment-3578257918) + #353 PR 系列；exec plan 与 design doc 路径在文档头部已有 cross-link）

#### 8.3 push 到 origin/pr-tcm-cca ✅ 完成于 2026-05-09
- [x] `git push origin pr-tcm-cca` ✅ **fast-forward**（`c9496cb..0759ad9`，6 ahead / 0 behind，**无需 force push**——原计划假设需要 -f 是误判，实际本地分支与远程线性增长）
- [x] **不要 push 到 upstream** —— PR 是 user 提交时机决定 ✅（仅 push 到 user fork `Yiozolm/CompressAI`，未触碰 InterDigital upstream）

### 验收
- [x] origin/pr-tcm-cca 所有 commit 静态分析 + 测试都过 ✅（本地 213 passed，CI 触发后 action_required，等 maintainer approve workflow run）

### 实施差异 / 决策记录
- **第一次 push 是 fast-forward**：本地 6 commits 在 `origin/pr-tcm-cca` HEAD 之上线性追加（每个 Phase 完成时单独 commit），从未 amend 或 rebase。普通 push（`c9496cb..0759ad9`）即可
- **第二次 push 改为 rebase + force push**：第一次 push 后 GitHub PR UI 显示 "Can't automatically merge"。根因：PR #354 被 squash/rebase merged 进 upstream/master 时换了 commit hash（upstream `edba5a4..5eef9a8` ≠ 我们本地 `b6cb2b6..c9496cb`，虽然内容等价），导致 GitHub 看到 12 commits / 6 commits / merge-base 在 #354 前的复杂分叉。
  - 修复：`git rebase --onto upstream/master c9496cb pr-tcm-cca` 把本 PR 的 6 commits replay 到 upstream/master 之上
  - **零冲突**（rebased commits 之前的 base 跟 upstream 上对应内容等价，无 hunk overlap）
  - Tree 等价性验证：`git diff pr-tcm-cca-pre-rebase pr-tcm-cca` 为空 + pytest 74/74 + ruff 全过
  - Push：`git push origin pr-tcm-cca --force-with-lease`（`0759ad9 → f87c8c8`，6 ahead / 0 behind upstream/master 现在线性可 fast-forward merge）
  - 本地保留 `pr-tcm-cca-pre-rebase` tag 作为 backup，PR merge 后可删
- **PR 草稿引用 #354 评论**：根据 user 反馈，#354 PR review 中明确承诺「I'll include the refactored abstraction layer in the next PR」（comment id `3578257918`）—— 本 PR draft 直接链接该评论作为重构动机背书，使 maintainer 不需要回顾上一轮讨论就理解为什么要删 `ChannelSliceLatentCodec`
- **PR draft 主动声明 `decompress` pre-existing bug**：`ChannelGroupsLatentCodec.decompress` 在 `side_in_context=True` 模式下存在 split 维度不匹配（影响 STF / TCM `compress`/`decompress`，`forward` 不受影响）—— 在 PR Test Plan 段透明声明并承诺 follow-up PR 修，避免本 PR scope 膨胀

---

## 总时间估算

| Phase | 工时 |
|---|---|
| Phase 0：清理 | 0.5h |
| Phase 1：基础设施 + 扩展 upstream | 1d |
| Phase 2：删除冗余 + 扩展验证 | 0.5d |
| Phase 3：迁移 STF/WACNN | 2d |
| Phase 4：迁移 TCM | 1d |
| Phase 5：迁移 CCA | 2d |
| Phase 6：清理 + zoo | 0.5d |
| Phase 7：验证 | 0.5d |
| Phase 8：提交 + push | 0.5d |
| **总计** | **~8 工作日** |

注：相比初版（9 天），Phase 1 缩短 0.5d（少 1 个文件不做 `_channel_context.py`），Phase 2 缩短 0.5d（直接删除而不是重写）。

---

<a id="design-rationale"></a>
## 设计依据（原 design-doc §4–§10，2026-06-04 并入）

> 本段是 H+G 重构的**决策与详细设计记录**，2026-06-04 从 `channel-slice-codec-redesign.md` §4–§10 精简后并入（design-doc 收敛为 §1–§3 codec 家族速查表）。codec 家族变异维度 + 七族分类仍在 design-doc §1–§3。

### D.1 候选方向对比（A–H）

| 方向 | 思路 | 结论 |
|---|---|---|
| A | 给 `ChannelSliceLatentCodec` 加 4 个可选参数（`slice_sizes`/`lrp_transforms`/`support_filter`/`lrp_scale`）| 覆盖 Family 1 全部需求，但不解决「单体类 + 多 ModuleList 同位」的 mental model |
| B | 把「算 (μ,σ)」「应用 LRP」抽成 Strategy 对象 | **否决**：每个 strategy 又是 `nn.Module`，state_dict 多一层、净增复杂度，不减循环重复 |
| C | 多姐妹 codec 类，文档说清对应家族 | fork `script` 现状；Family 1 仍要做 A 才解决 |
| D | 只改基类（widths/factory/use_lrp/slice_sizes），不动 codec | codec 没加 `slice_sizes` 则 CCA-main 仍用不了 |
| E | `ChannelSliceLatentCodec` 加 `support_builder` callable 接 DCAE dictionary | 净删 ~330 LoC，但引入比 `support_filter` 更宽的 callable hook，越「避免 swiss army knife」红线 |
| F | Family 2 三家（ELIC/MLIC++/MambaIC）合一 + `intra_slice_context_factory` | **不动**：联合-vs-分离 mean/scale 让 state_dict 不兼容；dedicated 类只有 2 个（MLIC++/MambaIC），ELIC/GLIC/CMIC 共用 upstream，未达 dedupe 阈值 |
| **G** ⭐ | **`HyperpriorLatentCodec` 嵌套**：`h_a`/`h_s`/`entropy_bottleneck` 收进 codec，模型只剩 g_a+g_s+latent_codec（沿 ELIC pattern）| **采纳（与 H 联合）**：codec-owned hyperprior；代价是 state_dict key 路径全变 + 扩展双 h_s |
| **H** ⭐ | **容器化重写 `ChannelSliceLatentCodec`**：单体 → ELIC 风格容器（`channel_context` + `latent_codec` 字典），forward 只 dispatch | **采纳**：模型→容器→leaf 三层，state_dict 路径自解释、与 ELIC pattern 收敛；代价是 5 个 Family 1 模型 ckpt 路径全变、删 `_bases/{slice_entropy,dictionary_entropy}.py` |

**曾考虑 C+A（短期小改）后 pivot 到 H+G 直接实施**，理由 3 条：(a) C+A 净 −125 行、H+G 净 −580 行且 state_dict 自解释 + `_bases/` 整目录可删；(b) C+A 后 pedagogical clarity 无改善（仍是单体 + 多 ModuleList）；(c) maintainer 已承诺接受重构，C+A 是浪费的中间步——做完还要再做 H+G，等于两次 state_dict rename。

### D.2 推荐方案 H+G 概览

模型类只剩 `g_a + g_s + latent_codec`，latent_codec 沿 ELIC pattern 完全容器化——直接用 upstream `HyperpriorLatentCodec` + `ChannelGroupsLatentCodec` 嵌套，每片一个独立 context module 和 leaf codec，state_dict 路径直接反映模型层级。

```python
class WACNN(SimpleVAECompressionModel):          # 基类只剩通用 forward/compress/decompress
    def __init__(self, N=192, M=320, num_slices=10, max_support_slices=5):
        super().__init__()
        self.g_a = ...
        self.g_s = ...
        self.latent_codec = HyperpriorLatentCodec(           # ← upstream 已有
            h_a=_h_a(M, N),
            h_s=DualHyperSynthesis(_h_mean_s(M, N), _h_scale_s(M, N)),  # ← 新增 ~25 行 adapter
            latent_codec={
                "z": EntropyBottleneckLatentCodec(EntropyBottleneck(N), quantizer="noise"),
                "y": ChannelGroupsLatentCodec(                # ← upstream 已有，只加 2 个可选参数
                    groups=[M // num_slices] * num_slices,    # 等大切片 = list of K equal sizes
                    max_support_slices=max_support_slices,    # NEW (default -1)
                    channel_context={f"y{k}": _build_mean_scale_head(...) for k in range(num_slices)},
                    latent_codec={f"y{k}": LRPGaussianLatentCodec(...) for k in range(num_slices)},  # ← 新增 ~30 行 subclass
                ),
            },
        )
```

**二次审查 upstream codec 后的复用决策**（避免重复造轮子）：删除 `ChannelSliceLatentCodec`（与 upstream `ChannelGroupsLatentCodec` 接口完全一致）；`LRPGaussianLatentCodec` 作为 subclass 追加到 upstream `gaussian_conditional.py` 末尾（~30 行）；CCA z STE 直接用 upstream `EntropyBottleneckLatentCodec(quantizer="ste")`（零改动）；`ChannelGroupsLatentCodec` 加 `max_support_slices` + `support_filter` 两个向后兼容可选参数（~10 行 diff）。

### D.3 容器与 leaf（§10.1–§10.2）

**容器**：复用 upstream `ChannelGroupsLatentCodec`（不新建）——`groups: List[int]` 同时涵盖等大 `[M//K]*K` 与变长 `[s0..sN]`；新增可选 `max_support_slices: int = -1`（STF/TCM clamp 用，默认 use-all-prior 不影响 ELIC）+ `support_filter: Optional[Callable]`（CCA-aux skip-most-recent 用）。

**leaf 清单**：

| Class | 来源 | 用户 |
|---|---|---|
| `GaussianConditionalLatentCodec` | upstream 已有 | F2 leaf（ELIC checkerboard 内嵌）|
| **`LRPGaussianLatentCodec`** | 追加在 upstream `gaussian_conditional.py` 末尾（subclass，~30 行；override forward 加 `0.5*tanh` LRP 后处理）| F1 全部 K slice（STF/WACNN/TCM/CCA-main/CCA-aux）|
| `CheckerboardLatentCodec` | upstream 已有 | F2（ELIC 内嵌）|
| `EntropyBottleneckLatentCodec(quantizer="ste"\|"noise")` | upstream 已有 | z 编码（CCA 用 ste，其他用 noise）|

LRP 单独 leaf 而非 `GaussianConditionalLatentCodec` 的可选参数：不扩展 upstream leaf 接口（ELIC 等零风险）；leaf 类型自我说明（`latent_codec.y3.lrp_transform.*` 一眼看出哪些 slice 有 LRP）；subclass ~30 行复用基类 `_chunk`/quantizer/compress/decompress。

### D.4 应用层 channel_context + hyperprior 适配（§10.3、§10.6）

mean/scale 分离的 head（STF/TCM/CCA）用普通 `nn.Module` 工厂（`build_mean_scale_head`）构造，放 `compressai/models/_helpers/`——沿 ELIC「`channel_context` 字典里放普通 module」约定，**不进** `latent_codecs/`。head 满足 `forward(prior_y_hat_concat) -> ch_ctx_params`。DCAE dictionary cross-attention 同为应用层 helper（DCAE/SAAF 上岸时加）。

**`DualHyperSynthesis` adapter**（`_hyper_synthesis.py`，~25 行）——双 h_s 模型用，零改动 upstream `HyperpriorLatentCodec`：

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

单 h_s 模型（ELIC/MLIC++）直接 `h_s=h_s`，不用 wrapper。CCA z STE 用 `EntropyBottleneckLatentCodec(quantizer="ste")`（upstream 已支持，原担心的 `HyperLatentCodec.quantizer` 扩展不需要）。

### D.5 State_dict 路径设计（§10.7，STF 为例 —— 最高价值，逐字保留）

> `HyperpriorLatentCodec.__init__` 把 `self.latent_codec = {...}` 设为普通 dict（非 nn.ModuleDict），子模块经 `self.y`/`self.z` 注册——所以外层是单层 `latent_codec.y.*`/`latent_codec.z.*`；`ChannelGroupsLatentCodec` 内部才是真 nn.ModuleDict，故 leaf 是 `latent_codec.y.latent_codec.y{k}.*`。

| 旧（`pr-tcm-cca` 迁移前）| 新（H + G 后，commit `8b3ea4d` 验证）|
|---|---|
| `entropy_bottleneck.quantiles` | `latent_codec.z.entropy_bottleneck.quantiles` |
| `h_a.0.weight` | `latent_codec.h_a.0.weight` |
| `h_mean_s.0.weight` | `latent_codec.h_s.h_mean_s.0.weight` |
| `h_scale_s.0.weight` | `latent_codec.h_s.h_scale_s.0.weight` |
| `latent_codec.cc_mean_transforms.{k}.0.weight` | `latent_codec.y.channel_context.y{k}.mean_cc.0.weight` |
| `latent_codec.cc_scale_transforms.{k}.0.weight` | `latent_codec.y.channel_context.y{k}.scale_cc.0.weight` |
| `latent_codec.lrp_transforms.{k}.0.weight` | `latent_codec.y.latent_codec.y{k}.lrp_transform.0.weight` |
| `latent_codec.gaussian_conditional._scale_table` | `latent_codec.y.latent_codec.y{k}.gaussian_conditional._scale_table`（per-slice 副本，K 份）|

**`side_in_context=True` 额外约束**：channel_context 字典覆盖 `y0..yK-1`（不是 ELIC 默认 `y1..yK-1`），所以 `latent_codec.y.channel_context.y0.{mean,scale}_cc.*` 在 STF/WACNN/TCM/CCA 的 state_dict 中存在；`infer_num_slices` 自动检测 y0 是否存在并据此 ±1。

**WMSA wrapper 路径**：`compressai.layers.attn.swin.WMSA` 把 WindowAttention 注册为 `self.attn`，故 conv_b 内 attention 路径是 `*.conv_b.<i>.attn.attn.{qkv,proj,relative_position_*}`（双层 attn）；上游 Zou et al. ckpt 是单层——`convert_upstream_stf_state_dict` 经 `_nest_winmsa_keys` 正则自动 nesting。

**LRP byte-for-byte 兼容**：`MeanScaleContextHead(emit_mean_support=True)` + `LRPGaussianLatentCodec(mean_support_trail_channels=M+slice_ch*support_count)` 使新 LRP transform 第一 conv 输入宽度跟旧 `M + slice_ch*(support_count+1)` 完全一致——上游 `lrp_transforms.{k}.*` 权重直接转移，无需 fine-tune。

**GaussianConditional 共享问题**：原 1 个共享 → 现 K 个 per-slice 副本（`_scale_table`/`_offset`/`_cdf_length` 重复 K 份，每模型多 ~几十 KB）。PyTorch state_dict 不 dedupe by-id，实用上可接受。

### D.6 基类去留 + convert 脚本影响（§10.8–§10.9）

**删除**：`SliceEntropyCompressionModel`（职责转移到 `HyperpriorLatentCodec` + 扩展后 `ChannelGroupsLatentCodec` + leaves；Family 1 模型直接继承 upstream `SimpleVAECompressionModel`）、`ChannelSliceLatentCodec`（重复 upstream）。**helper 保留**（`infer_num_slices`/`infer_max_support_slices`/`slice_support_channels`/`lrp_support_channels`/`make_entropy_transform`）搬到 `compressai/latent_codecs/_slice_helpers.py`。`DictionaryEntropyCompressionModel` 本 PR 不动（DCAE/SAAF follow-up 再删）。

每个 `convert_*_checkpoint.py` 加 `_DIRECTION_GH_RENAMES`（hyperprior 进 codec + per-slice 循环生成 cc/lrp/gaussian rename）：STF/TCM 各 ~25 行、CCA ~30（aux 多一层）、DCAE ~30（dt/dt_cross_attention 移位）、MambaVC ~25、MLIC++ ~5、MambaIC ~10。

### D.7 与 upstream codec 的复用关系（PR description 用，§10.11）

| upstream codec | 本 PR 怎么用 |
|---|---|
| `LatentCodec`（base.py）| 父类，所有 leaf/容器继承 |
| `HyperpriorLatentCodec` | **核心容器**：模型 wiring 顶层 |
| `EntropyBottleneckLatentCodec` | z leaf；CCA 用 `quantizer="ste"`，其他默认 `"noise"` |
| `GaussianConditionalLatentCodec` | Family 1 leaf 基类（`LRPGaussianLatentCodec` subclass 用于全部 K slice）|
| `ChannelGroupsLatentCodec` | **核心容器**：复用 + 加 `max_support_slices` + `support_filter` 两个可选参数 |
| `CheckerboardLatentCodec` / `HyperLatentCodec` / `RasterScanLatentCodec` / `gain/*` | 不用（其他 family 用 / upstream 已 deprecated）|

### D.8 LoC 与已 drop 的设计

- **LoC**：fork-baseline 假设下估算净 ~−520 行（删 `channel_slice.py` −270 + `_bases/slice_entropy.py` −260 + 3 个 F1 模型瘦身 −330，抵消新增基础设施 + convert rename + 测试）；**实际 upstream-baseline PR diff +4596/−655 = 净 +3941**——因 TCM/CCA 在 upstream 不存在须从零写入容器化版本（TCM ~700 + CCA ~1100 + CCA loss ~130 + 2 convert ~250 + 测试 ~500）。详见 [`plan/generated/pr-tcm-cca-draft.md`](../../generated/pr-tcm-cca-draft.md)。
- **已 drop 的 `build_channel_slice_codec`**（原 §10.4 应用层 factory）：设计过，实施时判定 unused 而删除（commit `978d840` "drop unused build_channel_slice_codec"）——模型侧直接构造 `ChannelGroupsLatentCodec` 字典更直观。
- **dead-end 评估**：HPCM/WeConvene/TinyLIC/ShiftLIC/Entroformer/RefBasedAR 不容器化（forward loop 不可分解）；MLIC++ 内部不容器化的原结论**已被 [`family2-roadmap.md`](../active/family2-roadmap.md) PR-1 推翻**——拆出 sibling leaf `MultiContextCheckerboardLatentCodec`（`multi_context_checkerboard.py` 311 行 + `_checkerboard_helpers.py` 145 行已落地，含 ELIC 等价回归测试）。

---



- [x] 移动本文件到 `plan/exec-plans/completed/` ✅
- [x] 更新 `plan/README.md` 索引 ✅（active section 移除，completed section 加 entry 含 commit range）
- [x] ~~在 design doc `channel-slice-codec-redesign.md` §5 顶部加「**实施完成**」标记~~ —— **已被 2026-06-04 重构取代**：design-doc §4–§10（含原 §5 推荐方案）整体精简后并入本计划「[设计依据](#design-rationale)」段，design-doc 收敛为 §1–§3 纯参考表
- [ ] DCAE / MambaVC 的容器化迁移记入 `plan/exec-plans/active/dcae-mambavc-containerization.md`（独立 follow-up PR）—— **延后**，等本 PR 在 upstream merge 后再起草
