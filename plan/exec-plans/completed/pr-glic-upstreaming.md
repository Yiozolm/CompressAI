# pr-glic: GLIC 上游迁入执行计划

**计划日期**：2026-05-10
**状态**：✅ 已完成（2026-06-03）—— `pr-glic` 已 push 到 origin 并开 PR #4（base `master`）。6 个逻辑 commit `7b658dd..4c4cc3c`：graph 子包 + lic.blocks gated 块（LayerNorm2d 复用 timm）+ glic 模型 + convert 脚本 + zoo 接线 + 测试。`pytest tests/test_models.py tests/test_layers.py` 64 passed，ruff 全过，import 审计干净。
**分支**：`pr-glic`（基于本地 `master`，三个前置 PR 已合并入 master）
**目标 PR**：Family 2 系列**第二个** PR
**前置依赖**：
- [`pr-tcm-cca`](../completed/codec-containerization-h-g-refactor.md) 已 merge（提供容器化基础设施 + `compressai/latent_codecs/_slice_helpers.py`）
- [`pr-dcae-saaf-auxt`](../completed/dcae-saaf-auxt-containerization.md) 已 merge（提供 `OLP` / `WLS` / `iWLS` 在 `compressai/models/_helpers/auxt.py` + `DWT2D` / `IDWT2D` 在 `compressai/layers/wave/wavelet.py` + `[wavelet]` extras）

**设计文档**：[`plan/design-docs/channel-slice-codec-redesign.md`](../../design-docs/channel-slice-codec-redesign.md) §2.2 Family 2 GLIC 列 / §3.1 Family 2 段（GLIC 与 ELIC 同构使用 upstream codec primitives）
**路线图**：[`family2-roadmap.md`](family2-roadmap.md) PR-2

---

## ⭐ Scope 锁定

**做**：
- lift `compressai/layers/graph/{__init__.py,graph.py,graph_gfa.py,graph_ops.py}` 子包（~826 LoC，自写实现，**不引入** `torch_geometric` / `DGL`）—— 包含：`GAL` / `GDFN` / `GraphAttentionLayer` / `GraphAggregator` / `GraphDepthwiseFeedForward` / `IPGGrapher` / `GFA` (Graph Feature Aggregation) / `MGB` / `GraphLayerStack` / `FeatureReshape` / `FeatureRestore` + ops (`compute_sobel_gradients` / `cosine_similarity` / `gaussian_blur` / `global_sampling` / `local_sampling`)
- lift `compressai/models/glic.py`（~490 LoC，model class）
- lift `examples/convert_glic_checkpoint.py`（CLI wrapper）
- zoo 接线（`glic` factory，软可选 gate `is_pytorch_wavelets_available()`）
- tests + state_dict round-trip + 上游 ckpt smoke

**不做**：
- 不动 `compressai/models/sensetime.py::Elic2022Official`（GLIC 与 ELIC 同构，但是独立模型，不继承 ELIC 类——保持与 fork `script` 一致直接继承 `SimpleVAECompressionModel`）
- 不引入 `torch_geometric` / `DGL`（lic-migration-roadmap §0.4 已决策：GLIC 用的是 feature map 局部邻域 graph attention，不需要稀疏大图）
- 不重构 OLP / WLS / iWLS（已在 pr-dcae-saaf-auxt 落地）
- 不动 GLIC 的 `entropy` 路径——它直接用 upstream `ChannelGroupsLatentCodec` + `CheckerboardLatentCodec`，**无新 codec 类**（设计文档 §3.4 明确）

---

## Phase 0-7（占位，详细内容待动手前展开）

### Phase 0：清理 working tree + 分支建立（30 分钟）

- [ ] 等 pr-dcae-saaf-auxt merge 进 upstream/master，验证 `compressai/models/_helpers/auxt.py::OLP/WLS/iWLS` 与 `compressai/layers/wave/wavelet.py::DWT2D/IDWT2D` 可 import
- [ ] 基于 `upstream/master` 创建 `pr-glic`

### Phase 1：lift `compressai/layers/graph/` 子包（1.5 d）

- [ ] cherry-pick `compressai/layers/graph/{__init__.py,graph.py,graph_gfa.py,graph_ops.py}`（~826 LoC）
- [ ] **不**在 `compressai/layers/__init__.py` 顶层 re-export `from .graph import *` —— 跟 `attn/`、`wave/` 一致 deep-import only
- [ ] 单元测试：每个核心 class（`GAL` / `GFA` / `MGB` / `GraphLayerStack` / `FeatureReshape` / `FeatureRestore`）forward shape + state_dict round-trip + ops 函数（`cosine_similarity` / `local_sampling` 等）数值校验
- [ ] 验证 `import compressai.layers.graph` 不引入新依赖（纯 PyTorch，无 timm/wavelets）

### Phase 2：lift `compressai/models/glic.py`（1 d）

- [ ] cherry-pick GLIC 模型类（~490 LoC）
- [ ] 适配 import 路径：`OLP` / `WLS` / `iWLS` 走 `compressai.models._helpers.auxt`（pr-dcae-saaf-auxt 落点）；`DWT2D` / `IDWT2D` 走 `compressai.layers.wave.wavelet`
- [ ] 沿用 fork `script` 的 `convert_upstream_state_dict`：处理 `latent_codec.hyper.*` / `.OLP.` / `.dwt.w_*` / `.idwt.filters` / `pytorch_wavelets` buffer 兼容
- [ ] `from_state_dict` 自动检测上游 layout 调 convert
- [ ] 验证：`OLP` 跟 SAAF / TCM-with-AuxT 共享同一 class，state_dict 路径不冲突

### Phase 3：convert script + 上游 ckpt 验证（0.5 d）

- [ ] cherry-pick `examples/convert_glic_checkpoint.py`
- [ ] 上游 candidate ckpt round-trip：从作者仓 https://github.com/UnoC-727/GLIC 拉一个 published ckpt（如有），strict-load + sinusoidal smoke PSNR；fresh-init 对照证明权重确实参与计算

### Phase 4：tests（0.5 d）

- [ ] `tests/test_models.py::TestGlic`：forward / state_dict round-trip / upstream conversion 三个测试
- [ ] `tests/test_layers.py::TestGraph`：graph 子包独立测试（forward shapes + ops 数值）

### Phase 5：清理 + zoo 接线（0.5 d）

- [ ] `compressai/zoo/__init__.py` 加 `glic` 到 `image_models`（gated by `is_pytorch_wavelets_available()`，与 fork `script` 一致）
- [ ] `compressai/zoo/image.py` `glic()` factory + `_LazyImport` proxy
- [ ] 验证 `import compressai` + `import compressai.zoo` 在缺 `pytorch_wavelets` 环境下不报错（GLIC 缺席但其他模型可用）

### Phase 6：全量验证（0.5 d）

- [ ] `make static-analysis` 全过
- [ ] `pytest tests/ -q --deselect tests/test_eval_model_video.py --deselect tests/test_zoo.py` 全过
- [ ] state_dict 路径自检：构造 small GLIC，验证关键 key 存在（含 `OLP` / `WLS` / `iWLS` / `GFA` / channel_context / latent_codec.y.*）
- [ ] Import audit：`import compressai` 触发 0 timm + 0 pytorch_wavelets 加载
- [ ] `uv lock --check` 一致

### Phase 7：提交 + push（0.5 d）

- [ ] 按 logical 分组打 commit（layers/graph / models/glic / convert / zoo / tests，5-6 commits）
- [ ] 写 PR description draft → `plan/generated/pr-glic-draft.md`
- [ ] push origin/pr-glic

---

## 关键风险

| 风险 | 严重度 | 缓解 |
|---|---|---|
| OLP 路径冲突：SAAF / TCM-with-AuxT / GLIC 都引用同一 class | 低 | `_aggregate_aux_loss(model)` 走 `model.modules()` 自动聚合，state_dict 路径不冲突 |
| `pytorch_wavelets` 在 CI 是否安装 | 低 | `[wavelet]` extras 在 pr-dcae-saaf-auxt 已落地；CI 应已加 `--extra wavelet` |
| Graph layer 自写而非用 `torch_geometric` | 低 | lic-migration-roadmap §0.4 已决策；GLIC graph attention 是 feature map 局部邻域，不是稀疏大图 |
| GLIC 的 OLP loss 训练时怎么暴露 | 低 | 沿 SAAF pattern：`GLIC.aux_loss()` delegate to `_aggregate_aux_loss(self)`，user 在 training loop 加权 |

---

## 总时间估算

| Phase | 工时 |
|---|---|
| Phase 0-7 | ~4 工作日 |

比 pr-mlicpp 长 1 天的增量来自 graph 子包的 lift + 单元测试覆盖（~826 LoC 全新 layer 子包）。

---

## 完成后

- 移动本文件到 `plan/exec-plans/completed/`
- 更新 `plan/README.md` 索引
- 在 design doc `channel-slice-codec-redesign.md` §1 表 / §3.4 表把 GLIC 状态从「fork `script` 已迁入」更新为「pr-glic 已合入 upstream」
- 把 `compressai/layers/graph/` 标记为 GLIC + CMIC 共享层（CMIC 复用，进 PR-4 scope）
- 在 [`family2-roadmap.md`](family2-roadmap.md) §4 PR 总览表把 PR-2 行加 ✅ + commit range
