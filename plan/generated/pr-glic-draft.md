# PR draft: feat(models): add GLIC with graph-feature-aggregation codec

**Branch**: `pr-glic` → `master`
**Family 2 PR-2**（见 `plan/exec-plans/active/pr-glic-upstreaming.md` / `family2-roadmap.md`）

## Summary

把 GLIC（Chen et al., CVPR 2026，上游 https://github.com/UnoC-727/GLIC，MIT）从 fork `script` 分支迁入 `compressai/`，叠在已合并的 TCM/CCA + DCAE/SAAF/AuxT + MLIC 三个 Family-2 PR 之上。GLIC 在小波/CNN 辅助变换之上引入内容自适应的图特征聚合（GFA）分支，熵模型沿用 ELIC 家族的 channel-group + checkerboard hyperprior 容器化 latent codec。

## Changes

**新模型**
- `feat(models): add GLIC with containerized codec` — `compressai/models/glic.py`，继承 `SimpleVAECompressionModel`，codec 接线 `HyperpriorLatentCodec → ChannelGroupsLatentCodec → 每组 CheckerboardLatentCodec → GaussianConditionalLatentCodec`，与 `Elic2022Official` 同构。

**新共享层**
- `feat(layers): add graph feature aggregation subpackage` — `compressai/layers/graph/{graph,graph_gfa,graph_ops}.py`（~869 LoC，自写，不引入 torch_geometric/DGL）。自包含、deep-import-only（与 attn/wave 一致）。后续 CMIC（PR-4）复用。
- `feat(layers): add gated depthwise transform blocks to lic.blocks` — 把 `GatedFFN / DepthwiseConv5x5 / GatedTransformCNN` lift 进 `compressai/layers/lic/blocks.py`。`LayerNorm2d` 复用 `timm.layers.LayerNorm2d`（数值等价、state_dict key 一致；MLICv2 已在用），不再自写。`OLP` 继续用 `models/_helpers/auxt.py`。

**zoo / examples / tests**
- `chore(zoo): wire glic zoo entry with lazy import` — `_LazyImport` + `glic()` factory，无 wavelet gate（依赖只在构造时经 WLS/iWLS lazy import 浮现）。
- `feat(examples): add GLIC checkpoint converter` — `examples/convert_glic_checkpoint.py`，承载 `convert_upstream_glic_state_dict`（上游 → compressai layout 重键），模型 `from_state_dict` 只做 shape 推断 + load。
- `test(glic): add GLIC model and graph subpackage tests` — `tests/test_models.py::TestGlic` + `tests/test_layers.py::TestGraph`。

## 设计要点

- **ELIC-style inline**：GLIC 的 codec 主体本就是 inline 容器化写法；本 PR 对齐 stf/tcm/cca 的「inline ELIC-style codec wiring; move convert to examples」先例——把 convert 逻辑移到 `examples/`，`from_state_dict` 瘦身。
- **import 路径修正**：GLIC 写于 tcm-cca/dcae 重构前，`WLS/iWLS/OLP` 改从 `models/_helpers.auxt` 导，gated blocks 改从 `layers.lic.blocks` 导。
- **aux loss 对齐**：`aux_loss()` 复用 `auxt.aux_loss`（0-d tensor），`ortho_loss()` 保留为向后兼容别名。
- **wavelet buffer 容差**：GLIC 把 AuxT 嵌在 `g_a.`/`g_s.` 下（key 如 `g_a.AuxT_enc.0.dwt.transform.*`），不被 auxt 的前缀锚定 helper 匹配，故保留 glic.py 自带的子串式 `_is_pytorch_wavelets_buffer_key`。

## 依赖 / License

- 无新硬依赖、不改 `pyproject.toml`（`[wavelet]`/`[attn]` extras 已存在）。
- License 无 blocker：文件头为「MIT 上游 + InterDigital BSD-3-Clause-Clear」双声明。

## 验证

- `import compressai` / `import compressai.zoo` 不触发 timm/pytorch_wavelets/graph 加载（deep-import-only + lazy 生效，`-X importtime` 审计为空）。
- `pytest tests/test_models.py tests/test_layers.py -q --deselect tests/test_eval_model_video.py --deselect tests/test_zoo.py` → 64 passed。
- `ruff check` / `ruff format --check` 全过。
- GLIC forward smoke（128×128）、from_state_dict round-trip allclose、上游 layout 转换 round-trip 均通过。

## Notes

- GFA 接 4D BCHW，最小空间尺寸 16×16；GLIC 端到端测试最小输入 128×128（g3 在 1/8 尺度）。
- CMIC（PR-4）将复用本 PR 的 `compressai/layers/graph/` 子包。
