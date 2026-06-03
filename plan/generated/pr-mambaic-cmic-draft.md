# PR draft: feat(models): add MambaIC + CMIC with SSM (state-space) infrastructure

**Branch**: `pr-mambaic-cmic` → `master`
**Family 2 PR-3 + PR-4 (combined)**（见 `plan/exec-plans/active/{pr-mambaic,pr-cmic}-upstreaming.md` / `family2-roadmap.md`）

## Summary

把 **MambaIC**（Zeng et al., CVPR 2025，https://github.com/AlbertZhangHIT/MambaIC）和 **CMIC**（Chen et al., ICLR 2026, OpenReview WwDNiisZQm）从 fork `script` 分支迁入 `compressai/`，叠在已合并的 TCM/CCA + DCAE/SAAF/AuxT + MLIC + GLIC 四个 Family-2 PR 之上。两者都是 channel-slice + slice 内 checkerboard 的 2-pass 熵模型，共同引入一个新的 SSM（visual state-space / Mamba）共享子包。

原计划的 PR-3（MambaIC）与 PR-4（CMIC）合并为**一个** PR：CMIC 复用 MambaIC 引入的 `compressai.layers.ssm.selective_scan`，放在一起可一次完成 SSM 基础设施 + 两个消费者。

## Changes

**新共享层**
- `feat(layers): add SSM (visual state-space) subpackage` — `compressai/layers/ssm/{__init__,ssm,ssm_ops,builders,inference}.py`（~860 LoC）。`SS2D` / `VSSBlock` / `selective_scan` 三档 backend（CUDA kernel → `mamba_ssm` → 纯 PyTorch `selective_scan_ref` fallback）+ VSS backbone factory + state-dict 推断助手。deep-import-only（不进 `compressai.layers`），`import compressai` 不触发 `mamba_ssm`。MambaIC + CMIC 共享，未来 MambaVC 复用。
- `feat(layers): add SWAtten state-dict introspection helpers` — lift `infer_swatten_{window_size,head_dim,attention_dim}` 进 `compressai/layers/attn/inference.py` 并从 attn 包导出（MambaIC 首个消费者）。

**新 codec**
- `feat(latent_codecs): add MambaICLatentCodec` — MambaIC 专属 channel-slice + checkerboard codec。**保留 dedicated**（不复用 MLIC 的 `MultiContextCheckerboardLatentCodec`）：MambaIC 的 mean/scale 走两条独立 SWAtten 链路、slice 循环跨越 channel-group 与 per-pass 两层，与该 leaf 的单 head 契约冲突（详见类 docstring + design doc §10.12）。CMIC 与 ELIC/GLIC 同构，直接用 upstream `ChannelGroupsLatentCodec` + `CheckerboardLatentCodec`，无新 codec 类。

**新模型**
- `feat(models): add MambaIC with VSS backbone and dedicated codec` — `compressai/models/mambaic.py`。import 从已删除的 `models._bases` 迁到 `latent_codecs._slice_helpers`；`from_state_dict` 用 MambaIC 自有的 `_infer_num_slices` / `_infer_max_support_slices`（从 support heads + SWAtten in_conv 宽度推断，替代只认 ELIC `channel_context` 的通用助手——后者对 MambaIC 永远返回 0）。
- `feat(models): add CMIC with content-aware Mamba blocks` — `compressai/models/cmic.py`。ELIC 家族 channel-group + checkerboard codec；复用 gated transform blocks（`lic.blocks`）、AuxT OLP/WLS/iWLS（`_helpers.auxt`）、SSM `selective_scan`。content-aware Mamba / window-attention / spatial-context blocks 全部内联（CMIC 独占）。`aux_loss()` 聚合 OLP 正交惩罚，`ortho_loss()` 保留为别名。`from_state_dict` 新增 `stage_mlp_ratio` 推断（修复 script 原版只支持默认 3.0 的隐性 bug）。

**extras / examples / zoo / tests**
- `build: add optional [ssm] extra (mamba-ssm, Linux only)` — pin `mamba-ssm==2.2.2; platform_system == 'Linux'`，`[tool.uv]` 内联 build metadata 让非 Linux 的 universal lock 不构建 sdist。lock 只增 mamba-ssm 一项，无连锁 churn。
- `feat(examples): add MambaIC and CMIC checkpoint converters` — `convert_upstream_{mambaic,cmic}_state_dict` 自由函数 + 薄 CLI；convert 逻辑移出模型（对齐 stf/tcm/saaf/dcae/glic 先例），`from_state_dict` 只做 shape 推断 + load。
- `feat(zoo): wire mambaic/cmic entries; add MambaIC/CMIC/SSM tests` — `_LazyImport` + `mambaic()` / `cmic()` factory（pretrained 抛 RuntimeError）。`TestMambaIC` / `TestCMIC`（forward、state_dict round-trip、上游转换、aux_loss）+ `TestSSM`（ref backend 数值对齐、SS2D/VSSBlock 形状、backbone factory、state-dict 推断）。

## 设计要点

- **MambaIC 保留 dedicated codec**：经源码核对，mean/scale 分离 head + 跨层 slice 装配与 MLIC leaf 的单 head 契约冲突，符合 roadmap 预注册的 fallback 判定。
- **convert-to-examples**：两模型的上游 ckpt 转换逻辑都移到 `examples/`，模型 `from_state_dict` 瘦身为纯 shape 推断 + `load_state_dict`。
- **三档 SSM fallback**：无 CUDA kernel / `mamba_ssm` 时自动用纯 PyTorch `selective_scan_ref`，CI / macOS 可跑（本 PR 全部测试即在纯 PyTorch backend 下通过）。

## 依赖 / License

- 新增 1 个 optional extra `[ssm]`（严格可选，platform-marker 限 Linux），不引入新硬依赖。
- CMIC 复用 pr-dcae-saaf-auxt 的 `[wavelet]` extra（构造时经 WLS/iWLS lazy import 浮现）。
- License：MambaIC / CMIC 均经**原作者邮件授权**集成；文件头为「上游 adapts + InterDigital BSD-3-Clause-Clear」双声明（仿 glic.py 先例；repo 无 COPYING/AUTHORS/SPDX 文件，沿用现有 docstring/header 署名惯例）。

## 验证

- `import compressai` / `import compressai.zoo` 不触发 timm / pytorch_wavelets / mamba_ssm 加载（deep-import-only + lazy 生效）。
- `pytest tests/test_models.py tests/test_layers.py` → 76 passed；全量 `pytest tests/`（deselect video/zoo）→ 342 passed, 4 skipped, 1 failed（唯一 failure 是 `test_train_example_ddp`，DDP socket 超时的环境问题，与本 PR 无关）。
- `ruff format/check`（含 import 排序 + lint）在本 PR 新增/改动文件上全过。
- `uv lock --check` 一致。
- MambaIC / CMIC forward smoke（64×64 tiny config）、from_state_dict round-trip allclose、上游 layout 转换 round-trip 均通过。

## Notes

- 原 PR-3 / PR-4 合并为本 PR。`compressai/layers/ssm/` 将作为 SSM 家族 cross-model 共享层（未来 MambaVC 复用）。
- SSM / attn-inference 子模块无 License header（与现有 attn/wave/graph 子包惯例一致）。
