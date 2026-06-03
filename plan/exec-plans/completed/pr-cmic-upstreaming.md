# pr-cmic: CMIC 上游迁入执行计划

**计划日期**：2026-05-10
**状态**：✅ 已完成（2026-06-03）—— 与 MambaIC（原 PR-3）**合并为一个 PR**（`pr-mambaic-cmic`，PR #5，base `master`，commits `645976b..50b9a41`）。CMIC 模型（ELIC 家族 channel-group + checkerboard，无新 codec 类）+ 内联 content-aware Mamba blocks + convert 脚本 + zoo + 测试，复用 GLIC 的 gated blocks（`lic.blocks`）与 MambaIC 引入的 SSM `selective_scan`。修了 script 原版 `from_state_dict` 不推断 `stage_mlp_ratio` 的隐性 bug。`pytest tests/test_models.py tests/test_layers.py` 76 passed，ruff/import 审计干净。
**分支**：`pr-mambaic-cmic`（基于 `master`，前置 PR-1/2/3 + GLIC 均已合并）
**目标 PR**：Family 2 系列**第四个**（最后一个）PR（与第三个 MambaIC 合并提交）
**前置依赖**（**全部**必须先 merge 进 upstream）：
- [`pr-tcm-cca`](../completed/codec-containerization-h-g-refactor.md)（容器化基础设施）
- [`pr-dcae-saaf-auxt`](../completed/dcae-saaf-auxt-containerization.md)（OLP / WLS / iWLS / `[wavelet]` extras）
- [`pr-glic-upstreaming`](pr-glic-upstreaming.md)（`compressai/layers/graph/` 子包：GFA / GAL / GraphLayerStack 等）
- [`pr-mambaic-upstreaming`](pr-mambaic-upstreaming.md)（`compressai/layers/ssm/` 子包 + `[ssm]` extras + 三档 fallback）

**设计文档**：[`plan/design-docs/channel-slice-codec-redesign.md`](../../design-docs/channel-slice-codec-redesign.md) §2.2 Family 2 CMIC 列 / §3.1 Family 2 段（CMIC 与 ELIC/GLIC 同构使用 upstream codec primitives，**无新 codec 类**；novelty 在 content-aware Mamba context blocks）
**路线图**：[`family2-roadmap.md`](family2-roadmap.md) PR-4

---

## ⭐ Scope 锁定

**做**：
- lift `compressai/models/cmic.py`（~1057 LoC）—— 包含 CMIC 模型类 + private blocks (`CMICChannelContextBlock` / `CMICSpatialContextBlock` / `_ContentAwareMamba` SSM block)
- lift `examples/convert_cmic_checkpoint.py`（CLI wrapper）
- zoo 接线（`cmic` factory，gated by `is_pytorch_wavelets_available()` AND (`is_mamba_ssm_available()` OR `is_selective_scan_cuda_available()`)）
- tests + state_dict round-trip + 上游 ckpt smoke
- License：作者邮件授权已获，merge 前补 SPDX header + COPYING / AUTHORS 致谢条目

**不做**：
- **不**新增 `compressai/layers/` 文件 —— CMIC 私有的 `_ContentAwareMamba` / `CMICChannelContextBlock` / `CMICSpatialContextBlock` 全部内联在 `compressai/models/cmic.py`，因为这些是 CMIC 独占（不像 OLP/WLS/iWLS/GFA 跨模型复用）
- **不**新增 codec 类 —— CMIC 与 ELIC/GLIC 同构使用 upstream `ChannelGroupsLatentCodec` + `CheckerboardLatentCodec`
- **不**碰 PR-2 (graph) / PR-3 (ssm) 已 lift 的 shared layer —— CMIC 是消费者
- **不**重构 `_ContentAwareMamba` 上升为通用 layer —— 它是 CMIC 论文 (Chen CVPR 2024) 的 novelty，无第二个消费者

---

## Phase 0-7（占位，详细内容待动手前展开）

### Phase 0：清理 working tree + 分支建立（30 分钟）

- [ ] 等 pr-tcm-cca + pr-dcae-saaf-auxt + pr-glic + pr-mambaic 全部 merge 进 upstream/master
- [ ] 验证关键 import 可用：
  - `from compressai.models._helpers.auxt import OLP`（pr-dcae-saaf-auxt）
  - `from compressai.layers.wave.wavelet import WLS, iWLS`（pr-dcae-saaf-auxt 后扩展，或直接从 _helpers/auxt）
  - `from compressai.layers.graph import GFA`（pr-glic）
  - `from compressai.layers.ssm import selective_scan, build_vss_backbone`（pr-mambaic）
  - `from compressai.layers.attn.swin_attention import WindowAttention`（pr-tcm-cca / STF Phase 0.1 已 lift）
- [ ] 基于 `upstream/master` 创建 `pr-cmic`

### Phase 1：lift `compressai/models/cmic.py`（1.5 d）

- [ ] cherry-pick CMIC 模型类（~1057 LoC，最大 family-2 单文件）
- [ ] 适配 import 路径（前置 4 个 PR 落点的最终 path）
- [ ] **CMIC private blocks 全部内联**（不 lift 到 `compressai/layers/`）：`_ContentAwareMamba` / `CMICChannelContextBlock` / `CMICSpatialContextBlock` / `_apply_permutation` / `_inverse_permutation` / `_batched_bincount` 等 helper
- [ ] 沿用 fork `script` 的 `convert_upstream_state_dict` 处理上游 ckpt 命名差异
- [ ] `from_state_dict` 自动检测 fork `script` legacy layout 调 convert

### Phase 2：convert script + 上游 ckpt 验证（0.5 d）

- [ ] cherry-pick `examples/convert_cmic_checkpoint.py`
- [ ] 上游 candidate ckpt round-trip：从作者仓拉 published ckpt（License 邮件确认时一并要），strict-load + sinusoidal smoke PSNR
- [ ] **测试矩阵**：纯 PyTorch ref SSM backend 跑通（macOS / 无 CUDA 环境）+ wavelet 路径数值

### Phase 3：tests（0.5 d）

- [ ] `tests/test_models.py::TestCMIC`：forward / state_dict round-trip / upstream conversion 三个测试，用纯 PyTorch ref backend
- [ ] **不**新增 layer 测试（所有 layer 在前置 PR 已测过）

### Phase 4：清理 + zoo 接线（0.5 d）

- [ ] `compressai/zoo/__init__.py` 加 `cmic` 到 `image_models`（gated by `is_pytorch_wavelets_available()`，与 fork `script` 一致；SSM 缺失时走纯 PyTorch ref，不 gate）
- [ ] `compressai/zoo/image.py` `cmic()` factory + `_LazyImport` proxy
- [ ] 验证 `import compressai` + `import compressai.zoo` 在缺 `pytorch_wavelets` 或 `mamba_ssm` 环境下不报错

### Phase 5：全量验证（0.5 d）

- [ ] `make static-analysis` 全过
- [ ] `pytest tests/ -q --deselect tests/test_eval_model_video.py --deselect tests/test_zoo.py` 全过
- [ ] state_dict 路径自检：构造 small CMIC，验证关键 key 存在（OLP / WLS / iWLS / GFA / SSM / channel_context / latent_codec.y.*）
- [ ] Import audit：`import compressai` 触发 0 timm + 0 pytorch_wavelets + 0 mamba_ssm 加载
- [ ] `uv lock --check` 一致

### Phase 6：License attribution + 提交（0.5 d）

- [ ] License 文件检查：`compressai/models/cmic.py` 顶部 SPDX + InterDigital 标准注释 + 原作者署名 + `COPYING` / `AUTHORS` 致谢条目
- [ ] 按 logical 分组打 commit（model / convert / zoo / tests，4 commits）
- [ ] 写 PR description draft → `plan/generated/pr-cmic-draft.md`，**显式说明 License 处理**：引用作者授权邮件 + 致谢条目位置 + 列出复用的 4 个前置 PR shared infra

### Phase 7：push（0.5 d）

- [ ] push origin/pr-cmic
- [ ] 不要 push upstream，等 user 决定时机

---

## 关键风险

| 风险 | 严重度 | 缓解 |
|---|---|---|
| **前置 PR 链长**（4 个 PR 要先 merge）| 高 | 开发期可临时基于 stack 分支（`pr-cmic` based on `pr-mambaic` based on `pr-glic` based on `pr-dcae-saaf-auxt` based on `pr-tcm-cca`），等上游 merge 后做 cascade rebase |
| `_ContentAwareMamba` 在纯 PyTorch ref backend 下慢 | 低 | 测试用 tiny input；文档说明加速 backend 是性能优化、不是正确性必需 |
| 1057 LoC 单文件超 roadmap 「200-400 行」上限 | 中 | 已是最简化（其他都 lift 走了，剩下都是 CMIC 独占）；如 reviewer 强烈反对，可拆 `cmic_blocks.py` （内联不变，仅文件分割），但增加路径间接性，**默认不拆** |
| **License attribution** | 中 | 同 pr-mambaic：作者邮件授权 + SPDX + COPYING 致谢；reviewer (`@YodaEmbedding`) 在 PR 中确认格式 |
| 与 GLIC PR 时间窗冲突（reviewer 同时看 graph + ssm + cmic）| 低 | 推迟到 GLIC + MambaIC 都 merge 后再开 review，避免堆积 |

---

## 总时间估算

| Phase | 工时 |
|---|---|
| Phase 0-7 | ~3 工作日 |

比 pr-glic / pr-mambaic 短 1-2 天，因为：
- 不 lift 任何新 shared layer（全靠前置 PR）
- 不引入新 codec 类
- 不引入新 extras
- 仅 model 文件 + convert + zoo + test

---

## 完成后

- 移动本文件到 `plan/exec-plans/completed/`
- 更新 `plan/README.md` 索引
- 在 design doc `channel-slice-codec-redesign.md` §1 表 / §3.4 表把 CMIC 状态从「fork `script` 已迁入」更新为「pr-cmic 已合入 upstream」
- 在 [`family2-roadmap.md`](family2-roadmap.md) §4 PR 总览表把 PR-4 行加 ✅ + commit range
- **Family 2 全部完成**：移动 `family2-roadmap.md` 到 `completed/`；剩余 LIC 迁入对象按 lic-migration-roadmap：WeConvene (Phase 10) + FTIC (Phase 8) + InvCompress (Phase 9) + MambaVC (Phase 11)
