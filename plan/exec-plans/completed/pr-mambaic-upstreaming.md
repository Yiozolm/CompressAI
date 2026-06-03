# pr-mambaic: MambaIC + SSM 基础设施上游迁入执行计划

**计划日期**：2026-05-10
**状态**：✅ 已完成（2026-06-03）—— 与 CMIC（原 PR-4）**合并为一个 PR**（`pr-mambaic-cmic`，PR #5，base `master`，commits `645976b..50b9a41`）。SSM 子包 + `infer_swatten_*` 助手 + `[ssm]` extra + MambaICLatentCodec（保留 dedicated）+ MambaIC 模型 + convert 脚本 + zoo + 测试。纯 PyTorch ref backend 下 `pytest tests/test_models.py tests/test_layers.py` 76 passed，ruff/import 审计干净。详见 [`pr-cmic-upstreaming.md`](pr-cmic-upstreaming.md)（合并 PR 的共同记录）。
**分支**：`pr-mambaic-cmic`（基于 `master`，前置 PR-1/2 + GLIC 均已合并）
**目标 PR**：Family 2 系列**第三个** PR（与第四个 CMIC 合并提交）
**前置依赖**：
- [`pr-tcm-cca`](../completed/codec-containerization-h-g-refactor.md) 已 merge（提供容器化基础设施 + `compressai/latent_codecs/_slice_helpers.py` + `compressai/layers/attn/swatten.py`）
- pr-dcae-saaf-auxt 状态**不阻塞**（MambaIC 不用 OLP/WLS/iWLS）

**设计文档**：[`plan/design-docs/channel-slice-codec-redesign.md`](../../design-docs/channel-slice-codec-redesign.md) §2.2 Family 2 MambaIC 列 / §3.4 dedicated codec 段
**路线图**：[`family2-roadmap.md`](family2-roadmap.md) PR-3

---

## ⭐ Scope 锁定

**做**：
- lift `compressai/layers/ssm/{__init__.py,ssm.py 250,ssm_ops.py 326,builders.py 190,inference.py 63}` 子包（~830 LoC）—— 包含：`SS2D` / `VSSBlock` / `selective_scan` / `selective_scan_ref`（纯 PyTorch fallback）/ `cross_scan` / `cross_merge` / `cross_selective_scan` / `get_selective_scan_backend` / `is_mamba_ssm_available` / `is_selective_scan_cuda_available` / `build_vss_backbone` / `build_vss_context_stage` / `infer_vss_block_kwargs` / `infer_vss_depths`
- 引入 **`[ssm]` extras**（严格可选）含 `mamba_ssm` + `selective_scan_cuda*` + `triton`，三档 fallback（CUDA kernel → mamba_ssm → 纯 PyTorch ref）
- lift `compressai/latent_codecs/mambaic.py`（dedicated `MambaICLatentCodec`，~383 LoC）
- lift `compressai/models/mambaic.py`（~450 LoC，**修复**：原 import `compressai.models._bases.*` 已被 pr-tcm-cca 删除，改为 `compressai.latent_codecs._slice_helpers.*`）
- lift `examples/convert_mambaic_checkpoint.py`（CLI wrapper）
- zoo 接线（`mambaic` factory，gated by `is_mamba_ssm_available() or is_selective_scan_cuda_available()` 或纯 PyTorch fallback always-on）
- tests + state_dict round-trip + 上游 ckpt smoke
- License：作者邮件授权已获，merge 前补 SPDX header + COPYING / AUTHORS 致谢条目

**不做**：
- 不引入 `VMamba` / `MambaVision` 作为依赖（lic-migration-roadmap §0.4 已决策：vendor 实现更干净，无干净 PyPI 包）
- 不为 `MambaICLatentCodec` 抽通用 codec primitive —— 它是 dedicated codec 类，与 MLIC++ 一样 §10.12 明确不重构
- 不动 CMIC（CMIC 在 PR-4，复用本 PR lift 的 SSM 子包）
- 不预留 MambaVC 钩子（MambaVC 是另一个 follow-up PR，本 PR scope 内不预测其需求）

---

## Phase 0-8（占位，详细内容待动手前展开）

### Phase 0：清理 working tree + 分支建立（30 分钟）

- [ ] 等 pr-tcm-cca merge 进 upstream/master
- [ ] 基于 `upstream/master` 创建 `pr-mambaic`
- [ ] 联系作者签署 license attribution（作者邮件授权已收到，记录到 PR description）

### Phase 1：lift `compressai/layers/ssm/` 子包（1.5 d）

- [ ] cherry-pick `compressai/layers/ssm/{ssm,ssm_ops,builders,inference}.py`（~830 LoC）
- [ ] 验证三档 fallback：`get_selective_scan_backend()` 在三种环境（CUDA + mamba_ssm 都装 / 仅 mamba_ssm / 都没装）下选择正确 backend
- [ ] 单元测试：
  - `selective_scan_ref` 纯 PyTorch 路径数值正确
  - `SS2D` / `VSSBlock` forward shape + state_dict round-trip
  - `build_vss_backbone` / `build_vss_context_stage` 工厂函数验证
  - `is_mamba_ssm_available` / `is_selective_scan_cuda_available` 报告正确
- [ ] 验证 `import compressai.layers.ssm` 不报错（无依赖时仅 ref backend 可用）

### Phase 2：加 `[ssm]` extras（0.5 d）

- [ ] `pyproject.toml` 加 `[project.optional-dependencies].ssm = ["mamba_ssm; platform_system == 'Linux'"]`（CUDA wheel 仅 Linux 有，加 platform marker）
- [ ] `selective_scan_cuda_oflex` / `_core` / `_cuda` / `triton` **不进 extras**（无 PyPI wheel 或要 build from source），仅 `is_*_available()` 检测，缺失走 fallback
- [ ] `uv lock` 后验证：`uv sync --all-extras --dev` 在 macOS aarch64 跳过 mamba_ssm 不报错（platform marker 生效）
- [ ] CI 配置同步（GitHub Actions Linux runner 加 `--extra ssm` 验证 CUDA path）

### Phase 3：lift `compressai/latent_codecs/mambaic.py`（1 d）

- [ ] cherry-pick `MambaICLatentCodec` 类（~383 LoC）
- [ ] `compressai/latent_codecs/__init__.py` export `MambaICLatentCodec`
- [ ] 单元测试：构造 + forward + compress / decompress round-trip + state_dict 路径自检

### Phase 4：lift `compressai/models/mambaic.py`（1 d）

- [ ] cherry-pick MambaIC 模型类（~450 LoC）
- [ ] **修复 import**：`from compressai.models._bases import infer_max_support_slices, infer_num_slices, lrp_support_channels, make_entropy_transform, slice_support_channels` → `from compressai.latent_codecs._slice_helpers import ...`（pr-tcm-cca 已搬过去）
- [ ] 沿用 fork `script` 的 `_LATENT_PREFIX_REWRITES` legacy state_dict 兼容
- [ ] `from_state_dict` 自动检测 fork `script` legacy layout 调 convert

### Phase 5：convert script + 上游 ckpt 验证（0.5 d）

- [ ] cherry-pick `examples/convert_mambaic_checkpoint.py`
- [ ] 上游 candidate ckpt round-trip：从作者仓拉 published ckpt（License 邮件确认时可一并要 ckpt 链接），strict-load + sinusoidal smoke PSNR；fresh-init 对照
- [ ] **测试矩阵**：在三档 backend 下 forward 结果数值一致（CUDA / mamba_ssm / ref）；纯 PyTorch ref 比 CUDA 慢但数值正确

### Phase 6：tests（0.5 d）

- [ ] `tests/test_models.py::TestMambaIC`：forward / state_dict round-trip / upstream conversion 三个测试，**用纯 PyTorch ref backend** 让 macOS / 无 CUDA CI 可跑
- [ ] `tests/test_layers.py::TestSSM`：SSM 子包独立测试（含三档 backend 切换）

### Phase 7：清理 + zoo 接线（0.5 d）

- [ ] `compressai/zoo/__init__.py` 加 `mambaic` 到 `image_models`（无 gate，因为有纯 PyTorch fallback；缺加速 backend 仅性能差）
- [ ] `compressai/zoo/image.py` `mambaic()` factory + `_LazyImport` proxy
- [ ] 验证 `import compressai` + `import compressai.zoo` 在缺 `mamba_ssm` 环境下 0 加载 `mamba_ssm`

### Phase 8：全量验证 + 提交 + push（0.5 d）

- [ ] `make static-analysis` 全过
- [ ] `pytest tests/ -q --deselect tests/test_eval_model_video.py --deselect tests/test_zoo.py` 全过（macOS 用纯 PyTorch ref backend）
- [ ] state_dict 路径自检
- [ ] License 文件检查：`compressai/{models,latent_codecs}/mambaic.py` 顶部 SPDX + InterDigital 标准注释 + 原作者署名 + `COPYING` / `AUTHORS` 致谢条目
- [ ] 按 logical 分组打 commit（layers/ssm / extras / codec / model / convert / zoo / tests，6-7 commits）
- [ ] 写 PR description draft → `plan/generated/pr-mambaic-draft.md`，**显式说明 License 处理**：引用作者授权邮件 + 致谢条目位置
- [ ] push origin/pr-mambaic

---

## 关键风险

| 风险 | 严重度 | 缓解 |
|---|---|---|
| `mamba_ssm` 在 macOS aarch64 / Windows 无 wheel | 中 | platform marker `; platform_system == 'Linux'` + 三档 fallback；纯 PyTorch ref 始终可用 |
| `selective_scan_cuda*` 是 vendored kernel，无 PyPI 包 | 中 | 不加进 extras；仅 `is_selective_scan_cuda_available()` 检测；缺失时 fallback 到 mamba_ssm 或 ref |
| 纯 PyTorch ref 跑 SS2D / VSSBlock 慢（10-100×）| 低 | 仅作 fallback 用，文档中说明性能差距；smoke test 用小 input 即可 |
| MambaIC import `compressai.models._bases.*` 已被删 | 低 | mechanical rewrite 到 `compressai.latent_codecs._slice_helpers.*` |
| **License attribution** 处理不当 | 中 | merge 前补 SPDX header + 原作者署名 + COPYING 致谢条目 + PR description 引用作者授权邮件；reviewer 可能要求额外格式 |
| reviewer 反对 `mamba_ssm` 作为 optional dep | 低 | 用 platform marker 限定 Linux；纯 PyTorch fallback 保证默认环境永远可用 |

---

## 总时间估算

| Phase | 工时 |
|---|---|
| Phase 0-8 | ~5 工作日 |

比 pr-glic 长 1 天的增量来自 SSM 三档 backend 测试 + `[ssm]` extras 设计 + License attribution 流程。

---

## 完成后

- 移动本文件到 `plan/exec-plans/completed/`
- 更新 `plan/README.md` 索引
- 在 design doc `channel-slice-codec-redesign.md` §1 表 / §3.4 表把 MambaIC 状态从「fork `script` 已迁入」更新为「pr-mambaic 已合入 upstream」
- 把 `compressai/layers/ssm/` 标记为 SSM 家族 cross-model 共享层（CMIC 复用，进 PR-4 scope；MambaVC 未来复用，进 lic-migration-roadmap Phase 11）
- 在 [`family2-roadmap.md`](family2-roadmap.md) §4 PR 总览表把 PR-3 行加 ✅ + commit range
