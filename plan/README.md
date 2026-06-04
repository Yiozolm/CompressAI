# Plan 文档索引

本目录是项目所有规划/设计/执行文档的统一入口。新增文档按用途归入对应子目录，并在本文件登记一行索引。

## 目录约定

| 子目录 | 用途 |
|---|---|
| `product-specs/` | 产品/路线图层面的需求与范围说明（"做什么、为什么"） |
| `design-docs/` | 架构与设计决策、跨模块评估（"怎么组织、为什么这么组织"） |
| `exec-plans/active/` | 当前在进行中的执行计划（任务级，可勾选进度） |
| `exec-plans/completed/` | 已落地的执行计划（保留作为实施记录） |
| `generated/` | 工具/脚本/Claude 生成的中间产物（PR draft、报告等） |
| `references/` | 复用模板、规范、第三方资料的本地副本 |

## 维护规则（来自 CLAUDE.md）

1. 完成 `exec-plans/active/` 内的计划后，必须同步更新对应文档状态。
2. 活跃执行计划完成后移入 `exec-plans/completed/`，并更新本索引及相关引用。
3. 新文档按用途分别放入 `product-specs/`、`design-docs/`、`exec-plans/`、`generated/` 或 `references/`。
4. 不要写出过大的文件；保持架构化、模块化设计。
5. Python 包管理统一用 `uv`。

---

## 当前文档

### product-specs/

- [lic-migration-roadmap.md](product-specs/lic-migration-roadmap.md) — `candidate/` 下 LIC 模型迁入 `compressai/` 的整体路线图与候选分组

### design-docs/

- [layers-abstraction-refactor.md](design-docs/layers-abstraction-refactor.md) — `compressai/layers/` 分层评估、按算子族重组的原则与执行记录
- [cca-cross-model-extension.md](design-docs/cca-cross-model-extension.md) — 把 CCA 辅助熵模型从 TCM 单点接入推广到所有 channel-slice LIC 模型的方向性备忘。**⚠️ 已过时**：描述的是 `script` 分支插件形态（`entropy_models/cca.py` + TCM `use_cca`）；`master` 上游迁入时已合并为**单宿主** `CCAModel`（TCM `use_cca` 已删）——以 [`cca-embedding-abstraction.md`](design-docs/cca-embedding-abstraction.md) 为准
- [cca-embedding-abstraction.md](design-docs/cca-embedding-abstraction.md) — **通用 CCA 嵌入抽象设计（基于 `master`）**：纠正上条过时备忘——`master` 上 CCA 是**单宿主** `CCAModel`（非跨模型插件），aux 分支是私有 `_CCAAuxEntropyModel`，经 `cca_training` opt-in flag + state_dict 自探测。**评估「单独 model + 继承覆盖」提案 → 不建议**（继承为 opt-in toggle 新增类层级，与 registry/zoo/`from_state_dict` 自配置冲突；显式 flag > 类层级间接）。唯一 borderline dedup = 主/aux 孪生栈（差异仅 `support_count`），倾向保留显式。承接 survey §8 从严原则
- [channel-slice-codec-redesign.md](design-docs/channel-slice-codec-redesign.md) — **codec 家族分类速查表**：跨 25 个 entropy head（9 个结构族 + baseline/wrapper 桶）的变异维度对比与结构分类，含容器化适用边界。2026-06-04 补充调研新增 Family 8（空间 raster-scan AR）/ Family 9（非高斯 VQ）。H+G 容器化重构的候选方向、推荐方案与详细 API/state_dict 设计已并入 [`codec-containerization-h-g-refactor.md`](exec-plans/completed/codec-containerization-h-g-refactor.md)「设计依据」段
- [model-abstraction-optimization-survey.md](design-docs/model-abstraction-optimization-survey.md) — **`compressai/models/` 模型架构抽象优化调研（基于 `master`）**：前两份 design-doc（layers/codec）未覆盖的"模型如何与 entropy 基类交互"层残留样板——`base.py` entropy isinstance 硬编码逼 vbr.py 重写 `update()`（P1，开闭气味非 bug）、`from_state_dict` 无基类脚手架 ~500 行（P2）、`SimpleVAECompressionModel` 采纳不一致（STF↔WACNN 不对称，P3）、`GaussianMixtureConditional` 死能力（P3）；含"已做对的对照组"与动作清单。**codec 装配工厂列 won't-do**——研究/教学库刻意保留逐模型显式装配以护网络结构可读性（显式性 > DRY）。**注意：调研主体是 master（上游形态），与 `script` fork 全量主干债务画像不同**
- [auxt-embedding-abstraction.md](design-docs/auxt-embedding-abstraction.md) — **通用 AuxT 嵌入抽象设计（基于 `master`）**：4 个 AuxT host（TCM 侧分支+通用 walker / SAAF integral 插值 / CMIC·GLIC 手展开嵌套）三套嵌入形态对比。结论——「通用抽象」= **契约（命名/语义/loss/容错约定）+ 共享 infra**，不是吞掉所有 walk 的 megawalker。**唯一真去重 gap = AuxT state_dict 管线**（wavelet-buffer 谓词 3 变体 + convert 重抄，P1，纯基础设施）；**手 walk 保留显式**（每模型结构图，won't-do 强合）。承接 survey §8 从严原则
- [mlic-family-reproduction.md](design-docs/mlic-family-reproduction.md) — MLIC v1 / MLIC+ / MLICv2 在 PR-1 落地的 `MultiContextCheckerboardLatentCodec` 抽象上的复现设计（v1/v1+ Phase 10-11 已落地；Phase 11.5 已把 MLIC++ 统一到 `_BaseMLIC` 模板并删除 `mlicpp.py`；Phase 12 已给 leaf 加 `selective_predictor` 可选 hook；Phase 13 已新增 MLICv2 layers 子包；Phase 14 已接入 MLICv2 model/factory/zoo；Phase 5 统一 convert script + MLIC++ 真实 ckpt smoke 已落地；SGA Layer A codec generalization 已完成并归档）；记录 v1/v1+/v2 合并进 PR-1 的 scope 与 family2-roadmap 接入；**2026-05-11 v3.1 修订**新增 Phase 11.5 + Phase 5 ckpt smoke 重排到 Phase 14 之后

### exec-plans/active/

- [dpict-ctc-integration.md](exec-plans/active/dpict-ctc-integration.md) — DPICT (CVPR 2022) + CTC (CVPR 2023) 两阶段迁入计划，引入 `trit_plane` ops 与 `Divide_*` 子包
- [family2-roadmap.md](exec-plans/active/family2-roadmap.md) — Family 2（channel-slice + intra-slice 空间上下文 2-pass）上游迁入路线图：4 个独立 PR + 依赖图 + shared layer lift 矩阵 + License 处理路径；v3 估算总计 ~9750-9950 LoC / 25 工作日（PR-1 MLIC 系列扩到四模型）；前置依赖 pr-tcm-cca + pr-dcae-saaf-auxt
- [pr-mlicpp-upstreaming.md](exec-plans/active/pr-mlicpp-upstreaming.md) — Family 2 PR-1：**MLIC 系列**上游迁入（MLIC + MLIC+ + MLIC++ + MLICv2 合并 PR）+ **`MultiContextCheckerboardLatentCodec` 抽象**（MLIC++ Track Phase 1-4 已落地；v1/v1+ Track Phase 10-11 已落地；**Phase 11.5 已 unify MLIC++ 到 `_BaseMLIC` 模板并删除 `mlicpp.py`**；Phase 12 leaf selective hook 已落地；Phase 13 MLICv2 layers 已落地；Phase 14 MLICv2 model/factory/zoo 已落地；Phase 5 统一 convert script + MLIC++ 真实 ckpt smoke 已落地）；与 pr-dcae-saaf-auxt 独立可并行；设计详见 [`mlic-family-reproduction.md`](design-docs/mlic-family-reproduction.md)
- [llic-reproduction.md](exec-plans/active/llic-reproduction.md) — LLIC (TMM 2024, arXiv 2304.09571v9) 复现计划：新增大感受野 self-conditioned transform 共享层，优先实现 `LLIC-ELIC`，再扩展 `LLIC-STF` / `LLIC-TCM`；覆盖训练配方、验证与 RD 复现口径

### exec-plans/completed/

- [invcompress-integration.md](exec-plans/completed/invcompress-integration.md) — InvCompress (ACMMM 2021) 上游迁入完成（2026-06-03，**PR #9**，merge `43d47eb`）：FrEIA-backed 可逆 INN transform（`GLOWCouplingBlock`/`Fixed1x1Conv`/`IRevNetDownsampling` 内联）+ 新 `[invcompress]`/FrEIA extra（CI `--all-extras` 真跑）+ convert-to-examples（block-diagonal 耦合融合 + 1×1 float64 求逆）+ 纯 `@register_model`（修 py3.8 `type[]` import 崩）；附 numpy<2 (py<3.12) cap 修 torch/numpy ABI 不兼容的 CI 教训
- [ftic-integration.md](exec-plans/completed/ftic-integration.md) — FTIC (ICLR 2024) 上游迁入完成（2026-06-03，**PR #8**，merge `36b48b7`）：新移位高斯 `GsnConditionalLocScaleShift` 入 `entropy_models/` + T-CA 通道自回归熵模型内联 + 复用 master `pad_to_window_multiple`（无新 layer 文件）+ convert-to-examples + deep-import-only；附 PR #8 首跑新 CI 照出的两个 pre-existing 问题修复（旧文件 ruff 格式 + `--no-extra ssm`）
- [weconvene-integration.md](exec-plans/completed/weconvene-integration.md) — WeConvene (ECCV 2024) 上游迁入完成（2026-06-03，**PR #7**，merge `0cbe7cf`）：小波域 transform 入 `layers/wave/weconv.py` + 新 `WeChARMLatentCodec`（wavelet-domain channel-AR）入 `latent_codecs/` + 复用 `[wavelet]` extra（无新依赖）+ convert-to-examples + deep-import-only
- [pr-glic-upstreaming.md](exec-plans/completed/pr-glic-upstreaming.md) — `pr-glic` Family 2 PR-2 完成（2026-06-03，commits `7b658dd..4c4cc3c`，PR #4）：新增 `compressai/layers/graph/` 子包（自写 GFA，无 torch_geometric）、把 GatedFFN/DepthwiseConv5x5/GatedTransformCNN lift 进 `layers/lic/blocks.py`（LayerNorm2d 复用 `timm.layers.LayerNorm2d`）、GLIC 容器化模型（ELIC 家族 channel-group + checkerboard）、convert 脚本入 examples、zoo 接线、TestGlic/TestGraph；64 passed，import 审计干净
- [pr-mambaic-upstreaming.md](exec-plans/completed/pr-mambaic-upstreaming.md) — Family 2 PR-3 完成（2026-06-03，与 PR-4 合并为 PR #5，commits `645976b..50b9a41`）：新增 `compressai/layers/ssm/` 子包（三档 selective-scan fallback）+ `attn/inference.py` 的 `infer_swatten_*` + `[ssm]` extra（`mamba-ssm==2.2.2; Linux`）+ `MambaICLatentCodec`（**保留 dedicated**，docstring 写明与 MLIC leaf 的区别）+ MambaIC 模型 + convert 脚本 + zoo/测试
- [pr-cmic-upstreaming.md](exec-plans/completed/pr-cmic-upstreaming.md) — Family 2 PR-4 完成（2026-06-03，与 PR-3 合并为 PR #5）：CMIC 模型（ELIC 家族，无新 codec 类）+ 内联 content-aware Mamba blocks，复用 GLIC gated blocks（`lic.blocks`）+ MambaIC 的 SSM `selective_scan` + AuxT OLP/WLS/iWLS；修了 `from_state_dict` 不推断 `stage_mlp_ratio` 的隐性 bug；`pytest tests/test_models.py tests/test_layers.py` 76 passed
- [codec-containerization-h-g-refactor.md](exec-plans/completed/codec-containerization-h-g-refactor.md) — `pr-tcm-cca` 容器化 H+G 重构 8 phase 执行计划完成（2026-05-09）：新增 latent_codecs 基础设施、删除 `ChannelSliceLatentCodec` + `_bases/slice_entropy.py`、迁移 STF/WACNN/TCM/CCA 到 ELIC pattern。本地 6 commits `c6d556a..f87c8c8` 已 push 到 origin/pr-tcm-cca，待 upstream review
- [dcae-saaf-auxt-containerization.md](exec-plans/completed/dcae-saaf-auxt-containerization.md) — `pr-dcae-saaf-auxt` 容器化迁入完成（2026-05-12，HEAD `819e10b`）：新增 dictionary cross-attention helper 与 DCAE / SAAF 容器化模型，落地 AuxT OLP/WLS/iWLS primitives、TCM `use_auxt=True` opt-in、`[wavelet]` optional dep、zoo 接线与真实 candidate checkpoint smoke；已 push 到 origin/pr-dcae-saaf-auxt
- [sga-codec-generalization.md](exec-plans/completed/sga-codec-generalization.md) — SGA quantizer 接入面扩展完成（2026-05-11，commit `9cdcb05`）：把 `quantizer="sga"` 推广到 `GaussianConditionalLatentCodec` / `CheckerboardLatentCodec`，并通过父类继承覆盖 `LRPGaussianLatentCodec`；Checkerboard 保持默认 STE、可选 SGA，覆盖 ELIC reconstruction/context `y_hat` 路径；上游无对应 ckpt 可做 ELIC checkpoint 回归，本地 targeted tests 25 passed
- [tbtc-integration.md](exec-plans/completed/tbtc-integration.md) — TBTC (ICLR 2022) 四模型迁入完成（2026-05-13，commit `89a1723`）：新增 Conv-Hyperprior / Conv-ChARM / SwinT-Hyperprior / SwinT-ChARM，复用现有 entropy 基础设施，新增 TBTC 专用 BHWC Swin transform 与 ChARM slice adapter；checkpoint 转换与真实 ckpt 数值对齐保留为后续项
- [shiftlic-tinylic-integration.md](exec-plans/completed/shiftlic-tinylic-integration.md) — TinyLIC + ShiftLIC small/middle/large 集成（共享 `MultistageCheckerboardLatentCodec`）；**已上游迁入 `Yiozolm/CompressAI` master，PR #6（2026-06-03，merge `2f453cb`）**——见该文档 Status 的上游迁入更新（deep-import / 无 natten extra / shift blocks 内联 / convert-to-examples）
- [mlicpp-latent-codec-refactor.md](exec-plans/completed/mlicpp-latent-codec-refactor.md) — 把 MLIC++ 的通道/上下文熵建模抽成 `MLICPlusPlusLatentCodec`
- [nvtc-compressai-port.md](exec-plans/completed/nvtc-compressai-port.md) — NVTC (CVPR 2023) 迁入 CompressAI：纯 `CompressionModel` forward/rate-estimation、VQ likelihood、state_dict 推断与 zoo 注册；上游 practical entropy coding 仍未实现

### generated/

- [pr-stf-wacnn-draft.md](generated/pr-stf-wacnn-draft.md) — `pr-stf-wacnn` 分支的 PR 描述草稿（STF + WACNN）
- [pr-mlicpp-draft.md](generated/pr-mlicpp-draft.md) — `pr-mlicpp` 分支的 PR 描述草稿（MLIC family + MultiContextCheckerboardLatentCodec）
- [pr-glic-draft.md](generated/pr-glic-draft.md) — `pr-glic` 分支的 PR 描述草稿（GLIC + graph 子包）
- [pr-mambaic-cmic-draft.md](generated/pr-mambaic-cmic-draft.md) — `pr-mambaic-cmic` 分支的 PR 描述草稿（MambaIC + CMIC + SSM 子包，PR #5）
- [pr-tinylic-shiftlic-draft.md](generated/pr-tinylic-shiftlic-draft.md) — `pr-tinylic-shiftlic` 分支的 PR 描述草稿（TinyLIC + ShiftLIC + 共享 `MultistageCheckerboardLatentCodec`，PR #6）

### references/

- [lic-model-integration-template.md](references/lic-model-integration-template.md) — 单个 LIC 模型迁入 CompressAI 的代码落点 / 注册 / zoo / 验证模板
