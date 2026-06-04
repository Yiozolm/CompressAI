# Family 2 上游迁入路线图（channel-slice + intra-slice 空间上下文 2-pass）

**计划日期**：2026-05-10（v2，2026-05-11 把 MLIC + MLIC+ + MLICv2 合并进 PR-1 同一 PR）
**状态**：active（PR-1 MLIC 系列：MLIC++ Phase 1-4 + Phase 11.5 unify 已落地；v1/v1+ Phase 10-11 已落地；v2 Phase 12 leaf hook + Phase 13 layers + Phase 14 model/factory/zoo 已落地，Phase 5/15/16 待动手；**PR-2 GLIC ✅ PR #4；PR-3 MambaIC + PR-4 CMIC ✅ 合并为 PR #5（2026-06-03）**；仅剩 PR-1 收尾）
**设计文档**：[`plan/design-docs/channel-slice-codec-redesign.md`](../../design-docs/channel-slice-codec-redesign.md) — §2.2 Family 2 表 + §3.1 Family 2 段 + §4 容器化适用边界；H+G wiring sketch / state_dict 设计见 [`codec-containerization-h-g-refactor.md`](../completed/codec-containerization-h-g-refactor.md#design-rationale)「设计依据」段（Family 1 已实施完毕，Family 2 复用相同思路但**不重新容器化** dedicated codec）

> **本文档定位**：Family 2 整体迁入的 master roadmap + scope 拆分 + 依赖图 + shared layer lift 矩阵 + PR 顺序。每个 PR 的 phase 级细节在各自 exec plan 文件里展开（链接见 §6）。

---

## 1. Family 2 scope

| 模型 | 文件（fork `script`）| Codec | 状态 | 本路线图覆盖 |
|---|---|---|---|---|
| ELIC | `compressai/models/sensetime.py` | upstream `ChannelGroupsLatentCodec` + `CheckerboardLatentCodec` | ✅ 已是 upstream 终态 | **不动** |
| GLIC | `compressai/models/glic.py` (490 LoC) | 同 ELIC（无新 codec 类） | ✅ 已迁入 compressai（PR #4） | **PR-2** |
| CMIC | `compressai/models/cmic.py` (1057 LoC) | 同 ELIC（无新 codec 类） | ✅ 已迁入 compressai（PR #5，与 MambaIC 合并） | **PR-4** |
| MLIC++ | `compressai/models/mlic.py`（与 MLIC / MLIC+ 共 `_BaseMLIC`）+ `layers/lic/mlic/` (638+ LoC) | **v2 pivot + Phase 11.5 unify**：拆为 upstream `ChannelGroupsLatentCodec` + 新 sibling leaf `MultiContextCheckerboardLatentCodec`，并把 thin model 合入 `mlic.py` | PR-1 Phase 1-4 + Phase 11.5 已落地（monolith codec 已被 sibling leaf + application factory + shared thin model 替代） | **PR-1** |
| MambaIC | `compressai/models/mambaic.py` (450 LoC) + `latent_codecs/mambaic.py` (383 LoC) | 专属 `MambaICLatentCodec`（**评估结论：保留 dedicated** —— mean/scale 分离 head + slice 循环跨两层，与 `MultiContextCheckerboardLatentCodec` 单 head 契约冲突） | ✅ 已迁入 compressai（PR #5，与 CMIC 合并） | **PR-3** |

**设计文档原结论**（§10.12）：「Family 2 内部 dedicated codec 类（MLIC++/MambaIC）保留，强行合并代价大于收益」。**v2 修订**（2026-05-10）：MLIC++ 在 PR-1 中部分容器化 —— **外层** sliced 循环复用 upstream `ChannelGroupsLatentCodec`，**内层** 双 head + 多 context + 双 LRP 的 anchor/nonanchor pattern 抽成新 sibling leaf `MultiContextCheckerboardLatentCodec`（与 upstream `CheckerboardLatentCodec` 同位）。这条 leaf 设计上预留 MambaIC 复用空间（spatial_context_nonanchor 槽位可接 VSS Mamba block），但 PR-3 启动时再独立评估。MLIC++ application-layer building blocks（`LocalContext` / `LinearGlobalInter/IntraContext` / `ChannelContext` 等）保留在 `compressai/layers/lic/mlic/`，作为应用层 helper 注入新 leaf 的参数。

---

## 2. 前置依赖链

```
upstream/master
    ↓
pr-tcm-cca ✅（已 push origin，待 upstream review）—— 提供：HyperpriorLatentCodec / ChannelGroupsLatentCodec 容器化基础设施 +
    LRPGaussianLatentCodec + DualHyperSynthesis + _slice_helpers + _helpers/{channel_slice,channel_context}
    ↓
pr-dcae-saaf-auxt ✅（已 push origin，待 upstream review）—— 提供：OLP / WLS / iWLS / DWT2D / IDWT2D（Phase 3 lift 到
    compressai/{models/_helpers/auxt.py, layers/wave/wavelet.py}），[wavelet] extras（pytorch_wavelets 软可选）
    ↓
┌── PR-1: pr-mlicpp     —— 不依赖 pr-dcae-saaf-auxt（MLIC++ 不用 OLP/wavelet/SSM），可与 pr-dcae-saaf-auxt 并行 review
│
└── pr-dcae-saaf-auxt merged into upstream
        ↓
        ├── PR-2: pr-glic       —— 复用 OLP / WLS / iWLS（来自 pr-dcae-saaf-auxt），新增 graph 子包
        │
        ├── PR-3: pr-mambaic    —— 不复用 pr-dcae-saaf-auxt（MambaIC 不用 OLP/wavelet），首次 lift SSM kernel + [ssm] extras
        │
        └── PR-2 + PR-3 都 merged
                ↓
                PR-4: pr-cmic   —— 复用 GLIC 的 graph 子包 + MambaIC 的 SSM kernel + WLS/iWLS（pr-dcae-saaf-auxt）
```

PR-1 与 pr-dcae-saaf-auxt **彼此独立**，理论可以同时 review；PR-2 / PR-3 **彼此独立**（GLIC 不需要 SSM，MambaIC 不需要 graph/wavelet），可在 pr-dcae-saaf-auxt merge 后并行；PR-4 是 leaf node。

---

## 3. Shared layer lift 矩阵

Family 2 需要 lift 到 upstream 的 shared layer 共 **3 个新子包 + 1 个新 extras**，按 PR 归属：

| Shared 资源 | fork `script` 落点 | 上游 lift PR | 估算 LoC | License | 谁消费 |
|---|---|---|---|---|---|
| `OLP` / `WLS` / `iWLS` | `compressai/models/_helpers/auxt.py` | **pr-dcae-saaf-auxt** ✅ 已落地 | 已 lift | — | GLIC, CMIC（外加 SAAF / TCM-with-AuxT 已用）|
| `DWT2D` / `IDWT2D` | `compressai/layers/wave/wavelet.py` | **pr-dcae-saaf-auxt** ✅ 已落地 | 已 lift | — | GLIC, CMIC（未来 WeConvene）|
| `[wavelet]` extras (`pytorch_wavelets`) | `pyproject.toml` | **pr-dcae-saaf-auxt** ✅ 已落地 | 已落地 | BSD | GLIC, CMIC |
| **`MultiContextCheckerboardLatentCodec`**（sibling of upstream `CheckerboardLatentCodec`）| 新增 `compressai/latent_codecs/multi_context_checkerboard.py` | **PR-1 (pr-mlicpp)** ✅ Phase 1 + Phase 12 落地 | 311+（实际 vs ~280 估；Phase 12 追加 selective hook）| BSD（自写）| MLIC / MLIC+ / MLIC++ / MLICv2 已用；MambaIC 候选（PR-3 评估）|
| **`_checkerboard_helpers.py`**（sibling 共享 single source of truth：`embed` / `unembed` / `mask_all*` / `merge` / `write_step` / `step_parity`）| 新增 `compressai/latent_codecs/_checkerboard_helpers.py` | **PR-1 (pr-mlicpp)** ✅ Phase 1 落地 | 145 | BSD（自写）| `CheckerboardLatentCodec` + `MultiContextCheckerboardLatentCodec` 共享 |
| `compressai/layers/lic/mlic/{__init__,context.py,transforms.py,utils.py}` | 同 | **PR-1 (pr-mlicpp)** ✅ Phase 2 + Phase 10 落地 | 976+（Phase 10 已追加 v1/v1+ context blocks）| Apache 2.0 | MLIC / MLIC+ / MLIC++ 共享 application-layer（不是 codec primitive）|
| `compressai/models/_helpers/multi_context_slice.py`（`build_mlic_slice_codec(variant=...)` factory + `_MlicppPriorAggregation` helper）| 新增 | **PR-1 (pr-mlicpp)** ✅ Phase 3 + Phase 11 + Phase 11.5 + Phase 14 落地 | 463+（实际；含 side layout wrappers，Phase 11.5 删除独立 `build_mlicpp_slice_codec`，Phase 14 扩到 `variant="mlicv2"`）| BSD（自写）| MLIC / MLIC+ / MLIC++ / MLICv2 application-layer factory |
| `compressai/layers/graph/{graph,graph_gfa,graph_ops}.py` | 同 | **PR-2 (pr-glic)** | ~826 | MIT | GLIC, CMIC |
| `compressai/layers/ssm/{ssm.py 250, ssm_ops.py 326, builders.py 190, inference.py 63}` + `[ssm]` extras（`mamba_ssm` 严格可选 + `selective_scan_cuda*` / `triton` 严格可选 + 纯 PyTorch fallback）| 同 | **PR-3 (pr-mambaic)** | ~830 + pyproject diff | 模块 BSD（重写）；外部 dep 各自上游 license | MambaIC, CMIC（未来 MambaVC）|
| `compressai/latent_codecs/mambaic.py` | 同 | **PR-3 (pr-mambaic)** | ~383（**或更少**：若 PR-3 评估决定改写为 `MultiContextCheckerboardLatentCodec` 应用层装配，可压缩到 ~50 LoC application-layer factory）| 同 MambaIC（已获作者邮件授权）| MambaIC 独占 |

**Family 2 需要新增 1 个新 optional extras**（`[ssm]`），不引入新硬依赖。pr-dcae-saaf-auxt 已加的 `[wavelet]` extras GLIC/CMIC 直接复用，无需重复工作。

**v2 抽象层影响**：PR-1 引入的 `MultiContextCheckerboardLatentCodec` 是 **`CheckerboardLatentCodec` 的 sibling 广义化**（separate anchor/nonanchor heads + 可插拔空间/通道 context + 可选 per-pass LRP + 可选 `selective_predictor`）。Phase 14 后已有 MLIC / MLIC+ / MLIC++ / MLICv2 四个消费者，MLICv2 通过 `selective_predictor` 注入 `GSCModule`，并在 application-layer factory 中接入 HGCP / Context Reweighting / 2D RoPE。MambaIC 在 PR-3 启动时**重新评估**是否改写为本 leaf 的应用层装配（spatial_context_nonanchor 槽位接 VSS Mamba block）—— 如可，PR-3 净 LoC 进一步压缩；如不可（MambaIC 第一遍 SWAtten 分离 mean/scale 与 sibling leaf 第二 pass 一体化设计冲突），则保留 dedicated `MambaICLatentCodec`。

---

## 4. PR 拆分总览

| PR | 模型 | 新 codec 类 | 新 shared 子包 | 新 extras | 估算 LoC（净）| 估算工时 | License |
|---|---|---|---|---|---|---|---|
| **PR-1** | **MLIC 系列**（MLIC + MLIC+ + MLIC++ + MLICv2，详见 [`mlic-family-reproduction.md`](../../design-docs/mlic-family-reproduction.md)）| **`MultiContextCheckerboardLatentCodec`**（sibling of `CheckerboardLatentCodec`，可复用；MLICv2 启用可选 `selective_predictor` hook 处理 GSC）+ **`_checkerboard_helpers.py`** | `compressai/layers/lic/mlic/`（v1/v1+/++ 共享 application-layer）+ `compressai/layers/lic/mlicv2/`（v2 专用：STM transforms / HGCP / Context Reweighting / 2D RoPE / GSC，Phase 13 已落地）+ `models/_helpers/multi_context_slice.py`（`build_mlic_slice_codec(variant=...)`，Phase 14 已扩到 `mlicv2`）+ thin models 统一放 `compressai/models/mlic.py`（Phase 14 后含 MLIC / MLIC+ / MLIC++ / MLICv2）| 无 | +5300~+5500（Phase 1-14 已显著高于初估；ckpt smoke/收尾仍待补）| 13 d（MLIC++ Phase 1-4 + 11.5 已落地，ckpt smoke 待补；v1/v1+ ~2.5 d ✅ 已落地；Phase 12-14 ✅；Phase 5/15/16 待动手）| Apache 2.0 ✅ |
| **PR-2** ✅ | GLIC | 无（用 upstream `ChannelGroupsLatentCodec` + `CheckerboardLatentCodec`）| `compressai/layers/graph/` | 无（复用 pr-dcae-saaf-auxt 的 `[wavelet]`）| +1450 | 4 d | MIT ✅ | **已完成 2026-06-03，PR #4，commits `7b658dd..4c4cc3c`**（LayerNorm2d 复用 timm；gated blocks 进 lic.blocks）|
| **PR-3** ✅ | MambaIC | `MambaICLatentCodec`（**评估结论：保留 dedicated** —— mean/scale 分离 head + slice 循环跨 channel-group/per-pass 两层，与 `MultiContextCheckerboardLatentCodec` 单 head 契约冲突，符合 §10.12 + roadmap 预注册 fallback）| `compressai/layers/ssm/` + `attn/inference.py`（`infer_swatten_*`）| `[ssm]`（严格可选，`mamba-ssm==2.2.2; Linux`）| +2200 | — | **作者邮件授权** ✅ | **已完成 2026-06-03，与 PR-4 合并为 PR #5，commits `645976b..50b9a41`** |
| **PR-4** ✅ | CMIC | 无（与 GLIC 同构）| 无（复用 PR-2 graph 的 gated blocks + PR-3 ssm）| 无（复用 `[wavelet]` + PR-3 `[ssm]`）| +1240 | — | **作者邮件授权** ✅ | **已完成 2026-06-03，与 PR-3 合并为 PR #5**（CMIC 不 import graph 子包，只复用 gated blocks；修了 `stage_mlp_ratio` 推断 bug）|
| **总计** | 4 PR（PR-3/4 合并提交，实际 3 个 PR）/ 7 模型（MLIC + MLIC+ + MLIC++ + MLICv2 + GLIC + MambaIC + CMIC）| 1 codec + 1 selective hook + 1 dedicated（MambaIC）| 5 子包 | 1 extras | +9750~+9950 | 25 d | — |

工时估算假设 pr-tcm-cca + pr-dcae-saaf-auxt 都已 merge（容器化基础设施 + auxt helper + wavelet extras 全部到位）。如要 hold 住等 upstream merge 才动手，整段排期会被拉长。

**v1→v2 工时净增 +2.5d（仅 PR-1）的原因**：抽象设计 + 新 leaf 实现 + 单元测试 + ELIC 等价回归 + 应用层 factory + thin model 重写。回报：(a) `MultiContextCheckerboardLatentCodec` 给 MambaIC 留复用空间，PR-3 可能 -1d；(b) MLIC++ 不再是 monolith codec，pedagogical clarity 与 ELIC pattern 收敛；(c) 删了 fork `script` 上 ~474 LoC 的 monolith codec，净 LoC 实际比 v1 更小。

---

## 5. 关键风险

| 风险 | 影响 PR | 严重度 | 缓解 |
|---|---|---|---|
| pr-tcm-cca / pr-dcae-saaf-auxt 在 upstream merge 节奏未定 | 全部 4 个 | 高 | 开发期可临时基于 `pr-tcm-cca` / `pr-dcae-saaf-auxt` 分支，等 upstream merge 后做 `git rebase --onto upstream/master <old-base>`（参考 pr-tcm-cca Phase 8.3 决策）|
| **CMIC / MambaIC License**：上游仓 README 未声明 license | PR-3, PR-4 | 中 | **已获作者邮件授权集成**（2026-05-10 user 沟通确认），merge 前补 SPDX header + 在 `COPYING` / `AUTHORS` 加致谢条目 + PR description 引用授权邮件 |
| `mamba_ssm` 在非 Linux+CUDA 环境无 wheel | PR-3, PR-4 | 中 | 严格可选 dep（`[ssm]` extras）；`compressai/layers/ssm/ssm.py` 三档回退（`selective_scan_cuda*` → `mamba_ssm.selective_scan_fn` → 纯 PyTorch `selective_scan_ref`）；缺失时 `@register_model` 跳过 MambaIC / CMIC |
| MambaIC 当前 import `compressai.models._bases`（已被 pr-tcm-cca 删）| PR-3 | 低 | 迁到 `compressai.latent_codecs._slice_helpers`（pr-tcm-cca 已搬过去），是 mechanical rewrite |
| CMIC 文件 1057 行（迁入路线图最大单文件之一）| PR-4 | 中 | 把 `_ContentAwareMamba` / `CMICChannelContextBlock` / `CMICSpatialContextBlock` 等内联 blocks 拆到 model 文件内部（不再 lift 到 `compressai/layers/`，因为 CMIC 独占）；总 LoC 不增反减 |
| MLIC++ 已有 `mlicpp-latent-codec-refactor.md` 历史决策 | PR-1 | 低 | 已 review 过的决策（model root 保留 `entropy_bottleneck` / `gaussian_conditional` 兼容属性 + `from_state_dict` 迁移旧 root-level keys）继续沿用 |

**License 的处理路径**（CMIC / MambaIC 共用）：
1. PR description 引用作者授权邮件（user 已联系，邮件存档）
2. 文件头加 InterDigital 标准 SPDX 注释 + 原作者署名
3. `COPYING` 加 third-party 致谢条目（指向上游仓 + 引用授权邮件日期）
4. 跟 reviewer (`@YodaEmbedding`) 在 PR conversation 里确认 License 处理符合 InterDigital 政策

---

## 6. 子 PR exec plan 索引

每个 PR 的详细 phase 级 exec plan：

- [pr-mlicpp-upstreaming.md](pr-mlicpp-upstreaming.md) — PR-1: **MLIC 系列**上游迁入（MLIC + MLIC+ + MLIC++ + MLICv2 合并 PR）+ **`MultiContextCheckerboardLatentCodec` 抽象**（MLIC++ Phase 1-4 + Phase 11.5 unify 已落地；v1/v1+ Phase 10-11 已落地；MLICv2 leaf 的 `selective_predictor` 可选 hook + MLICv2 layers + model/factory/zoo 已落地；Phase 5/15/16 待动手）；设计详见 [`mlic-family-reproduction.md`](../../design-docs/mlic-family-reproduction.md)
- [pr-glic-upstreaming.md](pr-glic-upstreaming.md) — PR-2: GLIC 上游迁入（含 `compressai/layers/graph/` 子包，复用 OLP/WLS/iWLS）
- [pr-mambaic-upstreaming.md](pr-mambaic-upstreaming.md) — PR-3: MambaIC + SSM 基础设施上游迁入（含 `MambaICLatentCodec` + `compressai/layers/ssm/` + `[ssm]` extras）
- [pr-cmic-upstreaming.md](pr-cmic-upstreaming.md) — PR-4: CMIC 上游迁入（复用 graph + ssm + wavelet 全套，纯 model + private blocks）

各子 plan 当前状态为 **scope 锁定 + phase 占位**（参考 pr-tcm-cca / pr-dcae-saaf-auxt 的 phase 结构），详细 phase 内容在真正动手前展开。

---

## 7. 总时间估算 + 完成后

**乐观顺序串行**（pr-tcm-cca + pr-dcae-saaf-auxt 先 merge）：~25 工作日。v3 把 MLIC / MLIC+ / MLICv2 合并进 PR-1 后，PR-1 从原先单 MLIC++ 的 ~5.5d 扩到 ~13d。
**实际**：因为 PR-1 不需要 pr-dcae-saaf-auxt、PR-2/PR-3 在 pr-dcae-saaf-auxt merge 后可并行，墙钟时间可压到 ~16-18 工作日（取决于 reviewer 节奏）。

完成后：
- 移动本路线图 + 4 个 PR plan 到 `plan/exec-plans/completed/`
- 更新 `plan/README.md` 索引
- 在 design doc `channel-slice-codec-redesign.md` §1 / §3.4 表把 GLIC/CMIC/MLIC++/MambaIC 状态从「fork `script` 已迁入」更新为「Family 2 PR 系列已合入 upstream」
- 把 `compressai/layers/ssm/` 标记为 SSM 家族 cross-model 共享层（MambaIC 已消费；~~未来 MambaVC 复用~~ MambaVC 已决策不迁入——仅 arXiv 预印本，见 lic-migration-roadmap §E.14 / Phase 11）
- Family 2 完成后剩余迁入对象按 lic-migration-roadmap：WeConvene (Phase 10) ✅ PR #7 + FTIC (Phase 8) ✅ PR #8 + InvCompress (Phase 9) ✅ PR #9，**均已迁入**（2026-06-03）；**MambaVC 已剔除**（仅 arXiv 预印本，未同行评审）。独立 model 候选池已清空
