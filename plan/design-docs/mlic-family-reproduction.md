# MLIC 家族（v1 / v1+ / v2）在 `MultiContextCheckerboardLatentCodec` 抽象上的复现设计

**文档日期**：2026-05-11
**状态**：设计 + 执行阶段（pr-mlicpp Phase 1-4 抽象层 + Phase 10-11 v1/v1+ application-layer / factory / thin model / zoo lazy factory 已落地；Phase 11.5 已把 MLIC++ 统一到 `_BaseMLIC` 模板并删除 `mlicpp.py`；Phase 12 已给 leaf 加 `selective_predictor` hook；Phase 13 已新增 MLICv2 layers 子包；Phase 14 已接入 MLICv2 model/factory/zoo；Phase 5 统一 convert script + MLIC++ 真实 ckpt smoke 已落地；Phase 15 联合验证已落地；Phase 16 push + PR draft 已完成，实际打开 upstream PR 等 user 时机；SGA Layer A codec generalization 已完成并归档）
**前置依赖**：[`pr-mlicpp-upstreaming.md`](../exec-plans/active/pr-mlicpp-upstreaming.md) 引入的 `MultiContextCheckerboardLatentCodec` + `compressai/layers/lic/mlic/` 应用层子包 + `build_mlic_slice_codec(variant=...)` factory（**2026-05-11 决策**：v1 / v1+ / v2 不再拆独立 PR，合并到 `pr-mlicpp` 同一 PR 内的 Phase 10-16；同期决定把 Phase 5 ckpt smoke 重排到 Phase 14 之后做，与 MLICv2 published ckpt 一起验证）
**参考论文**：
- *MLIC: Multi-Reference Entropy Model for Learned Image Compression*（ACMMM 2023, arxiv:2211.07273）—— 覆盖 MLIC + MLIC+
- *MLIC++: Linear Complexity Multi-Reference Entropy Modeling*（NCW ICML 2023 / ACM TOMM 2025, arxiv:2307.15421）
- *MLICv2: Enhanced Multi-Reference Entropy Modeling*（ACM TOMM 2025, arxiv:2504.19119）

**源码盘点**：`JiangWeibeta/MLIC` 仓只 release MLIC++ 实现（`candidate/MLIC/MLIC++/`）；MLIC v1 / MLIC+ / MLICv2 **无官方独立代码 release**，复现以论文为主线 + MLIC++ 源码为基底。

---

## 1. 家族演进概览

```
MLIC  ─── + inter-slice global ──→ MLIC+
 │                                   │
 │   conv stacked checkerboard       │   overlapped window attention
 │   quadratic global intra          │   quadratic global intra + inter
 │   N=192, M=320, slice_ch=32       │
 ▼                                   ▼
 (only conv local)              ─── replace quadratic w/ linear attn ──→ MLIC++
                                                                          │
                                                ┌─ replace ResBlock w/ STM
                                                ├─ HGCP (slice 0 也跑 global，via hyperprior)
                                                ├─ Context Reweighting (channel-wise attn)
                                                ├─ 2D RoPE replaces RPE
                                                ├─ GSC (post-training skip predictor)
                                                ▼
                                              MLICv2 ── + SGA inference-time refine ──→ MLICv2+
```

| 维度 | MLIC | MLIC+ | MLIC++ | MLICv2 | MLICv2+ |
|---|---|---|---|---|---|
| Transform | simplified Cheng'20 ResBlock + GDN/IGDN | 同 | 同 | **STM** (LN + DepthRB + DWConv5×5 + Conv1×1 + Gate) | 同 v2 |
| Local context (`spatial_context_nonanchor`) | **Stacked Checkerboard Conv** (3 conv5×5, 奇数 K, 中间无 norm) | **Overlapped Window Checkerboard Attention**（vanilla softmax) | 同 v1+（vanilla → 论文 §3.4.2 自述与 v1+ 同 module）| 同 ++（+ 2D RoPE 替代 RPE）| 同 v2 |
| Global intra (`intra_channel_context_nonanchor`) | **vanilla softmax attn**（quadratic）on prev slice y_hat + anchor | 同 v1 | **linear attn**（Softmax2(Q) · Softmax1(K) · V）+ Gate | 同 ++（+ CR + 2D RoPE）| 同 v2 |
| Global inter (slice→slice，外层 `channel_context`) | **无** | quadratic attn 跨所有 prev slices | linear attn 跨所有 prev slices | 同 ++（+ CR + 2D RoPE）| 同 v2 |
| Anchor side global（slice 0）| 无 | 无 | 无 | **HGCP**（用 hyperprior 互注意力）| 同 v2 |
| Slice 0 entropy_parameters input | 仅 `hyper_params` (2M) | 同 v1 | 同 v1 | `cat(hgcp_ctx, hyper_params)` | 同 v2 |
| LRP | per-pass `0.5·tanh(conv chain)` | 同 | 同 | 同 | 同 |
| Quantization | mixed (STE for distortion + AUN for entropy) | 同 | 同 | 同 + GSC skip map | + SGA latent refine |
| 训练数据 / batch | DIV2K+CLIC+COCO+ImageNet, 256→448, 2M steps | 同 | 同 + 512×512 from 1.2M steps, batch=32 | 同 ++（+ GSC post-train phase）| + SGA inference-time |

---

## 2. `MultiContextCheckerboardLatentCodec` 抽象的覆盖度

PR-1 (`pr-mlicpp`) Phase 1 落地的 leaf 接口：

```python
MultiContextCheckerboardLatentCodec(
    entropy_parameters_anchor:           nn.Module,                 # 必填
    entropy_parameters_nonanchor:        nn.Module,                 # 必填
    spatial_context_anchor:              Optional[nn.Module] = None,
    spatial_context_nonanchor:           Optional[nn.Module] = None,
    intra_channel_context_nonanchor:     Optional[nn.Module] = None,
    lrp_anchor:                          Optional[nn.Module] = None,
    lrp_nonanchor:                       Optional[nn.Module] = None,
    lrp_input_builder:                   Optional[Callable]  = None,
    lrp_activation:                      Optional[Callable]  = torch.tanh,
    lrp_scale:                           float = 0.5,
    anchor_parity:                       str = "even",              # ⚠ MLIC 家族 ckpt 用 "odd"
    gaussian_conditional:                EntropyModel,              # 必填
)
```

**家族 → leaf 槽位映射**：

| 模型 | spatial_anchor | spatial_nonanchor | intra_channel_nonanchor | lrp_anchor / lrp_nonanchor |
|---|---|---|---|---|
| MLIC | None | StackedCheckerboardConv (3 conv5×5, 奇数 K) | k>0: VanillaGlobalIntraContext(prev_y_hat, anchor_y_hat) | LatentResidualPrediction |
| MLIC+ | None | OverlappedWindowCheckerboardAttn (vanilla softmax) | k>0: VanillaGlobalIntraContext | 同 |
| MLIC++ | None | LocalContext（overlapped window, vanilla 内部）| k>0: LinearGlobalIntraContext | 同 |
| MLICv2 | **k==0**: HGCPModule（互注意力 hyperprior anchor↔nonanchor）; **k>0**: None | LocalContext | k>0: LinearGlobalIntraContext + **CR + 2D RoPE** | 同 |

**leaf 抽象覆盖度 = 100%** for v1 / v1+ / ++ / v2 模型主体。v2 的 **GSC** 需要改变熵编码符号集，Phase 12 已给 leaf 补 `selective_predictor` hook；Phase 13 已实现 `GSCModule`；Phase 14 已在 factory 中注入。详见 §4.3。当前 Phase 14 实现把 `ContextReweighting` + `RoPE2D` 作为 application wrapper 包在 LinearGlobalInter/IntraContext 输出外，未改 MLIC++ 既有 `LocalContext` 内部 attention。

---

## 3. PR-1 (pr-mlicpp) 抽象层的小修

为了让同一抽象同时承载 v1 / v1+ / v2，pr-mlicpp 已完成以下小修。Phase 11 已回补 `anchor_parity="odd"`；Phase 12 已补 `selective_predictor` hook；Phase 13 已实现 MLICv2 layer building blocks；Phase 14 已完成 MLICv2 slot 接线。

### 3.1 anchor_parity = "odd" P0 修复

上游 `candidate/MLIC/MLIC++/utils/ckbd.py:36-44` 的 `ckbd_anchor` 把 anchor 放在 `[..., 0::2, 1::2]` + `[..., 1::2, 0::2]`（i+j **odd**）。compressai `_checkerboard_helpers.py:76-80` 的 `anchor_parity="even"` 把 anchor 放在相反位置。

**MLIC-family factory 内的 `MultiContextCheckerboardLatentCodec(...)` 必须显式传 `anchor_parity="odd"`** —— 这条规则对整个 MLIC 家族（v1/v1+/++/v2 同样训自相同 ckbd convention）通用。Phase 11 / 11.5 / 14 已在 `build_mlic_slice_codec` 的 MLIC / MLIC+ / MLIC++ / MLICv2 variants 中回补该代码修复。

ELIC / Cheng2020 等用 upstream `CheckerboardLatentCodec` 的现有模型不受影响（它们的 ckpt 与 compressai "even" convention 对齐）。

### 3.2 anchor 槽位首次启用（仅 v2 需要）

leaf 的 `spatial_context_anchor` 在 PR-1 设计时预留但未使用。MLICv2 的 HGCP 需要为 slice 0 提供 anchor-side spatial context（用 hyperprior 的 anchor / nonanchor 互注意力）。

Phase 14 为此槽位补了 side-params aware spatial context 调用：普通 spatial context 仍只接收 `y_hat`，带 `requires_side_params=True` 的 HGCP wrapper 额外接收 `side_params`，从而能读取 hyperprior params。factory 内只在 k==0 注入 HGCP module。

但 PR-1 的 §state_dict 路径表 应该补一行 `latent_codec.y.latent_codec.y{k}.spatial_context_anchor.*`（即便 MLIC++ 不用，文档中记录槽位存在）。

### 3.3 `lrp_input_builder` 的 anchor / nonanchor pass 语义文档化

目前文档不够明确 nonanchor pass 时 builder 能看到的 "current_slice" 是 (anchor post-LRP) 还是 (anchor post-LRP + nonanchor pre-LRP)。Phase 3 实施差异里已澄清是后者，但需要补到 leaf 类 docstring 里，避免 v2 复现者重新踩坑。

---

## 4. 新增 application-layer building blocks

按家族成员归类。所有新 block 放 `compressai/layers/lic/mlic/`（v1/v1+ 与 ++ 共享同一子包，避免按论文版本切分）或 v2 专用子包 `compressai/layers/lic/mlicv2/`（如果改动量大）。

### 4.1 MLIC v1 / v1+ 新增

| Block | 落点 | 实现要点 |
|---|---|---|
| `StackedCheckerboardConv(dim, kernel=5, num_layers=3)` | `compressai/layers/lic/mlic/context.py` 追加 | 3 层 stride=1 conv5×5（奇数层；中间无 norm），每层后 GELU，最后不激活；接受 anchor_y_hat → 输出 `(B, 2*dim, H, W)` 给 nonanchor EP head |
| `VanillaGlobalIntraContext(dim)` | 同上 | `LinearGlobalIntraContext` 的 quadratic 版本：vanilla softmax(Q·Kᵀ/√d_k)·V + relative_pos_bias，签名同 `LinearGlobalIntraContext(prev_y_hat, anchor_y_hat)` |
| `VanillaGlobalInterContext(in_dim, out_dim, num_heads)` | 同上 | `LinearGlobalInterContext` 的 quadratic 版本（仅 MLIC+ 用，MLIC 无） |
| `WindowCheckerboardAttn(dim, window_size)` | 同上 | MLIC+ 的 overlapped window attention（vanilla softmax，with mask避免 anchor↔anchor 信息泄漏；详见论文 fig 7-8） |

### 4.2 MLICv2 新增

| Block | 落点 | 实现要点 |
|---|---|---|
| `SimpleTokenMixing(dim)` | `compressai/layers/lic/mlicv2/transforms.py` | `timm.layers.LayerNorm2d` + DepthRB + DWConv5×5 + Conv1×1 + (`LayerNorm2d` + Gate)；Gate = `LN→Conv1×1→GELU→DWConv3×3→GELU→Conv1×1→sigmoid * raw`（论文 §3.3） |
| `STMAnalysis` / `STMSynthesis` | 同上 | g_a/g_s 把 ResidualBlock 替成两个 STM block 串联；其它 stage 跟 MLIC++ 一致（N=192, M=320, 4 stride-2 stages） |
| `HGCPModule(M, slice_ch)` | `compressai/layers/lic/mlicv2/context.py` | 用 hyperprior anchor/nonanchor 互注意力预测 slice 0 的 anchor-side global context；放进 leaf 的 `spatial_context_anchor` 槽位（仅 k==0） |
| `ContextReweighting(dim)` | 同上 | channel-wise softmax(Q·K)·V + Gate；wrap 在 LinearGlobalIntra/InterContext 后；详见论文 §3.4.2 + fig 5 |
| `RoPE2D(dim, learnable_thetas=True)` | 同上 | 2D Rotary Position Embedding（替代 RPE 加 bias 方案）；初始化 θ_x = θ_y = 10000，可学；详见论文 §3.4.3 + eq (9-10) |
| `GSCModule(slice_ch, threshold=0.3)` | 同上 | 后训练 skip map 预测器，输入 scale + prev slices + hyperprior，输出 sigmoid 的 selective_map（详见 §4.3） |

### 4.3 GSC（仅 MLICv2，本节单独展开）

**为什么 GSC 不能塞 leaf**：GSC 改变熵编码的符号集合——只对 |y - μ| ≥ threshold 的元素进行算术编码，其余跳过。它在 forward / compress / decompress 三处都改 likelihoods / strings 的语义：

| 路径 | 改动 |
|---|---|
| forward | 训练时 GSC module 在 anchor / nonanchor pass 产出 `s_anchor / s_nonanchor ∈ {0,1}`；rate loss 只对 `s==1` 的位置累加 cross-entropy；同时单独 cross-entropy loss 监督 `s` 的二分类 |
| compress | 只对 `s==1` 的位置写 bit stream；解码端用同一 GSC module 重算 `s`（无需写额外 side info）|
| decompress | 用 GSC 重算 `s`，按 `s` 决定从 bitstream 读 / 跳过；跳过的元素填 means_anchor / means_nonanchor |

**两种实现路径**：

**(A) 给 `MultiContextCheckerboardLatentCodec` 加 `selective_predictor: Optional[nn.Module]` hook**
- 好：保持单一 sibling leaf，v1 / v1+ / ++ / v2 一律走它
- 坏：leaf 类 forward / compress / decompress 都要分支「有 / 无 GSC」；逻辑更密
- 坏：GSC 的训练 phase 二阶段（先冻 backbone 训 GSC 再联调）在 leaf 里表达不优雅

**(B) 新 sibling `SelectiveMultiContextCheckerboardLatentCodec`**（继承现有 leaf 或独立类）
- 好：现有 leaf 保持简洁；MLIC++ Phase 4 测试 0 改动
- 坏：~250 LoC 与 base leaf 重复
- 中：与 `_checkerboard_helpers.py` single source of truth 收益类似——可以再抽一层 `_apply_selective_compression` helper 复用

**推荐**：先按 **(A)** 走，给 leaf 加 `selective_predictor` 可选 hook（默认 None）；只有 MLICv2 PR 启用。如果 review 反馈 leaf 变复杂超过 1.3×，再 pivot 到 (B)。

### 4.4 MLICv2+ — Stochastic Gumbel Annealing（已落地于 pr-mlicpp Phase 17）

MLICv2+ 在 v2 model 上叠加 inference-time latent re-optimization，用 Yang et al. NeurIPS 2020 ([arxiv:2006.04240](https://arxiv.org/abs/2006.04240)) 提出的 Stochastic Gumbel Annealing。论文 §3.5 把它套到 v2 上：单图推理时把 latent y 和 hyper-latent z 包成 `nn.Parameter`，用 Adam 在 SGA-quantized RD loss 上 ~2000 iter 微调，然后 round 到最终 y_hat 走 entropy coding。

**关键洞察**：SGA 对 model arch 完全正交。它只改 **quantizer**：把 codec 内部的 STE round 替换成 stochastic gumbel sample，温度 `T = 0.5 * exp(-1e-3 * (it - 0.35*total_iter))` 退火到 0。任何 LIC 模型（ScaleHyperprior 也行，论文用它做 ablation）都能用，**MLICv2+ ≈ MLICv2 + 调用 SGA 推理工具**，没有新 model class。

**落地决策**（同 §4.3 GSC 选项 A）：SGA 作为 codec 的 quantizer 选项（与 `quantize_ste` / `"noise"` 同位），不走 monkey-patch / sibling fork。新增 `compressai.ops.SGAQuantizer(nn.Module)`，与 `quantize_ste` 同级；`EntropyBottleneckLatentCodec(quantizer="sga", sga=...)` 和 `MultiContextCheckerboardLatentCodec(quantizer="sga", sga=...)` 接受同一 SGA module 实例（全模型共享）；`_BaseMLIC.set_sga_mode(sga | None)` 一键切换全部内部 codec。

**model-side API**（已加到 `_BaseMLIC`）：
- `refine_extract(x) -> (y, z)`：跑一次 g_a + h_a 取初始 latent
- `refine_forward(y, z) -> {x_hat, likelihoods}`：跳过 g_a/h_a，复用 HyperpriorLatentCodec 内部接力（z_codec → h_s → y_codec → g_s）
- `set_sga_mode(sga: SGAQuantizer | None)`：切换全部 leaf 与 z_codec 到 SGA mode（共享同一实例），None 时 revert

**SGA 与 GSC 交互**（v2+ 独有问题）：fresh-init MLICv2 的 GSCModule 输出全 False mask（GSC 没训练过）→ `apply_selective_y_hat` 把 y_hat 全替成 means → y_hat 与 y 解耦 → SGA 梯度死掉。**这是预期行为**，不是 bug：trained ckpt 上 GSC 输出 True/False 混合 mask，SGA 梯度正常流动。测试套件里 MLICv2 只做接口 sanity（shape 对、set_sga_mode 成功）；MLIC++（无 GSC）做 SGA refine 实跑（50 iter RD loss 真实下降）。PR description 须显式说明 v2+ 测试需要 trained ckpt。

**LoC**：~+460（详见 pr-mlicpp Phase 17）。

**Layer A codec generalization 已完成**：后续独立计划 [`sga-codec-generalization.md`](../exec-plans/completed/sga-codec-generalization.md) 已归档，commit `9cdcb05 feat(latent_codecs): generalize sga quantization` 把 `quantizer="sga"` 推广到 `GaussianConditionalLatentCodec` / `CheckerboardLatentCodec`，并通过父类继承覆盖 `LRPGaussianLatentCodec`。上游没有对应 ELIC checkpoint 可做 ckpt 数值回归；当前以 default STE 行为不变、targeted codec tests、`tests/test_sga.py` 25 passed 作为完成标准。Layer B（通用 attach/refine helper）仍保持本计划之外，等更多模型家族有真实需求再设计。

---

## 5. 单 PR 内 phase 拆分建议

**决策（2026-05-11）**：MLIC + MLIC+ + MLIC++ + MLICv2 + MLICv2+(SGA) 合并到 **同一个 PR**（`pr-mlicpp` 分支扩展 scope），不拆 PR-5 / PR-6 / pr-sga。SGA 作为 v2+ 等价件，与 v2 model 在同 PR 自然耦合。

**理由**：
- 四个模型共享 90% 抽象：同一 `MultiContextCheckerboardLatentCodec` + 同一 `compressai/layers/lic/mlic/` application-layer 子包 + 同一 `build_*_slice_codec` factory 模式
- v1 / v1+ / v2 都**没有官方独立代码 release**（只 ++ 有），复现是凭论文，reviewer 一次看完整家族演进比分三次看碎片更高效
- 一次 review cycle vs 三次 —— InterDigital reviewer 节奏可观察值，单 PR 大但少 ping
- 抽象层（leaf + helpers）只新增一次，四模型一起验证它的通用性
- ckpt 验证只对 MLIC++ 一档可做（v1 / v1+ / v2 暂无 published ckpt），分 PR 也不会让 ckpt 验证更彻底
- SGA 是 inference utility 单文件 + 两 codec hook + model API；与 v2 model 同 PR 让 reviewer 看到完整 v2+ 工作链

**风险**：PR diff 体量大（~3300 LoC 净增），reviewer 可能要求拆。Mitigation：commit 序列高度结构化（按 phase / 按模型），每个 commit 独立可 review；若 reviewer 真要求拆，按 phase 边界把后半截（v1/v1+/v2 part）单独切到 follow-up PR。

### 5.1 PR 内 phase 序列（在 `pr-mlicpp-upstreaming.md` Phase 0-9 之后追加）

> **执行顺序调整（2026-05-11，Phase 11 完成后 / Phase 12 启动前）**：原 §5.1 表把 Phase 5（convert script + 上游 ckpt smoke）放在 Phase 11 / Phase 12 之间。**新决策**：Phase 5 延后到 **Phase 14 之后**，与 MLICv2 published ckpt（如有）一起统一验证。理由：(a) MLIC v1 / v1+ 无 published ckpt（design doc §7 risk），ckpt smoke 只能验 MLIC++ 一档，分散做没收益；(b) 把 4 个模型 wiring 全部到位后再回头打 ckpt，能在同一 convert script 框架内复用 path-rename 模板（per-variant `_LEGACY_LIST_RENAMES` 表 + 共享 `_strip_data_parallel_prefix` / `_infer_*` helper）；(c) Phase 5 之前的所有 wiring 已通过 self-consistent `from_state_dict(model.state_dict())` round-trip 测试覆盖正确性，published ckpt 验证只是补 byte-for-byte 兼容。
>
> **Phase 11.5 已完成（MLIC++ 统一到 `_BaseMLIC` 模板）**：`compressai/models/mlicpp.py` 内容已合并到 `compressai/models/mlic.py`，`mlicpp.py` 物理文件已删除（不留 shim）。详见 §5.1.2。
>
> **新执行顺序**：1-4 ✅ → 10-11 ✅ → **11.5 ✅** → 12-14 ✅ → **5 ✅** → 15-16

| Phase | 范围 | 估算 | 状态 |
|---|---|---|---|
| **Phase 0-4** | MLIC++ 抽象层 + 应用层 + thin model（Phase 4 临时落在 `compressai/models/mlicpp.py`，Phase 11.5 合到 `mlic.py`）| 4 d | ✅ 完成于 2026-05-10 |
| **Phase 5** | convert script + 上游 published ckpt smoke：统一 `examples/convert_mlic_checkpoint.py --variant {mlic,mlic+,mlicpp,mlicv2}` + MLIC++ 真实 ckpt strict-load + sinusoidal smoke PSNR + fresh-init 对照；MLIC / MLIC+ / MLICv2 因无 published ckpt skip 真实 ckpt 验证（保留 fresh-init self-consistent 测试）| 1 d | ✅ 完成于 2026-05-11 |
| **Phase 6** | model tests | 0.5 d | ✅ 完成于 2026-05-11（Phase 4 加了 TestMlicPlusPlus，Phase 11 加了 TestMlicFamily，Phase 14 已追加 TestMlicv2；Phase 5 追加 MLIC++ model-level compress/decompress round-trip）|
| **Phase 7** | zoo 接线 | 0.5 d | ✅ 完成于 2026-05-11（mlic / mlicplus / mlicpp / mlicv2 均为 lazy factory；`pretrained=True` 对无 hosted 权重的模型 raise）|
| **Phase 8 / 15** | 联合验证 | 0.5 d | ✅ 完成于 2026-05-11（targeted regression / broad regression / import audit / lock check 已记录；本地 macOS DDP smoke 仍交给 Linux CI 复验）|
| **Phase 9 / 16** | 提交 + push + PR draft | 0.5 d | ✅ 完成于 2026-05-12（commits `b0924fc` / `9cdcb05` 已创建，`origin/pr-mlicpp` 已在 `9cdcb05`，PR draft 写入 `plan/generated/pr-mlicpp-draft.md`；实际打开 upstream PR 等 user 时机） |
| **Phase 10** | MLIC + MLIC+ 共用的 application-layer building blocks | 1 d | ✅ 完成于 2026-05-11 |
| **Phase 11** | MLIC + MLIC+ slice factory + thin model + Phase 4.5 anchor_parity 修复回补到 MLIC-family factories | 1.5 d | ✅ 完成于 2026-05-11 |
| **Phase 11.5（NEW）** | MLIC++ 统一到 `_BaseMLIC` 模板：`MLICPlusPlus` + `convert_upstream_mlicpp_state_dict` + 4 个 `_infer_*` helper + `_LEGACY_*` regex / table 全部从 `compressai/models/mlicpp.py` 内联到 `compressai/models/mlic.py`；删除 `mlicpp.py`（不留 shim，import 路径 breaking change：`from compressai.models.mlicpp import MLICPlusPlus` → `from compressai.models.mlic import MLICPlusPlus`）；扩展 `build_mlic_slice_codec(variant=Literal["mlic","mlic+","mlicpp"])` 支持 "mlicpp"；删除 `build_mlicpp_slice_codec`（test 调用点改 `build_mlic_slice_codec(variant="mlicpp")`）；`MLICPlusPlus.from_state_dict` 通过 `_BaseMLIC._legacy_convert` 前处理 legacy ckpt；zoo `_LazyImport` target 同步改；Phase 4 / Phase 11 测试零回归 | 0.5 d | ✅ 完成于 2026-05-11 |
| **Phase 12** | MLICv2 leaf hook：`MultiContextCheckerboardLatentCodec` 加可选 `selective_predictor: Optional[nn.Module]` + `_selective_checkerboard.py` 内部 helper；默认 None 时与 PR-1 Phase 1 行为完全等价；启用时 skipped symbols 不写入 bitstream 且 y_hat 填 means；新增 `test_selective_predictor_none_is_identity` / `test_selective_predictor_skip_semantics` | 0.5 d | ✅ 完成于 2026-05-11 |
| **Phase 13** | MLICv2 子包：新建 `compressai/layers/lic/mlicv2/{__init__,transforms.py,context.py}`，含 `SimpleTokenMixing` / `STMAnalysis` / `STMSynthesis` / `HGCPModule` / `ContextReweighting` / `RoPE2D` / `GSCModule`；单元测试每个 block 的 forward shape + state_dict + 关键不变量（RoPE 位置不变性、CR channel attention 形状、HGCP slice-0 anchor context shape、GSC skip-rate）| 2 d | ✅ 完成于 2026-05-11 |
| **Phase 14** | MLICv2 model + factory：`build_mlic_slice_codec(variant="mlicv2")`（注入 HGCP 到 anchor 槽位 / CR + 2D RoPE wrapper 包到现有 context blocks / 注册 GSC 到 `selective_predictor`）；`MLICv2(_BaseMLIC, _variant="mlicv2")` 直接加进 `compressai/models/mlic.py`（**不**新建 `mlicv2.py`）；单元测试 forward + state_dict + GSC skip-rate sanity check；zoo `mlicv2` lazy factory；4 个 thin model 全部到位 | 1.5 d | ✅ 完成于 2026-05-11 |

**Phase 10-16 总计**：~7.5 工作日（在已落地的 Phase 1-4 之上）；Phase 10-14 与重排后的 Phase 5/6 已完成，剩余 Phase 15-16 约 1 工作日。单 PR 全部完成约 13 工作日。

### 5.1.2 Phase 11.5 设计细节

**目标**：4 个 MLIC family 模型（v1 / v1+ / ++ / v2）共一个文件 `compressai/models/mlic.py`，共一个 `_BaseMLIC` 模板，共一个 `build_mlic_slice_codec(variant=...)` factory。无 `compressai/models/mlicpp.py`，无独立 `compressai/models/mlicv2.py`。

**Phase 4 / Phase 10-11 结构现状（Phase 11.5 前）**：

```
compressai/models/mlic.py    -- _BaseMLIC + MLIC + MLICPlus
compressai/models/mlicpp.py  -- MLICPlusPlus（独立，跟 _BaseMLIC 平行）
compressai/models/_helpers/multi_context_slice.py
    build_mlic_slice_codec(variant=Literal["mlic","mlic+"])
    build_mlicpp_slice_codec(...)
```

**Phase 11.5 后**：

```
compressai/models/mlic.py    -- _BaseMLIC + MLIC + MLICPlus + MLICPlusPlus
                                + convert_upstream_mlicpp_state_dict (legacy convert helper)
compressai/models/mlicpp.py  -- (deleted)
compressai/models/_helpers/multi_context_slice.py
    build_mlic_slice_codec(variant=Literal["mlic","mlic+","mlicpp"])
    -- build_mlicpp_slice_codec deleted
```

**Phase 14 后**：

```
compressai/models/mlic.py    -- _BaseMLIC + MLIC + MLICPlus + MLICPlusPlus + MLICv2
                                + convert_upstream_mlicpp_state_dict (legacy convert helper)
compressai/models/mlicpp.py  -- (deleted)
compressai/models/mlicv2.py  -- (never created)
compressai/models/_helpers/multi_context_slice.py
    build_mlic_slice_codec(variant=Literal["mlic","mlic+","mlicpp","mlicv2"])
```

**2026-05-11 实施结果**：上述结构已落地；`tests/test_models.py::TestMlicPlusPlus` 的 legacy conversion、`tests/test_models.py::TestMlicv2` 的 state_dict round-trip、`tests/test_models_helpers.py::TestBuildMlicppSliceCodec` / `TestBuildMlicv2SliceCodec` 的 state_dict path 断言和 `tests/test_zoo.py::TestMlicZoo` 的 lazy factory 均已通过。

**关键修改点**：

1. **`build_mlic_slice_codec` 扩展 variant 支持**：
   - `variant="mlic"`: `StackedCheckerboardConv` local + 无 global inter + `VanillaGlobalIntraContext` intra
   - `variant="mlic+"`: `WindowCheckerboardAttn(=LocalContext)` local + `VanillaGlobalInterContext` inter + `VanillaGlobalIntraContext` intra
   - `variant="mlicpp"` (NEW): `LocalContext` local + `LinearGlobalInterContext` inter + `LinearGlobalIntraContext` intra
   - `variant="mlicv2"` (Phase 14): + HGCP / CR / RoPE / GSC

2. **`_MlicppPriorAggregation` 的 `global_inter_factory` 参数已支持 callable，加 `_build_linear_global_inter_context` / `_build_vanilla_global_inter_context` 两 factory 选择即可**（Phase 11 已就位）

3. **`_BaseMLIC.from_state_dict` 加 `_legacy_convert` hook**（最小侵入式扩展）：
   ```python
   class _BaseMLIC(CompressionModel):
       _variant = "mlic"
       _legacy_convert: Optional[Callable] = None  # 子类 override
       
       @classmethod
       def from_state_dict(cls, state_dict):
           if cls._legacy_convert is not None:
               state_dict = cls._legacy_convert(state_dict)
           ... # 现有 N/M/slice_num/context_window 推断 + load_state_dict
   ```
   
   `MLICPlusPlus._legacy_convert = staticmethod(convert_upstream_mlicpp_state_dict)` 即可

4. **删除文件**：`compressai/models/mlicpp.py`（290 LoC）

5. **新文件 LoC**：`compressai/models/mlic.py` 从 262 → ~430（净增 ~170：MLICPlusPlus class ~30 + convert helpers + regex ~140）

6. **`compressai/models/_helpers/multi_context_slice.py` 收缩**：删 `build_mlicpp_slice_codec` (~50 LoC) + 加 `variant="mlicpp"` 分支到 `build_mlic_slice_codec` (~10 LoC)，净减 ~40 LoC

7. **Test 改动**（**non-trivial**）：
   - `tests/test_models.py::TestMlicPlusPlus`: `from compressai.models.mlicpp import MLICPlusPlus, convert_upstream_mlicpp_state_dict` → `from compressai.models.mlic import MLICPlusPlus, convert_upstream_mlicpp_state_dict`
   - `tests/test_models_helpers.py::TestBuildMlicppSliceCodec`: `from compressai.models._helpers.multi_context_slice import build_mlicpp_slice_codec` → `from compressai.models._helpers.multi_context_slice import build_mlic_slice_codec` + 改 `build_mlicpp_slice_codec(...)` 为 `build_mlic_slice_codec(variant="mlicpp", ...)`
   - 测试 path 断言保持不变（state_dict 路径 byte-compatible）
   - **`pretrained=True` 路径**：`compressai/zoo/image.py::mlicpp` 函数体内 `from compressai.models.mlicpp import MLICPlusPlus` → `from compressai.models.mlic import MLICPlusPlus`
   - **`_LazyImport` proxy**：`_LazyImport("compressai.models.mlicpp", "MLICPlusPlus")` → `_LazyImport("compressai.models.mlic", "MLICPlusPlus")`

8. **净 LoC 影响（Phase 11.5）**：
   - `mlicpp.py` 删除 -290
   - `mlic.py` 扩展 +170
   - `multi_context_slice.py` 收缩 -40
   - test imports 改 ±0
   - **Phase 11.5 净 -160 LoC**

9. **回滚策略**：计划中的 `git tag pr-mlicpp-pre-unify` 暂未执行，因为当前 Phase 1-11.5 仍是未提交工作树改动，给 `HEAD` 打 tag 不能覆盖 pre-unify 工作树状态；待 commit 分组前再按实际 commit 边界补 tag / backup。

**风险评估**：

| 风险 | 严重度 | 缓解 |
|---|---|---|
| Linear vs Vanilla module submodule 名 / shape 差异 | 低 ✅ 已验证 | 5.1.1 已 grep 验证 `_pointwise_then_dwconv` / `reprojection` / `mlp` 等 submodule 名 + shape 完全一致 |
| 删 `compressai/models/mlicpp.py` 后 user 代码 `from compressai.models.mlicpp import ...` breaking | 低 | mlicpp.py 没 push 到 origin（只在 pr-mlicpp 本地分支），无下游 user 代码依赖；upstream 也没合 |
| Phase 4 `TestMlicPlusPlus.test_legacy_state_dict_conversion` 跑挂 | 低 | `convert_upstream_mlicpp_state_dict` 函数体不变，只换 import 来源；test 内部断言 path-string 也不变 |
| zoo `_LazyImport("compressai.models.mlicpp", ...)` 还引用旧 path | 低 ✅ | Phase 11.5 已把 `mlicpp` factory / `_LazyImport` 指向 `compressai.models.mlic`，`TestMlicZoo.test_factories` 已覆盖 |

### 5.1.3 Phase 5 重排理由

**原排（设计 doc 初版）**：Phase 5 在 Phase 11 / Phase 12 之间，先把 MLIC++ ckpt 验证完再启动 MLICv2。

**新排（2026-05-11 决策）**：Phase 5 重排到 Phase 14 之后，与 MLICv2 published ckpt（如有）一起验证。

**理由**：
1. **MLIC v1 / v1+ 无 published ckpt**（design doc §7 risk 表明确）—— Phase 5 ckpt smoke 只能验 MLIC++ 一档；分散做没批量优势
2. **convert script 可模板化**：4 个模型用同一 `_strip_data_parallel_prefix` / `_LEGACY_LIST_RENAMES` / `_infer_*` 模板，一次写 `examples/convert_mlic_checkpoint.py` 通过 `--variant {mlic,mlic+,mlicpp,mlicv2}` flag 处理 4 个模型；Phase 14 之后做能复用 `_BaseMLIC._legacy_convert` 钩子
3. **Self-consistent 验证已覆盖正确性**：Phase 4 `test_forward_and_state_dict_round_trip` + Phase 11 `TestMlicFamily` + Phase 14 `TestMlicv2` 已经通过 `from_state_dict(model.state_dict())` round-trip 验证 model wiring 正确；published ckpt smoke 只是额外补 byte-for-byte 上游兼容
4. **避免 Phase 11.5 unify 与 Phase 5 ckpt smoke 时序耦合**：如果 Phase 5 在 Phase 11.5 前做，会把「上游 ckpt 加载是否成功」和「unify 是否引入 regression」两件事绑在一起；分开做更容易定位问题
5. **批量验证的 reviewer 价值**：PR description 一次性给 4 个模型的 published ckpt smoke 结果（MLIC++ 真实 + 其他 3 个 self-consistent），比分散在中间 Phase 给单个验证更有说服力

**风险（已关闭）**：Phase 5 推迟意味着 Phase 11.5 + 12-14 期间任何潜在的「上游 ckpt 路径漂移」bug 不会被发现。**缓解 / 结果**：Phase 11.5 已跑 `MLICPlusPlus.from_state_dict` 加载 fork-script-style synthetic state_dict（`test_legacy_state_dict_conversion`），证明 unify 没破坏 convert path；Phase 5 已用 `candidate/MLIC/mlicpp_mse_q5_2960000.pth.tar` 完成 published ckpt strict-load + smoke。

### 5.1.1 Phase 11 回填（2026-05-11）

- `build_mlic_slice_codec(variant=Literal["mlic", "mlic+"])` 已落地，并在 Phase 11.5 扩展到 `variant="mlicpp"`；复用 MLIC++ 的 side layout / LRP input / entropy-parameter wrapper；`variant="mlic"` 不注入 global inter，`variant="mlic+"` 注入 `VanillaGlobalInterContext`，`variant="mlicpp"` 注入 `LinearGlobalInterContext`。
- `compressai/models/mlic.py` 已落地，包含 `MLIC` 与 `MLICPlus` 两个 thin model；默认配置按 MLIC 论文 Table 2（MLIC: `N=192, M=192, slice_num=6`；MLIC+: `N=192, M=320, slice_num=10`）。
- zoo 已接入 `mlic` / `mlicplus`，通过 `_LazyImport("compressai.models.mlic", ...)` 保持 optional `[attn]` 依赖 lazy；`pretrained=True` 因无官方 v1/v1+ ckpt 暂时 raise。
- Phase 4.5 `anchor_parity="odd"` 代码修复已同步回补到 `build_mlic_slice_codec` 的 MLIC / MLIC+ / MLIC++ variants；真实 MLIC++ published ckpt smoke 已在 Phase 5 完成。
- 验证：Phase 11 targeted tests 17 passed；组合回归 57 passed；`make static-analysis` passed；`git diff --check` clean；import audit 确认 `import compressai` / `compressai.models` / `compressai.zoo` 不加载 `timm`。

### 5.2 commit 序列建议

按 phase 边界打 commit，每个 commit 独立可 review。**新执行顺序**（Phase 5 / Phase 7 / Phase 8 / Phase 9 在 Phase 14 之后做，与 MLICv2 一起验证）：

```
# Phase 1-4（已落地，6 commits）
feat(latent_codecs): MultiContextCheckerboardLatentCodec sibling of CheckerboardLatentCodec  (Phase 1)
refactor(latent_codecs): extract _checkerboard_helpers.py shared by both checkerboard codecs (Phase 1)
feat(layers/lic/mlic): MLIC++ application-layer building blocks                              (Phase 2)
feat(models/_helpers): build_mlicpp_slice_codec factory                                       (Phase 3)
feat(models): MLICPlusPlus thin model + state_dict legacy compat                             (Phase 4)

# Phase 10-11（已落地，2 commits）
feat(layers/lic/mlic): MLIC v1 / MLIC+ application-layer building blocks                     (Phase 10)
feat(models): MLIC + MLICPlus thin models + factory + zoo + Phase 4.5 anchor_parity fix      (Phase 11)

# Phase 11.5（已落地）
refactor(models): unify MLIC++ into _BaseMLIC template; delete compressai/models/mlicpp.py   (Phase 11.5)

# Phase 12-14
feat(latent_codecs): MultiContextCheckerboardLatentCodec.selective_predictor hook            (Phase 12)
feat(layers/lic/mlicv2): STM transforms + HGCP + CR + 2D RoPE + GSC                          (Phase 13)
feat(models): MLICv2 added to mlic.py + factory + zoo                                        (Phase 14)

# Phase 5（重排到 Phase 14 之后）
feat(examples): convert_mlic_checkpoint.py with --variant flag (mlic / mlic+ / mlicpp / mlicv2)  (Phase 5)
test(models): MLIC++ published ckpt smoke + MLICv2 published ckpt smoke (if available)            (Phase 5)

# Phase 8/15 + Phase 9/16（合并最终 commit 系列）
chore(*): joint validation pass + import audit + uv.lock check                               (Phase 15)
docs(plan): close out pr-mlicpp-upstreaming.md, mlic-family-reproduction.md                  (Phase 16)
```

如果 reviewer 要求拆 PR，建议的拆分边界是 Phase 11.5 / Phase 12 之间（MLIC + MLIC+ + MLIC++ 是一个 self-contained PR，MLICv2 作为 follow-up PR 复用同一抽象），**比初版「Phase 9 / Phase 10 之间」更优**：unify 后边界更干净，且 MLICv2 引入的新依赖（GSC selective hook + 2D RoPE）天然适合作为单独 PR。

---

## 6. 路线图位置

合并到现有 `pr-mlicpp` 单 PR，**不**新建 PR-5 / PR-6。`family2-roadmap.md` 仍是 4 PR 结构：

```
PR-1 pr-mlicpp        MLIC 系列（MLIC + MLIC+ + MLIC++ + MLICv2 合并）
PR-2 pr-glic
PR-3 pr-mambaic
PR-4 pr-cmic
(已并入 PR-1) sga-codec-generalization  MLICv2+ inference refine + generic codec Layer A（commit `9cdcb05`；Layer B 通用 attach/refine helper 如有需要再单独评估）
```

PR-1 内部按 §5.1 phase 序列实施。Phase 10-11 已按 user 指令先落地 v1/v1+ 代码，并同步回补 Phase 4.5 的 `anchor_parity="odd"` 代码修复；Phase 11.5 已把 MLIC++ unify into `mlic.py` 并删除 `mlicpp.py`；Phase 12 已给 `MultiContextCheckerboardLatentCodec` 补 `selective_predictor` hook；Phase 13 已新增 `compressai/layers/lic/mlicv2/`；Phase 14 已把 `variant="mlicv2"`、`MLICv2` thin model 和 zoo lazy factory 接入；Phase 5 已完成 4 模型统一 convert CLI 与 MLIC++ 真实 ckpt smoke；Phase 15 联合验证已落地；Phase 16 已完成 commits（`b0924fc` MLIC family + `9cdcb05` SGA generalization）并写入 PR draft，后续只剩实际打开 upstream PR。Phase 14 完成后 MLIC / MLIC+ / MLIC++ / MLICv2 共一个文件 `compressai/models/mlic.py`，共一个 `_BaseMLIC` 模板，共一个 `build_mlic_slice_codec(variant=Literal["mlic","mlic+","mlicpp","mlicv2"])` factory。

---

## 7. 风险

| 风险 | 严重度 | 缓解 |
|---|---|---|
| v1 / v1+ ckpt 作者未 release | 中 | 联系作者邮件请求；若不可得，PR description 显式标注 "fresh-init only"，zoo `pretrained=True` raise |
| MLICv2 GSC 的训练 phase 二阶段在 CompressAI 标准 `examples/train.py` 框架内表达困难 | 中 | 新加 `examples/train_mlicv2.py` 显式 two-stage；如果 reviewer 反对，把 GSC 改为单 phase 联调（论文也提供这条 ablation）但 RD 会有损失 |
| 给 `MultiContextCheckerboardLatentCodec` 加 `selective_predictor` hook 让 leaf 变胖 | 中 | hook 默认 None；启用时单独 `_apply_selective_compression` helper 在 leaf 内实现；如最终 LoC 涨幅超 30%，pivot 到独立 sibling `SelectiveMultiContextCheckerboardLatentCodec` |
| 2D RoPE 与现有 LocalContext 的 `relative_position_index` 共存复杂 | 低 | Phase 14 采用 application wrapper：RoPE + CR 只包在 LinearGlobalInter/IntraContext 输出外，MLIC++ 既有 `LocalContext` 不改；如后续 reviewer 要求更贴近论文，再单独评估 `MLICv2LocalContext` |
| MLICv2 的 STM transform 与 MLIC++ ResidualBlock 共存导致 layers 子包碎片化 | 低 | 已在 PR-MLICpp Phase 2 实施差异里指出独立的 follow-up PR "ResidualBlock `act=` / `norm=` kwargs refactor"；v2 是该 follow-up 的另一个驱动用例，可合并提案 |
| MLICv2 的 ckpt（一旦 release）convert 跟 v1 / ++ 完全不一样（HGCP / CR / RoPE / STM 新参数）| 低 | convert script 模板化，每个 model 一份 `convert_mlicvX_checkpoint.py`，路径 rename map 独立 |

---

## 8. 完成后

- 移动本 doc 到 `plan/exec-plans/completed/`（在 `pr-mlicpp` PR merge 之后；本 doc 是单 PR scope 的家族设计依据，无独立完成节点）
- 在 `family2-roadmap.md` PR-1 描述里指向本 doc（已就位）+ 更新总 LoC / 工时
- 在 `plan/README.md` 加本 doc 的索引行（已就位）
- 单 PR 完成时 `pr-mlicpp-upstreaming.md` 一并移到 completed
- `channel-slice-codec-redesign.md` §3.4 表格把 MLIC v1 / MLIC+ / MLICv2 列加 `MultiContextCheckerboardLatentCodec + application-layer factory` 列项
- SGA Layer A codec generalization 已完成并归档到 [`sga-codec-generalization.md`](../exec-plans/completed/sga-codec-generalization.md)；Layer B 通用 attach/refine helper 如有真实跨模型需求，再新建独立 follow-up plan
