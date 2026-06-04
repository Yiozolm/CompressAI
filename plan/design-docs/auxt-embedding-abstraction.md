# 通用 AuxT 嵌入抽象设计（基于 `master`）

**调研/设计日期：** 2026-06-04
**对象：** `master` 分支（HEAD `43d47eb`）上 4 个 AuxT host 模型（TCM / SAAF / CMIC / GLIC）+ 共享 helper `compressai/models/_helpers/auxt.py` + 3 个 convert 脚本。
**方法：** 直接读 `master` 真实文件，所有 `file:line` 已核验。
**定位：** 纯设计/计划，**不改代码**（用户约束「仅做计划文档」）。
**背景：** AuxT = Auxiliary Transform（Li et al., ICLR 2025 Spotlight, arXiv 2501.13751）。primitives 的容器化已在 [`dcae-saaf-auxt-containerization.md`](../exec-plans/completed/dcae-saaf-auxt-containerization.md) 落地；本文聚焦其上一层——**「AuxT 如何嵌入 host transform」这层目前三套写法各异**，设计一个跨 4 模型的通用嵌入抽象。

> **治理原则前置（承接 [`model-abstraction-optimization-survey.md`](model-abstraction-optimization-survey.md) §8）**：本库研究/教学定位下，凡「以间接性换去重、且模糊网络结构可读性」的改动默认不做。下文据此把 AuxT 嵌入拆成**两类**：① 与网络结构无关的**基础设施样板**（state_dict 管线、loss 聚合、检查点容错）→ 该统一；② 模型 `forward` 里那张**结构图**（手展开的 stage→+aux walk）→ **保留显式，won't force-unify**。同一原则刚否决了 codec 装配工厂，此处一致适用。

---

## 0. 总体结论

`auxt.py` 已经把 **primitives（OLP/WLS/iWLS）+ `aux_loss` 聚合器** 抽好了（对照组，§1.1）。但「嵌入」这层在 4 个 host 上是**三种形态**：

| Host | 嵌入形态 | aux block | 合并 | state_dict key | walk 位置 |
|---|---|---|---|---|---|
| **TCM** | 侧分支 + **通用 walker**（opt-in `use_auxt`）| `build_wls_branch` 标准梯度 | flat sum at 整数 position | `AuxT_enc.*`（顶层前缀）| `forward_with_auxt`（已抽象）|
| **SAAF** | integral + **手 walk 分段** | `_AdaptiveFrequencyBlock`（freq-attn+OLP，**无 DWT**）| sum + **双线性插值** | 无 wavelet buffer | `_encode`/`_decode` 手写 |
| **CMIC** | integral + **手展开** | `nn.Sequential(WLS…)` **自定义宽度** | flat sum（尺寸天然对齐）| `g_a.AuxT_enc.*`（**嵌套前缀**）| `*Transform.forward` 手展开 |
| **GLIC** | integral + **手展开 + 返回 energies** | `nn.Sequential(WLS…)` 自定义宽度 | flat sum | `g_a.AuxT_enc.*`（嵌套）| `forward_energy` 手展开 |

**真正的抽象 gap 只有一处，且是纯基础设施**：**AuxT state_dict 管线被三套写法重复**——own-model wavelet-buffer 容错谓词有 3 个变体（`is_auxt_wavelet_buffer_key` 前缀锚定 / GLIC `_is_pytorch_wavelets_buffer_key` 子串 / CMIC 内联子串），convert 脚本里 upstream key 处理（drop wavelet buffer、`.OLP.`→`.olp.` normalize）又被 CMIC 用正则重抄一遍。这层与网络结构无关，**该统一**（§3.2，P1）。

**嵌入 walk 本身不强合**：TCM 用通用 walker 是因为它的 `g_a` 是扁平 `Sequential`、整数位置可寻址；SAAF/CMIC/GLIC 的手 walk 是模型私有的 stage 编排（分段/插值/嵌套/返回中间 energy），**这正是它们的结构图**。把它们塞进一个配置化 megawalker = 重蹈 codec 工厂覆辙（§4，won't-do）。

> 「通用 AuxT 嵌入抽象」的正确含义 = **契约（contract）+ 共享 infra**，不是一个吞掉所有 walk 的函数。契约 = 命名约定（`AuxT_enc`/`AuxT_dec`）+ 语义（并行支路在 stage 边界求和）+ 共享 loss 聚合 + 共享 state_dict 容错；walk 保持每模型显式。

---

## 1. 现状：4 个 host 的三种嵌入形态（证据）

### 1.1 已共享的 primitives（对照组，明确不动）

`auxt.py`（HEAD `43d47eb`）已抽好、且被正确复用：

- **`OLP`**（`auxt.py:71`，`loss()` 在 `:90`）——正交线性投影 + 正交正则；`identity_matrix` 以 `persistent=False` 注册（`:84-86`，故永不进 state_dict）。被 SAAF `_AdaptiveFrequencyBlock`、WLS/iWLS 内部、CMIC/GLIC 共用。
- **`WLS`/`iWLS`**（`auxt.py:114`/`:145`）——DWT/IDWT + 每子带可学缩放 + OLP 通道混合；lazy import `compressai.layers.wave` 以免硬依赖 `pytorch_wavelets`。被 TCM（经 builder）、CMIC、GLIC 共用。
- **`aux_loss(model)`**（`auxt.py:303`）——遍历子树聚合所有 `OLP.loss()`，无 OLP 时返回 0-d 零张量。**4 个 host 全部经它聚合**（tcm `:449`、saaf `:735`、cmic `:867`、glic `:381`）——这是嵌入层**唯一已经做对的跨模型统一点**，是设计的范本。

> 结论：primitives 与 loss 聚合是范本，**不要动**。设计只补「嵌入」这层缺的契约与 infra。

### 1.2 Pattern A — TCM：侧分支 + 通用 walker（opt-in，已是半成品抽象）

- opt-in `use_auxt: bool = False`（`tcm.py:223`）；`True` 时 `self.AuxT_enc = build_wls_branch(N, M)` / `build_iwls_branch(N, M)`（`:428-429`，标准 `WLS(3,2N)→WLS(2N,2N)×2→WLS(2N,M)` 梯度，`auxt.py:177`/`:197`）。
- 合并位置由 config 推出：`compute_analysis_aux_positions(config)` / `compute_synthesis_aux_positions(config)`（`tcm.py:425-426`，`auxt.py:261`/`:281`）。
- forward 走**通用 walker** `forward_with_auxt(self.g_a, self.AuxT_enc, positions, x)`（`tcm.py:439-447`，`auxt.py:213`）——逐层走 `g_a`，在 `merge_positions` 处把 `AuxT[i]` 求和进主路；`AuxT_enc is None` 时塌缩成 `transform(x)`，故无条件可调。
- state_dict：`use_auxt=has_auxt_state(state_dict)` 自动探测（`tcm.py:500`，`auxt.py:324`）；allowed_missing 经 `is_auxt_wavelet_buffer_key`（`tcm.py:513`）。
- **这是 4 个里唯一已抽象的 walk**——`auxt.py` 的 walker/positions/builders 全是**为 TCM 量身造的**，但 CMIC/GLIC 没复用（见 §1.4 原因）。

### 1.3 Pattern B — SAAF：integral 手 walk + 插值合并

- AuxT **无条件启用**（非 opt-in）；`aux_enc`/`aux_dec` 是 `nn.ModuleList` of **`_AdaptiveFrequencyBlock`**（`saaf.py:521`/`:572`，`:89` 定义）——freq-attn softmax 加权 + OLP，**不含 DWT**（与 WLS 不同的 aux block 类型）。
- 合并：`_merge_features`（`saaf.py:697`）在 aux 尚未下采样、空间尺寸不匹配时 **`F.interpolate(..., mode="bilinear")`** 对齐再求和——这是 SAAF 独有的需求。
- walk：`_encode`/`_decode`（`:706`/`:722`）**手写**——`g_a[0]` 起步，再遍历 `(m_down1, m_down2, m_down3)` 分段（每段是 layer 列表），段边界后 sum aux。**主 transform 是分段结构，不是扁平 Sequential**，整数位置寻址不适用。
- state_dict：无 wavelet buffer；allowed_missing 是 `relative_position_index`/`global_alpha`（`saaf.py:784-788`），与 AuxT 无关。convert 脚本另需 drop 上游 `.olp.identity_matrix`（`convert_saaf_checkpoint.py:107-114`）。
- **不能用 `forward_with_auxt` 的三重原因**：① 主路分段非扁平；② 合并需插值；③ aux block 非 WLS。

### 1.4 Pattern C — CMIC / GLIC：integral 手展开 + 嵌套 key + 自定义宽度

- AuxT 建为 `nn.Sequential(WLS(3,embed0), WLS(embed0,embed1), …)`（cmic `:545`/`:626`，glic `:116`/`:199`），**宽度是模型私有 `embed_dim0/1/2`**，非 TCM 的 `2N` 标准梯度 → `build_wls_branch` 不适配。
- walk **完全手展开**在 `*Transform.forward`（cmic `:585`/`:662`，glic `forward_energy` `:159`）：`down_i → g_i → aux=AuxT[i](aux); out=out+aux` 逐 stage 写出，尺寸由 DWT 减半与 stride-2 conv 天然对齐（**无插值**）。GLIC 还**额外返回中间 energies**给 graph context——walk 是模型功能的一部分。
- **aux 支路嵌套在 transform 子模块内** → state_dict key 是 `g_a.AuxT_enc.0.dwt.transform.*`（嵌套前缀），故：
  - `has_auxt_state`（锚 `AuxT_enc.` 顶层前缀）**匹配不到** → CMIC/GLIC 不用它（AuxT 恒开，无需探测）。
  - `is_auxt_wavelet_buffer_key`（锚 `AuxT_enc.` 前缀 + 子串）**匹配不到嵌套 key** → GLIC 另写 `_is_pytorch_wavelets_buffer_key`（纯子串 `.dwt.transform.`/`.idwt.inverse.`，glic `:72-83`，docstring 明说「前缀锚定的 helper 匹配不到嵌套，故 GLIC 保留子串版」）；CMIC 在 `from_state_dict` 内联同款子串 backfill（cmic `:929-931`）。
- convert：CMIC 用**正则**重抄 upstream 处理——drop `AuxT_(enc|dec)\.\d+\.dwt\.w_(ll|lh|hl|hh)$` + `(dwt|idwt)\.filters$`（`convert_cmic_checkpoint.py:90-92`），`.OLP.`→`.olp.`（`:98`）——而 TCM 已有 `is_auxt_upstream_wavelet_buffer_key`（`auxt.py:352`）/ `normalize_upstream_auxt_key`（`:368`）做同一件事。

---

## 2. 变异维度：哪些是真实架构差异、哪些是基础设施重复

| 维度 | TCM | SAAF | CMIC | GLIC | 判定 |
|---|---|---|---|---|---|
| 启用方式 | opt-in `use_auxt` | 恒开 | 恒开 | 恒开 | **真实差异**（TCM 兼容无-AuxT ckpt）|
| aux block | WLS/iWLS | `_AdaptiveFrequencyBlock` | WLS/iWLS | WLS/iWLS | **真实差异**（SAAF 用 freq-attn 而非 DWT）|
| aux 宽度 | 标准 `2N` 梯度 | `feature_dims` | `embed_dim*` | 写死 128/192 | **真实差异**（每模型设计选择）|
| 主路结构 | 扁平 `Sequential` | 分段 list | down+stage 对 | down+stage 对 | **真实差异** → 决定 walk 形态 |
| 合并 | flat sum | sum + **插值** | flat sum | flat sum | **真实差异**（SAAF 需对齐）|
| walk 是否带副产物 | 否 | 否 | 否 | **返回 energies** | **真实差异**（GLIC 功能耦合）|
| **state_dict 容错谓词** | `is_auxt_wavelet_buffer_key` | n/a | 内联子串 | `_is_pytorch_wavelets_buffer_key` | **基础设施重复** ← 唯一该统一项 |
| **upstream key 处理** | `*_upstream_*` helpers | `.olp.identity_matrix` strip | 正则重抄 | （走 TCM helper 风格）| **基础设施重复** |
| loss 聚合 | `aux_loss` | `aux_loss` | `aux_loss` | `aux_loss` | **已统一**（对照组）|

**分线结论**：前 6 行全是真实架构差异——把它们抹平需要一个高度配置化的 walker（传 stage 列表 / 合并函数 / 副产物收集器），那等于用一层间接埋掉 4 张不同的结构图。**只有后 2 行（state_dict 管线）是与结构无关的样板，是真正的去重目标。**

---

## 3. 设计：通用 AuxT 嵌入契约（contract + 共享 infra）

分三层，从「该统一」到「保持显式」：

### 3.1 Tier 1 — primitives（保持现状）

OLP / WLS / iWLS / `aux_loss` 已抽好且复用充分（§1.1）。**不动。** 设计只在其上补嵌入层。

### 3.2 Tier 2 — 统一 AuxT state_dict 管线（**真正的 gap，P1**）

把 §1.4 的三套 wavelet-buffer 谓词 + convert 重抄收口到 `auxt.py` 单一真值源。这层纯基础设施，零结构图成本：

1. **own-model 容错谓词改子串锚定，兼容 flat + nested**：
   - 现 `is_auxt_wavelet_buffer_key` 锚 `AuxT_enc.`/`AuxT_dec.` 顶层前缀 → 匹配不到 GLIC/CMIC 的 `g_a.AuxT_enc.*`。
   - 子串 `.dwt.transform.` / `.idwt.inverse.` 本身已是 AuxT-DWT 唯一标识（来自 `pytorch_wavelets`），**前缀锚定是多余约束**。
   - 方向：把判定改成「子串命中即真」（即 GLIC `_is_pytorch_wavelets_buffer_key` 的形态），让 **TCM（flat）/ CMIC / GLIC（nested）共用一个谓词**。删除 GLIC 私有副本与 CMIC 内联子串。
   - 命名建议：保留 `is_auxt_wavelet_buffer_key` 名（向后兼容 convert 脚本 import），放宽其实现；或在 `compressai/layers/wave` 暴露一个 `is_wavelet_kernel_buffer_key`（因为这本质是 wavelet 层的 buffer 约定，非 AuxT 专属），`auxt.py` re-export。**倾向后者**——谓词描述的是 wave 层 buffer，归属 wave 更准。
2. **`has_auxt_state` 加 nested-aware 变体**：现锚顶层前缀，只服务 TCM opt-in 探测。若未来出现「嵌套 key + opt-in」的 host，需要 `any(".AuxT_enc." in k or k.startswith("AuxT_enc."))` 形态。**先记设计点，当前 4 host 无此组合，可不立即改**（避免过度设计）。
3. **convert 端统一**：CMIC convert 的正则（drop upstream wavelet buffer + `.OLP.`→`.olp.`）改调 `is_auxt_upstream_wavelet_buffer_key` + `normalize_upstream_auxt_key`（TCM convert 已用，`auxt.py:352`/`:368`）。让 upstream key 处理只有一处实现。SAAF 的 `.olp.identity_matrix` strip 可加一个 `is_auxt_olp_identity_buffer_key` 小 helper 收口（或并入 upstream 谓词）。
4. **可选 `AuxTStateDictMixin`**：把「`load_state_dict(strict=False)` → 扣掉 allowed_missing（wavelet buffer + 各自的 attn buffer）→ 校验残余」这套 4 模型同骨架的容错流程，做成一个 mixin 方法 `_load_with_auxt_tolerance(state_dict, extra_allowed_missing)`。**注意**：各 host 的 *非*-AuxT allowed_missing 不同（TCM `relative_position_index`；SAAF `+global_alpha`；CMIC backfill 风格而非 strict=False），故 mixin 只统一 AuxT 部分、把模型私有 buffer 作参数传入。**此项 P2**——收益中等、需逐模型核对 strict 语义（CMIC 用 backfill+strict=True，与其余 strict=False 不同），勿一刀切。

### 3.3 Tier 3 — 嵌入 walk：generic walker 作 opt-in，手 walk 作结构图保留

- **`forward_with_auxt` 保留为「扁平 Sequential host」的 opt-in 机制**（TCM 在用；未来主路是扁平 `nn.Sequential` 的新 host 可直接复用）。**可选小幅增强**：给它加一个 `merge_fn` 参数（默认 `lambda a,b: a+b`，传入即支持 SAAF 式插值合并），使「扁平/分段 + 求和/插值」两轴可组合——但**仅作为能力提供，不强推 SAAF 迁移**。
- **CMIC / GLIC / SAAF 的手 walk 保持显式**（§4 won't-do）。它们的 `down_i→g_i→+aux[i]`、分段插值、返回 energies 都是模型私有结构图，显式写出正是教学价值。
- **新 host 接入指引（契约落点）**：主路扁平 → 用 `forward_with_auxt` + `build_*_branch`；主路分段/带副产物/需插值 → 手 walk，但**遵守契约**（§3.4）。

### 3.4 契约约定（「通用抽象」的真正载体，文档化）

通用性体现在**约定**而非单一函数。新 AuxT host 必须遵守、`auxt.py` 模块 docstring 固化：

- **命名**：分析支 `AuxT_enc`、合成支 `AuxT_dec`（嵌套时 `g_a.AuxT_enc`）；OLP 子模块属性名 `olp`（小写，convert 把上游 `.OLP.` normalize 过来）。
- **语义**：AuxT 是与主 transform 并行的支路，在**每个 stage 边界**把 `AuxT[i]` 输出**求和**进主路；支路自身串行推进（`aux = AuxT[i](aux)`）。
- **loss**：必须 `return aux_loss(self)`（共享聚合器），训练循环把它加进 RD 目标。
- **state_dict 容错**：wavelet kernel buffer 视为 allowed-missing（用 Tier 2 统一谓词）；convert 用 Tier 2 统一 upstream helper。
- **opt-in（可选）**：若支持无-AuxT 变体，用 `use_auxt` + `has_auxt_state` 探测（参照 TCM）。

### 3.5 可选 — 参数化 ladder builder（**borderline，倾向不做**）

`build_wls_branch(N, M)` 写死标准 `2N` 梯度；CMIC/GLIC 因自定义宽度而内联 `nn.Sequential(WLS(3,embed0),…)`。可加 `build_wls_branch(widths: Sequence[int])` 表达任意梯度统一三者。**但**：宽度序列是每模型刻意的设计选择，内联 `nn.Sequential` 本身也是可读的结构声明 → 抽 builder 收益薄、略埋结构。**倾向保留内联**，builder 仅服务 TCM 标准梯度。列此项仅为完整性。

---

## 4. 边界：明确不做（won't-do）

| 不做项 | 原因 |
|---|---|
| **把 4 个 walk 强合进配置化 megawalker** | 主路结构（扁平/分段/down-stage 对）、合并（sum/插值）、副产物（GLIC energies）是 4 张真实不同的结构图。配置化 walker 用间接性埋掉它们 = 重蹈 codec 装配工厂覆辙（[survey §2](model-abstraction-optimization-survey.md)）。手 walk 的显式性正是教学/可读价值。|
| **强制 SAAF/CMIC/GLIC 迁到 `forward_with_auxt`** | 同上；walker 留作扁平 host 的 opt-in，不回填既有手 walk。|
| **统一 aux block 类型** | SAAF `_AdaptiveFrequencyBlock`（freq-attn）vs WLS（DWT）是不同算子，非 dup。|
| **抽 aux 宽度 builder（§3.5）覆盖 CMIC/GLIC** | 宽度是设计选择，内联 `nn.Sequential` 即结构声明。|

> **从严原则复用**：能统一的只有「与网络结构无关的基础设施样板」——AuxT state_dict 管线（Tier 2）。凡触及 `forward` 里那张 stage→+aux 结构图的，默认保留显式。

---

## 5. 落地动作清单（优先级）

| 优先级 | 动作 | 证据 | 工作量 | 风险 |
|---|---|---|---|---|
| **P1** | wavelet-buffer 容错谓词统一为子串锚定（兼容 flat+nested），删 GLIC `_is_pytorch_wavelets_buffer_key` 与 CMIC 内联子串，三 host 共用一个（建议落 `layers/wave`，`auxt.py` re-export）| §1.4 / §3.2.1 | 低 | 低（纯 key 判定，state_dict round-trip 回归即可）|
| **P1** | convert 端统一：CMIC convert 正则改调 `is_auxt_upstream_wavelet_buffer_key` + `normalize_upstream_auxt_key`；SAAF `.olp.identity_matrix` strip 收成 helper | §1.4 / §3.2.3 | 低 | 低（examples 非库主体；真实 ckpt smoke 验证）|
| **P2** | `AuxTStateDictMixin._load_with_auxt_tolerance`（仅统一 AuxT 部分，模型私有 buffer 作参数）；先核对 CMIC backfill+strict=True vs 余者 strict=False 的语义分歧 | §3.2.4 | 中 | 中（strict 语义不一，需逐模型判定）|
| **P2** | `auxt.py` 模块 docstring 固化嵌入契约（§3.4 命名/语义/loss/容错/opt-in）；写进 [`lic-model-integration-template.md`](../references/lic-model-integration-template.md) 的 AuxT 接入段 | §3.4 | 低 | 零 |
| **P3** | `forward_with_auxt` 加 `merge_fn` 参数（默认求和，可传插值）作能力提供 | §3.3 | 低 | 低（默认行为不变）|
| **won't-do** | ~~强合 4 个 walk / 回填手 walk 到 walker / 统一 aux block / 宽度 builder 覆盖 CMIC-GLIC~~ | §4 | — | 保留结构图显式性 |

**建议顺序**：P1 两项（纯 infra 去重，零结构成本，先收口 state_dict 管线）→ P2 契约文档化（让新 host 有规可循）→ P2 mixin（需核 strict 分歧）→ P3 walker 增强（按需）。每项独立 commit，沿用「一关注点一 commit」纪律；落地各自单开 PR 分支并与上游风格对齐（尤其 Tier 2 谓词归属 `layers/wave` 涉及上游 API 面）。

---

## 6. 引用（均为 `master` HEAD `43d47eb`）

- `compressai/models/_helpers/auxt.py` — primitives + walker + state_dict helpers：`OLP:71`(`loss:90`)、`WLS:114`/`iWLS:145`、`build_wls_branch:177`/`build_iwls_branch:197`、`forward_with_auxt:213`、`compute_*_aux_positions:261`/`:281`、`aux_loss:303`、`has_auxt_state:324`、`is_auxt_wavelet_buffer_key:337`、`is_auxt_upstream_wavelet_buffer_key:352`、`normalize_upstream_auxt_key:368`
- `compressai/models/tcm.py:223,425-447,481-540` — Pattern A：opt-in `use_auxt` + 通用 walker + `has_auxt_state` 探测 + `is_auxt_wavelet_buffer_key` 容错（唯一已抽象的 walk）
- `compressai/models/saaf.py:89,521-572,697-744,777-789` — Pattern B：`_AdaptiveFrequencyBlock`（freq-attn，无 DWT）+ `_merge_features` 插值 + 手 walk `_encode`/`_decode`；无 wavelet buffer
- `compressai/models/cmic.py:545,626,585-684,919-931` — Pattern C：自定义宽度 `nn.Sequential(WLS…)` + 手展开 walk + 内联子串 backfill（嵌套 key）
- `compressai/models/glic.py:72-83,116-263,397-413` — Pattern C：私有 `_is_pytorch_wavelets_buffer_key`（子串，因嵌套 `g_a.AuxT_enc.*`）+ `forward_energy` 手展开返回 energies
- `examples/convert_{tcm,cmic,saaf}_checkpoint.py` — convert 端：TCM 用 `auxt` upstream helper（`convert_tcm_checkpoint.py:37-39,222-226`）vs CMIC 正则重抄（`convert_cmic_checkpoint.py:90-98`）vs SAAF `.olp.identity_matrix` strip（`convert_saaf_checkpoint.py:107-114`）
- `compressai/layers/wave/__init__.py` — `is_pytorch_wavelets_available`；wavelet 层是 buffer 谓词的更准归属
- `tests/test_dcae_saaf_helpers.py:133-302` — 现有 AuxT 单测（OLP/WLS/iWLS/aux_loss/forward_with_auxt/has_auxt_state/buffer-key）；Tier 2 改动的回归落点
- 同源文档：[`model-abstraction-optimization-survey.md`](model-abstraction-optimization-survey.md)（治理原则 §8）、[`dcae-saaf-auxt-containerization.md`](../exec-plans/completed/dcae-saaf-auxt-containerization.md)（primitives 落地）、[`cca-cross-model-extension.md`](cca-cross-model-extension.md)、[`lic-model-integration-template.md`](../references/lic-model-integration-template.md)（契约写入处）
