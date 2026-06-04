# `compressai/models/` 模型架构抽象优化调研（基于 `master`）

**调研日期：** 2026-06-04
**调研对象：** **`master` 分支**（上游 `InterDigitalInc/CompressAI` 形态，HEAD `43d47eb`）的 `compressai/models/`（19 个模型类 / 18 文件）、`compressai/latent_codecs/`、`compressai/entropy_models/`、`compressai/registry/`、`compressai/zoo/image.py`、`examples/`
**调研方法：** 在 `/tmp` 临时 worktree 检出 `master` 实读（不动当前 `script` 工作树），所有 `file:line` 均指 `master` 真实文件并已核验。
**文档归属：** 本文跟踪在 `script` 分支（`plan/` 仅在此分支维护），但**调研主体是 `master`**。`script` 是 fork 全量主干（含大量未上游模型与 `_bases/` 等 fork 专属抽象），其债务画像与 `master` 不同——前一版本误把 `script` 现状当 `master` 调研，本版已纠正。

> **定位**：纯调研/诊断，不改代码。聚焦 `master` 容器化后**仍存在的模型级抽象空白**——这些不属于 [`layers-abstraction-refactor.md`](layers-abstraction-refactor.md)（算子分层）或 [`channel-slice-codec-redesign.md`](channel-slice-codec-redesign.md)（codec 家族）的范畴。落地动作见 §8。

---

## 0. 总体结论

`master` 已经是相当干净的容器化形态：`models/__init__.py` 只剩 6 个 wildcard（stf/mlic 因 `timm` 可选依赖刻意 deep-import-only，且有注释说明）；`latent_codecs/` 的 `_slice_helpers` / `_checkerboard_helpers` / `_selective_checkerboard` / `_hyper_synthesis` 私有 helper 分解得当、复用充分；entropy_models 精简到 4 文件。**容器化在"codec 内部"完成；"模型如何与 entropy 基类交互"这层仍带系统性样板**：

1. **`base.py` 用硬编码 `isinstance` 枚举 entropy 基类**——`EntropyBottleneckVbr` 逸出，逼 `vbr.py` 自己 override `update()` + 重抄 isinstance（§3）。
2. **`from_state_dict` 无基类脚手架**——18 个模型各写 infer→construct→load，~500 行，`N = state_dict["g_a.0.weight"].size(0)` 同式重复 8+ 次（§4）。
3. **`SimpleVAECompressionModel` 采纳不一致**——7 个模型继承它（白拿 forward/compress/decompress），12 个直接继承 `CompressionModel` 重写三件套（即便它们也组合了 latent_codec）；最刺眼的是 WACNN 用 SimpleVAE、其同文件兄弟 STF 却没用（§5）。
4. **`GaussianMixtureConditional` 是死能力**——`entropy_models` 里定义了，`master` 上**零模型使用**（§6）。

> **明确不做（2026-06-04，maintainer 决策）**：codec 装配外壳（13 处 `HyperpriorLatentCodec(h_a=, h_s=, latent_codec={"z":…, "y":…})` 同构嵌套）**保留逐模型手抄，不抽工厂**。本库是研究/教学向，这段显式装配是**自文档化的网络结构**——读一个模型 `__init__` 就能看清其 entropy stack 怎么搭，不必跳进工厂反推。工厂会把结构藏进一层间接,牺牲 LIC 网络最该保留的可读性与初学者上手成本；显式性 > DRY。详见 §2（原 P1 已撤）。

> 重要边界（纠前一版之误）：§3 的 entropy 耦合是**开闭原则气味，非 active bug**（逸出类已各自自救）。`DualHyperSynthesis` **没有**"被 dcae/saaf 本地重复定义"的问题——两者都正常 import 复用它（`dcae.py:534` / `saaf.py:680`），其本地 `_make_hyper_synthesis()` 构造的是模型私有的内层 conv 栈（DCAE 用 Swin block），**不是** dedup 目标。

---

## 1. 已经做对的部分（对照组，明确不要动）

避免把"好设计"误判成债务——`master` 这些是范本：

- **`latent_codecs/_slice_helpers.py`**：`infer_num_slices` / `infer_max_support_slices` / `make_entropy_transform` 单一真值源，被 stf/tcm/cca/dcae/saaf 复用（`_slice_helpers.py:115`/`:146`）。
- **`latent_codecs/_checkerboard_helpers.py`**：8 个纯函数（embed/unembed/merge/mask…），`checkerboard.py:42` 与 `multi_context_checkerboard.py:41` 共用；`_selective_checkerboard.py:39` 再 import 复用——anchor-parity 边界修复一处生效全局。
- **`models/_helpers/`**：`channel_context.py`（3 模型）、`auxt.py`（4 模型）、`dictionary_context.py`（2 模型）分解合理。`channel_context.py:49` 依赖 `latent_codecs._slice_helpers` 是 models→codecs 正确方向，非循环。
- **`models/__init__.py`**：6 wildcard + stf/mlic deep-import-only（`__init__.py:37-42` 有注释）——比 `script` 的 32 wildcard 干净得多。**本项在 `master` 上不是债务**。
- **各 family 的 forward loop 不可强合**——见 [`channel-slice-codec-redesign.md`](channel-slice-codec-redesign.md) §3.2（9 结构族是真实架构差异）。
- **codec 装配外壳的显式手抄**——13 处 `HyperpriorLatentCodec(h_a=, h_s=, latent_codec={"z":…, "y":…})` 看似可抽工厂，但**刻意保留**：见 §2。

---

## 2. codec 装配外壳逐模型手抄（明确不做 / won't-do）

### 2.1 现状

`master` 上 13 处 `HyperpriorLatentCodec(...)` 装配点（`grep -c` 核验，其中 `dcae.py:302` 是 docstring 示例，真实代码 12 处），每个容器化模型在 `__init__` 里显式写出同一个外壳：

```python
self.latent_codec = HyperpriorLatentCodec(
    h_a=h_a,
    h_s=h_s,                                # 或 DualHyperSynthesis(h_mean_s, h_scale_s)
    latent_codec={
        "z": EntropyBottleneckLatentCodec(
            entropy_bottleneck=EntropyBottleneck(N),
            quantizer="ste",                # DCAE/SAAF/MLIC 为 "noise"
        ),
        "y": <ChannelGroupsLatentCodec | CheckerboardLatentCodec>(...),
    },
)
```

证据：`sensetime.py`（ELIC/Cheng 三处）、`stf.py:335`/`:590`、`tcm.py:399`、`cca.py:535`、`dcae.py:532`、`saaf.py:678`、`glic.py:363`、`cmic.py:851`、`mlic.py:280`——外壳同构，变化点是 `"y"` 叶子（ChannelGroups vs Checkerboard）、`h_s`（裸 vs `DualHyperSynthesis`）、z quantizer（`"ste"` vs `"noise"`）、EntropyBottleneck 通道（`N` vs `hyper_channels`）。

### 2.2 决策：保留手抄，不抽工厂（2026-06-04，maintainer 拍板）

曾考虑抽 `make_hyperprior_latent_codec(...)` 工厂把 12 处收口（净减 ~45 行、state_dict 不变）。**结论是不做**，理由优先级高于去重收益：

- **本库是研究/教学向基础设施**。每个模型 `__init__` 里这段显式装配，本身就是该模型 entropy stack 的**自文档化结构图**——读者读一个模型文件即可看清 `z`/`y` 如何嵌套、用什么 quantizer、hyper 怎么接，无需跳进工厂反推。
- **工厂用一层间接换 DRY，恰好牺牲 LIC 网络最该保留的可读性**与初学者上手成本。这里**显式性 > DRY**。
- 所谓"隐性知识"（新模型作者得照抄现有模型）在教学语境下是**优点**：照着一个看得见全貌的范例搭，比调用一个隐藏细节的工厂更利于理解。
- "改默认值要扫 13 处"并非真痛点——`quantizer`/通道本就是**每模型有意的设计选择**（§2.1 已列出 ste/noise、N/hyper_channels 的真实差异），不存在"应统一的默认值"。

> 教训（自查）：重复 ≠ 债务。判断一段重复该不该消除,要看它在该库语境下承载的**意图表达价值**;对教学/研究库,显式的结构重复往往是特性而非缺陷。

---

## 3. `base.py` 与 entropy 基类的硬编码耦合（P1）

### 3.1 现状

`CompressionModel`（`base.py`）三方法用 `isinstance` 只认 **2 个**类：

| 方法 | 行 | 认的类 |
|---|---|---|
| `load_state_dict` | `base.py:99` / `:108` | `EntropyBottleneck` / `GaussianConditional` |
| `update` | `base.py:138` / `:140` | 同上 |
| `aux_loss` | `base.py:172` | 仅 `EntropyBottleneck` |

`master` 的 entropy 类（核验自 `entropy_models/__init__.py` + 各文件）：
- `entropy_models.py`：`EntropyModel`(`:97`)、`EntropyBottleneck`(`:331`)、`GaussianConditional`(`:572`)、`GaussianMixtureConditional`(`:712`，继承 GaussianConditional)
- `entropy_models_vbr.py`：`EntropyModelVbr`(`:51`)、`EntropyBottleneckVbr`(`:369`)
- `gaussian_conditional_shifted.py`：`GsnConditionalLocScaleShift`(`:76`，继承 GaussianConditional)、`Scaler`

### 3.2 逸出覆盖的类如何自救（已核验，均能 work）

| Entropy 类 | 为何逸出 | 现状 |
|---|---|---|
| **`EntropyBottleneckVbr`**（vbr.py）| 平行 `EntropyModelVbr` 链，`isinstance(EntropyBottleneck)` miss | **vbr.py 自己 override `update()`**(`vbr.py:188` 附近) **+ 自加 `isinstance(…, EntropyBottleneckVbr)`**(`vbr.py:193`)——把 base 逻辑在子类重写。这是耦合代价的实证 |
| **`GsnConditionalLocScaleShift`**（ftic.py）| 子类 `GaussianConditional` → 被 `isinstance(GaussianConditional)` 间接命中 | 能 work（靠继承），FTIC 不需 override |

### 3.3 问题与方向

- **开闭违背**：base 写死类名 → 新增 entropy 基类族（VBR 已发生；未来 GMM/VQ）要么改 `base.py`、要么像 vbr.py 在模型里重写 `update()`。
- **无 entropy registry**：`registry/torch.py` 有 MODELS/CRITERIONS/DATASETS/MODULES/OPTIMIZERS/SCHEDULERS 六个注册表，**唯独没有 entropy 注册表**。
- **方向（最轻、最稳）**：把"可更新/出 aux loss/需 buffer 注册"协议化——base 改查 `hasattr(module, "update")` / `hasattr(module, "loss")` 并让各 entropy 类自报 buffer 名单，而非 `isinstance` 具体类。一处改动让 vbr.py 不再需要 override。registry 化（`@register_entropy_model`）收益主要在可发现性/可插拔后端，**优先级低，等真有第二后端再说**。

> 此项与 [`cca-cross-model-extension.md`](cca-cross-model-extension.md) 同源（都受益于 entropy 不被 base 硬编码枚举）。

---

## 4. `from_state_dict` 无基类脚手架（P2）

### 4.1 规模

- `master` 上 **~24 个 `from_state_dict`**（18 文件，含 google/sensetime/vbr 各多个），约 **500 行**。
- `base.py` **不提供任何 `from_state_dict` 脚手架**；统一骨架是 `infer 超参 → net = cls(...) → net.load_state_dict → return net`，骨架部分逐模型重抄。
- `N = state_dict["g_a.0.weight"].size(0)` 在 google.py(`:148`/`:308`/`:559`)、sensetime.py(`:170`)、stf.py(`:354`)、tinylic.py、vbr.py、waseda.py 等 **8+ 处同式重复**。
- 复杂模型（dcae/saaf/ftic）的 `_infer_*` helper 各自本地定义（如 dcae/saaf 各有 ~62 行 `_infer_config_from_state_dict`，结构相近但落在各自文件）。

### 4.2 方向

1. **轻量（推荐先做）**：把"读 weight 推通道数"这类原子 helper（`infer_channel_count(sd, key, dim=0)`）收进 `models/utils.py`（现仅有 conv/deconv/buffer 工具，无推断 helper），消化 §4.1 的 8+ 处同式。
2. **可选脚手架**：`base.py` 加模板 `from_state_dict`，子类只实现 `_infer_hparams(state_dict) -> dict`；骨架（construct→load→return）由基类提供。**注意**：上游 CompressAI 历史上保持 per-model `from_state_dict`，此项若上游要先与 maintainer 对齐风格。
3. **不要做**：强合 dcae/saaf 的 `_infer_config_from_state_dict`——内含模型私有的 Swin head_dim/window/stage 推断，合并只增间接性。

---

## 5. `SimpleVAECompressionModel` 采纳不一致（P3）

### 5.1 现状（核验自 class 声明）

- **继承 `SimpleVAECompressionModel`（白拿 forward/compress/decompress）= 7 个**：CMIC(`cmic.py:711`)、GLIC(`glic.py:264`)、WACNN(`stf.py:214`)、Cheng2020AnchorCheckerboard / Elic2022Official / Elic2022Chandelier(`sensetime.py:69`/`:177`/`:344`)、TCM(`tcm.py:182`)。
- **直接继承 `CompressionModel`（自写三件套）= 12 个**：CCAModel、DCAE、SAAF、MambaIC、ShiftLIC、TinyLIC、WeConvene、_BaseMLIC、FTIC、SymmetricalTransFormer、FactorizedPrior、ScaleHyperprior。

### 5.2 问题

- **同文件不对称**：`stf.py` 里 WACNN 用 `SimpleVAECompressionModel`、SymmetricalTransFormer(`stf.py:411`) 却直接继承 `CompressionModel` 并重写 forward/compress/decompress——两者都组合 latent_codec，没有理由一个用一个不用。
- 部分"自写三件套"的模型（如那些组合了标准 `HyperpriorLatentCodec` 的）其 forward/compress/decompress 与 `SimpleVAECompressionModel` 的实现实质等价，属可消除的重写。

### 5.3 方向

逐个核对"自写三件套"的 12 个模型：凡 forward/compress/decompress 实质等价于 `SimpleVAECompressionModel`（即纯 `g_a → latent_codec → g_s`、无额外旁路）的，改继承 `SimpleVAECompressionModel` 删重写。**注意**带 `aux_likelihoods`/AuxT/多分支旁路的模型不适用——逐个判定，勿一刀切。STF↔WACNN 不对称是最干净的起点。

---

## 6. `GaussianMixtureConditional` 死能力（P3，先定性）

- `entropy_models.py:712` 定义了 `GaussianMixtureConditional(GaussianConditional)`，但 `master` 上 `grep -rln GaussianMixtureConditional compressai/models/` **零命中**——没有任何模型用它。
- 两种处置：(a) 若它是为未来 GMM 模型（RefBasedAR / ContextFormer / [`flash-gmm-integration-plan.md`](../exec-plans/active/flash-gmm-integration-plan.md) 的 `GaussianMixtureConditionalLatentCodec`）预留的基础设施，应在 docstring/`__init__` 标注"infra-for-future, no current consumer"，避免被误删；(b) 若是迁移残留，考虑移除或挪到 follow-up 分支。**先定性，不擅动**。

---

## 7. 低优先项

- **`examples/convert_*_checkpoint.py`（14 个，P3）**：结构同构（parse_args → load → model 专属 `convert_upstream_*` → main + smoke），无共享 lib，各自重抄 argparse/load/smoke 外壳。转换主体是模型私有 key-rename（不可合并），但 CLI 外壳可抽 `examples/_convert_cli.py`。收益有限（examples 非库主体），列 P3。
- **注册一致性测试（P3，先诊断）**：`master` 有 32 个 `register_model` 装饰器、zoo 另有变体命名（如 `shiftlic-small/middle/large` vs 基名注册）。**两个调研 agent 给出的"X 注册但不在 zoo / Y 在 zoo 未注册"具体清单互相矛盾、未经我逐键核验，故本文不断言具体条目**。建议加 `tests/test_registry_consistency.py` 让不一致**可见**（断言每个 `@register_model` 名字要么有 zoo entry、要么显式标 candidate-only），先诊断后治理。

---

## 8. 优先级与动作清单

| 优先级 | 动作 | 证据 | 工作量 | 风险 |
|---|---|---|---|---|
| **P1** | `base.py` entropy 耦合协议化（`hasattr` 鸭子类型替 isinstance），让 vbr.py 不必 override | §3 | 中 | 中（须全 entropy 类 + state_dict round-trip 回归）|
| **P2** | `from_state_dict` 原子推断 helper 入 `models/utils.py`；可选基类模板 `from_state_dict` | §4 | 低-中 | 低-中 |
| **P3** | 核对 12 个自写三件套模型，等价者改继承 `SimpleVAECompressionModel`（STF↔WACNN 不对称起步）| §5 | 中 | 中（逐模型判定 + forward 回归）|
| **P3** | `GaussianMixtureConditional` 定性：标注 infra-for-future 或移除 | §6 | 极低 | 低 |
| **P3** | 注册一致性测试（先诊断）；convert CLI 外壳抽取 | §7 | 低 | 零-低 |
| **won't-do** | ~~抽 codec 装配工厂~~ | §2 | — | 保留显式手抄以护可读性/教学性 |

**建议顺序**：P3 注册测试（让问题可见，零风险）→ P1 entropy 协议化（需充分回归）→ P2 from_state_dict → P3 余项。每条独立成 exec-plan 子任务、逐条 commit，沿用既有"一模型一 commit"纪律。

> **从严原则**：本库研究/教学定位下，凡"以间接性换去重"且会模糊网络结构可读性的改动（典型即 §2 的 codec 装配工厂），默认不做。§3–§6 之所以保留，是因为它们消除的是**与网络结构无关的基础设施样板**（entropy 基类交互、state_dict 推断、注册一致性），不触及模型 `__init__` 里那张"结构图"。

> **上游适配提示**：这些都针对 `master`（上游形态），落地时建议各自单开 PR 分支、与上游 maintainer 风格对齐（尤其 §4 的 from_state_dict 模板、§5 的继承变更涉及上游约定）。

---

## 9. 引用（均为 `master` HEAD `43d47eb`）

- `compressai/models/base.py:99-172` — 三方法的 entropy isinstance（仅 EntropyBottleneck + GaussianConditional）
- `compressai/models/vbr.py:188,193` — 因 base 不认 `EntropyBottleneckVbr` 而 override `update()` + 自加 isinstance 的实证
- `compressai/entropy_models/entropy_models.py:331,572,712` — EntropyBottleneck / GaussianConditional / **GaussianMixtureConditional（零模型使用）**
- `compressai/entropy_models/{entropy_models_vbr,gaussian_conditional_shifted}.py` — VBR 平行链 + GsnConditionalLocScaleShift（子类继承覆盖）
- `compressai/models/sensetime.py` / `stf.py:337,592` / `tcm.py:401` / `cca.py:537` / `dcae.py:534` / `saaf.py:680` — 13 处 `HyperpriorLatentCodec` 装配外壳
- `compressai/models/google.py:148,308,559` 等 — `N = state_dict["g_a.0.weight"].size(0)` 8+ 处同式
- class 声明对照：`SimpleVAECompressionModel` 7 个（cmic/glic/stf:214/sensetime×3/tcm）vs `CompressionModel` 直继承 12 个（含 stf:411 STF 与 WACNN 的不对称）
- `compressai/latent_codecs/{_slice_helpers,_checkerboard_helpers,_selective_checkerboard,_hyper_synthesis}.py` — 已做对的 helper 分解（对照组）
- `compressai/models/_helpers/{channel_context,auxt,dictionary_context}.py` — 已做对的应用层 head 分解（对照组）
- `compressai/models/__init__.py:30-42` — 6 wildcard + stf/mlic deep-import-only（已注释，非债务）
- `compressai/registry/torch.py:30+` — 6 注册表，无 entropy registry
- 互补文档：[`layers-abstraction-refactor.md`](layers-abstraction-refactor.md)、[`channel-slice-codec-redesign.md`](channel-slice-codec-redesign.md)（§3.2 划清不可合并边界）、[`cca-cross-model-extension.md`](cca-cross-model-extension.md)、[`flash-gmm-integration-plan.md`](../exec-plans/active/flash-gmm-integration-plan.md)（与 §3/§6 同源）
