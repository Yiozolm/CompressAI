# 通用 CCA 嵌入抽象设计（基于 `master`）

**调研/设计日期：** 2026-06-04
**对象：** `master` 分支（HEAD `43d47eb`）上的 CCA 实现：`compressai/models/cca.py`（`CCAModel` + 私有 `_CCAAuxEntropyModel`）、`compressai/losses/cca.py`（`CCARateDistortionLoss`）、`examples/convert_cca_checkpoint.py`、`tests/test_models.py` 的 3 个 CCA 测试。
**方法：** 直接读 `master` 真实文件，所有 `file:line` 已核验；并对照 `script` 分支确认形态分叉。
**定位：** 纯设计/计划，**不改代码**（用户约束「仅做计划文档」）。
**回答的具体问题：** 用户提议「设计一个单独的 CCA compression model，然后通过继承来覆盖」——本文给出诚实评估（结论：**不建议继承，现有 `cca_training` opt-in flag 更优**），并附带纠正一份已过时的同源备忘。
**背景：** CCA = Causal Context Adjustment（Han et al., NeurIPS 2024, arXiv 2410.04847）。

> **治理原则前置（承接 [`model-abstraction-optimization-survey.md`](model-abstraction-optimization-survey.md) §8 / [`auxt-embedding-abstraction.md`](auxt-embedding-abstraction.md)）**：本库研究/教学定位下，凡「以间接性换去重、且模糊网络结构可读性」的改动默认不做（显式性 > DRY；重复 ≠ 债务）。本文据此评估「继承」这层**新增的间接性**是否买得起。

---

## 0. 总体结论

**两条与直觉/旧备忘相悖的事实，先摆出来：**

1. **`master` 上 CCA 是单宿主（single-host），不是跨模型插件。** [`cca-cross-model-extension.md`](cca-cross-model-extension.md)（2026-05-08）记载的「CCA 作为可挂到 TCM/STF/WACNN 的插件，`compressai/entropy_models/cca.py::CausalContextAdjustmentEntropyModel`」**是 `script` 分支的形态**（已核验 `script` 仍有 `entropy_models/cca.py` 且 `tcm.py` 仍有 `use_cca`）。**上游迁入时这套被合并掉了**：`master` 上 TCM 没有任何 CCA 路径（`git grep cca master:compressai/models/tcm.py` 零命中），aux 熵模型变成了 `cca.py` 内的**私有 `_CCAAuxEntropyModel`**，只服务 `CCAModel` 一家。

2. **aux 分支已经是 opt-in，且从 state_dict 自动探测。** `CCAModel(cca_training=False)` 只建主栈；`cca_training=True` 多建一个 `self.aux_entropy_model`（`cca.py:553-559`）。`from_state_dict` 用 `any(key.startswith("aux_entropy_model."))` 自动判定开关（`cca.py:660`）——**一个类、一个 `@register_model("cca")`、一个 zoo entry，自配置**。

**对「单独 model + 继承覆盖」提案的结论：不建议。** 现有 `cca_training` 布尔 opt-in 是更优解，理由排在去重收益之上（§2）。核心症结：registry / zoo / `from_state_dict` 都是 **1:1 名字→类、且从 state_dict 自配置**；继承会逼这套「先选类、再加载」，与「先看 key、再自配置」的现有设计正面冲突。

**真正存在的去重candidate只有一处，且与继承无关**：`CCAModel` 主栈 与 `_CCAAuxEntropyModel` aux 栈的 `channel_context`+`y_latent_codec` 构造**近乎孪生**，唯一真实差异是 `support_count`（主栈 `k` 用全部 prior / aux 栈 `max(k-1,0)` skip-most-recent）。这是否该抽成共享 builder——**borderline，倾向保留显式**（§3.2）。

> 「通用 CCA 嵌入抽象」在 `master` 上其实**没有跨模型的「嵌入」可抽**——CCA 只活在一个模型里。本文的产出因此是：① 纠正过时备忘；② 把单宿主 opt-in 形态固化为契约；③ 否决会把它复杂化的继承提案。

---

## 1. 现状（证据）

### 1.1 形态分叉：`script`（插件）vs `master`（单宿主）

| | `script`（fork 主干） | `master`（上游形态，本文对象） |
|---|---|---|
| aux 熵模型归属 | `compressai/entropy_models/cca.py::CausalContextAdjustmentEntropyModel`（独立插件 module）| `compressai/models/cca.py::_CCAAuxEntropyModel`（**cca.py 私有**）|
| TCM `use_cca` | **有**（`script:compressai/models/tcm.py` 命中）| **无**（`master` TCM 零 CCA 痕迹）|
| 宿主数 | 设想多宿主（TCM 已接，拟推广）| **单宿主 `CCAModel`** |
| 开关 | `use_cca: bool` | `cca_training: bool` |

→ [`cca-cross-model-extension.md`](cca-cross-model-extension.md) 描述的是左列。**右列（master）才是本设计的事实基础。** 旧备忘的「推广到更多 channel-slice 模型」愿景在上游被放弃了。

### 1.2 `master` 的真实结构：单模型 + opt-in aux + 自配置

- **`CCAModel(CompressionModel)`**（`cca.py:395`，`@register_model("cca")` `:394`）——standalone autoencoder（对齐上游 `LICAutoencoder`）。
  - `__init__`（`:430`）：建 `g_a/g_s`（NAF transforms）、`h_a/h_mean_s/h_scale_s`、主熵栈 `self.latent_codec = HyperpriorLatentCodec(...)`（`:535`，Family 1 inline ELIC 装配，`y` 叶子是 `ChannelGroupsLatentCodec` + 每 slice `LRPGaussianLatentCodec`）。
  - **opt-in aux**：`self.cca_training`（`:454`）为真时，末尾 `self.aux_entropy_model = _CCAAuxEntropyModel(M, slice_sizes, em_hidden_channels, em_num_layers)`（`:553-559`）。
  - `forward`（`:561`）：主路恒跑；`cca_training` 为真时**额外**从 `z` round-trip 重算 `side_params=cat(means,scales)` 喂给 aux，产出 `result["aux_likelihoods"]={"y_aux","y_cca"}`；否则 `aux_likelihoods=None`（`:569-582`）。
  - `compress`/`decompress`（`:584`/`:589`）**不碰 aux**——**aux 是纯训练期正则信号**，推理期 dormant。
  - `from_state_dict`（`:605`）→ `_infer_config_from_state_dict`（`:617`）从 key 反推全部超参，含 `cca_training = any(key.startswith("aux_entropy_model."))`（`:660`）。
- **`_CCAAuxEntropyModel(nn.Module)`**（`cca.py:282`）——内部自带 `self.y_entropy_bottleneck = EntropyBottleneck(M)` + `self.inner_codec = ChannelGroupsLatentCodec(..., support_filter=skip-most-recent, side_in_context=True)`（`:363-372`）。`forward(y, latent_means, latent_scales)` 返回 `{"y_aux": factorised, "y_cca": gaussian}`（`:374-388`）。
- **loss 契约**：`CCARateDistortionLoss`（`losses/cca.py:51`）要求 `output["aux_likelihoods"]` 含 `y_aux`/`y_cca`，否则 `KeyError`（`:93-101`）——消费的就是 §1.2 的产出。
- **测试**：`test_cca_forward_and_state_dict_round_trip`（`cca_training=False`，`test_models.py:1543`）、`test_cca_training_branch_forward_and_round_trip`（`=True`，`:1615`）、`test_cca_upstream_state_dict_conversion`（`:1664`）——两条 opt-in 分支 + convert 都已有回归。

### 1.3 唯一的真去重 candidate：主栈 / aux 栈孪生构造

`CCAModel.__init__`（主栈，`:500-552`）与 `_CCAAuxEntropyModel.__init__`（aux 栈，`:324-372`）build 的 `channel_context` + `y_latent_codec` **结构同构**，逐字段对比：

| 字段 | 主栈 | aux 栈 | 同/异 |
|---|---|---|---|
| `support_count(k)` | `k`（use-all-prior）| `max(k-1,0)`（skip-most-recent）| **异（唯一真实差异）** |
| `build_mean_scale_head(...)` 调用形 | `slice_ch/support_ch=2M+cum[·]/widths/side_split=M/emit="post"/naf_factory` | 同形（仅 `cum[·]` 的下标用 `support_count`）| 同 |
| `y` 叶子 | `LRPGaussianLatentCodec(make_entropy_transform(...), mean_support_trail_channels, quantizer="ste")` | 同 | 同 |
| `widths` / `naf_factory` / `cumulative` | `(em_hidden,128)` / `_NAFTransform` / `accumulate` | 同 | 同 |
| 外壳 | 直接进 `HyperpriorLatentCodec` 的 `"y"` | 包进 `inner_codec`（多 `support_filter`）+ `y_entropy_bottleneck` | 异（aux 多包一层）|

→ 两段约 30 行 dict-comprehension 几乎重复，**真实变化点只有 `support_count` 一个函数**。这是与「继承」无关的、cca.py **内部**的局部 dedup 题（§3.2 评估）。

---

## 2. 评估用户提案：单独 model + 继承覆盖

### 2.1 提案具体化（继承会长成什么样）

把 `cca_training` 拆成基类 + 子类：

```python
@register_model("cca")
class CCAModel(CompressionModel):
    def __init__(self, ...):        # 去掉 cca_training / aux 分支
        ...                          # 只建主栈
    def forward(self, x):
        ...                          # 不产 aux_likelihoods

@register_model("cca-train")        # ← 被迫起第二个注册名
class CCATrainingModel(CCAModel):
    def __init__(self, ...):
        super().__init__(...)
        self.aux_entropy_model = _CCAAuxEntropyModel(self.M, self.slice_sizes, ...)
    def forward(self, x):
        result = super().forward(x)
        z_out = self.latent_codec.latent_codec["z"](self.latent_codec.h_a(result["y"]))
        side = self.latent_codec.h_s(z_out["y_hat"])
        means, scales = torch.split(side, self.M, dim=1)
        result["aux_likelihoods"] = self.aux_entropy_model(result["y"], means, scales)
        return result
```

### 2.2 评估（对照现有 `cca_training` flag）

| 维度 | `cca_training` flag（现状）| 继承（提案）| 判定 |
|---|---|---|---|
| **registry / zoo** | 1 名 `"cca"`、1 zoo entry，自配置 | 需 **2 个注册名**（或工厂 dispatch）；「该 load 哪个类」成新问题 | flag 胜 |
| **`from_state_dict` round-trip** | 一个类从 `aux_entropy_model.*` key **自探测**开关（`:660`），strict-load 两形态都对 | 基类 `from_state_dict` 看到 aux key 要么 strict-fail、要么**必须 dispatch 到子类**（基类反向依赖子类 = 设计倒挂）| flag 胜（**决定性**）|
| **超参共享** | aux 与主栈共享 `M/slice_sizes/em_*`，末尾一句 `if` 即接上 | 子类 `super().__init__()` 后再 build aux——**与现状那句 `if` 等价、零行节省** | 平 |
| **forward 复杂度** | 内联 `if self.cca_training:` 重算 side_params | 子类 override 同样重算 side_params（还得 `super().forward()` 全跑一遍主路）| 平/略劣 |
| **与库内约定一致性** | 与 TCM `use_auxt`、（历史）TCM `use_cca` 同构——**布尔 opt-in 是本库既定惯例** | 训练 toggle 用类层级表达，是**异类** | flag 胜 |
| **aux 是训练期专属** | flag 在构造期分配、推理期 dormant，语义清晰 | 子类也能表达，但没比 flag 更清晰 | 平 |

### 2.3 结论：不建议继承

**继承在这里是「为一个 opt-in toggle 新增一层类层级」，买不起它的间接性**，且与 registry/zoo/`from_state_dict`「先看 key 再自配置」的现有设计正面冲突（§2.2 决定性行）。`cca_training` 布尔 flag 已经：① 单类自配置；② 与库内 `use_*` 惯例一致；③ 让 checkpoint round-trip 无歧义。**这是「显式 flag > 类层级间接」的一例**，与上次否决 codec 工厂同源。

> 注：唯一会让「独立 model + 组合（非继承）」重新合理的场景，是把 CCA **重新**做成跨模型插件（旧备忘愿景）。但那是 **mixin/组合**（aux 作独立 `nn.Module` 挂到多宿主），不是「继承覆盖单一 standalone model」；且 `master` 已显示该愿景在上游被放弃（TCM `use_cca` 被删）。故此路在 master 现状下不立。

---

## 3. 设计结论

### 3.1 契约：单宿主 opt-in（固化现状，文档化）

CCA 在 `master` 的正确抽象**不是**跨模型嵌入，而是把现有单宿主形态写成可复用契约（写进 `cca.py` docstring + [`lic-model-integration-template.md`](../references/lic-model-integration-template.md)），供未来若有「带训练期 aux 正则的模型」参照：

- **opt-in flag**：训练期 aux 用 `<feat>_training: bool`（或 `use_<feat>`）构造参数，**不拆类**；推理期 dormant。
- **自配置**：`from_state_dict` 从 `aux_entropy_model.*`（或对应命名空间）key 自探测开关——**单类、单 `@register_model`、单 zoo entry**。
- **aux 作私有 `nn.Module`**：命名空间隔离（`aux_entropy_model.*`），`strict=False` / key-prefix 探测保证加/去开关不破主干 ckpt。
- **loss 契约**：aux 产 `output["aux_likelihoods"]`，专用 criterion 消费；主 `forward` 在关时置 `None`。

### 3.2 唯一 borderline dedup：主/aux 孪生栈（倾向保留显式）

§1.3 的两段孪生 `channel_context`+`y_latent_codec` 可抽成 cca.py **私有** helper：

```python
def _build_slice_stack(slice_sizes, M, widths, naf_factory, support_count):
    cum = list(accumulate(slice_sizes, initial=0))
    channel_context = {f"y{k}": build_mean_scale_head(
        slice_ch=slice_sizes[k], support_ch=2*M+cum[support_count(k)],
        widths=widths, side_split=M, emit_mean_support="post",
        support_transform_factory=naf_factory) for k in range(len(slice_sizes))}
    y_codec = {f"y{k}": LRPGaussianLatentCodec(
        lrp_transform=make_entropy_transform(M+cum[support_count(k)]+slice_sizes[k],
        slice_sizes[k], widths=widths),
        mean_support_trail_channels=M+cum[support_count(k)], quantizer="ste")
        for k in range(len(slice_sizes))}
    return channel_context, y_codec
# 主栈: _build_slice_stack(..., support_count=lambda k: k)
# aux 栈: _build_slice_stack(..., support_count=lambda k: max(k-1, 0))
```

**正面**：把唯一真实差异（`support_count`）抬成显式参数，读者一眼看清「主=use-all、aux=skip-most-recent」；净减 ~30 行。
**反面（从严原则）**：这段是 **network structure**，不是 infra 样板；两栈是两张真实不同的子网络结构图。抽 helper = 用一层间接换 DRY，正是默认要警惕的。且 dict-comprehension 本身已是「recipe」形态、可读性尚可。

**裁决：borderline，倾向保留两段显式**（与 §2 同一把尺：显式 > DRY）。**若**要做，须满足「helper 让 `support_count` 差异*更*显眼、且 state_dict key 逐字节不变」两条，作为 cca.py 局部私有重构、单 commit、`test_models.py:1543/1615` round-trip 回归把关——**不上升为跨模型抽象**。本项列为**可选 P3**，非推荐动作。

### 3.3 primitives / 主熵栈（不动）

`build_mean_scale_head`（`channel_context.py:175`）、`make_entropy_transform`、`ChannelGroupsLatentCodec`、`LRPGaussianLatentCodec`、`HyperpriorLatentCodec` 已是充分复用的共享 infra（被 STF/WACNN/TCM/CCA 共用）。CCA 主栈的 inline ELIC 装配是**自文档化结构图**，按 [survey §2](model-abstraction-optimization-survey.md) won't-do（不抽 codec 工厂）。**不动。**

---

## 4. 边界：明确不做（won't-do）

| 不做项 | 原因 |
|---|---|
| **把 `cca_training` 拆成基类 + 子类继承** | §2.3：为 opt-in toggle 新增类层级，与 registry/zoo/`from_state_dict` 自配置冲突；flag 更优。|
| **按旧备忘把 CCA 重做成跨模型插件 / 给 STF/WACNN/MLIC 加 `use_cca`** | `master` 已放弃该愿景（TCM `use_cca` 被删合并为单宿主）。重新铺开 = 逆上游决策、且 RD 收益未经实证（旧备忘 §4 自承不确定）。如确需，另立 product-spec 重新立项，不在本文 scope。|
| **抽 CCA 主熵栈装配工厂** | 同 survey §2，护网络结构可读性。|
| **强行合并主/aux 孪生栈** | §3.2 borderline，默认保留显式；仅在「更显眼 + key 不变」双条件下作可选局部重构。|

---

## 5. 落地动作清单（优先级）

| 优先级 | 动作 | 证据 | 工作量 | 风险 |
|---|---|---|---|---|
| **P1（文档）** | 把单宿主 opt-in 契约（§3.1）写进 `cca.py` 模块 docstring + [`lic-model-integration-template.md`](../references/lic-model-integration-template.md) 的「训练期 aux 正则」段 | §3.1 | 低 | 零 |
| **P2（文档）** | 给 [`cca-cross-model-extension.md`](cca-cross-model-extension.md) 顶部加 staleness banner，指明其描述 `script` 插件形态、`master` 已单宿主，指向本文 | §1.1 | 极低 | 零 |
| **P3（可选）** | cca.py 私有 `_build_slice_stack(support_count)` 收孪生栈——**仅当**让差异更显眼且 key 不变；单 commit + round-trip 回归 | §3.2 | 低 | 低（state_dict 必须逐字节不变）|
| **won't-do** | ~~继承拆类 / 跨模型重铺 CCA / 抽主栈工厂 / 强合孪生栈~~ | §4 | — | 护显式性与上游一致 |

**建议顺序**：P1/P2 文档先行（纠偏 + 固化契约，零风险）→ P3 仅在确认收益时按需做。**P1/P2 不涉代码逻辑改动，纯 docstring/plan**；P3 若做须单开分支与上游对齐。

---

## 6. 引用（均为 `master` HEAD `43d47eb`，另注 `script` 对照）

- `compressai/models/cca.py` — `CCAModel:395`(`@register_model:394`)、`__init__:430`（`cca_training:454`、主熵栈 `HyperpriorLatentCodec:535`、aux 分配 `:553-559`）、`forward:561`（aux 重算 side_params `:569-582`）、`compress:584`/`decompress:589`（不碰 aux）、`from_state_dict:605`、`_infer_config_from_state_dict:617`（`cca_training` 自探测 `:660`）；`_CCAAuxEntropyModel:282`（`support_count:327`/`support_filter:330`、`inner_codec:365`、`forward:374`）
- `compressai/losses/cca.py:51-112` — `CCARateDistortionLoss`，要求 `aux_likelihoods={y_aux,y_cca}`（`:93-101`）
- `compressai/models/_helpers/channel_context.py:175` — `build_mean_scale_head`（主/aux 栈共用的共享 head builder，`emit_mean_support="post"` 语义见其 docstring）
- `examples/convert_cca_checkpoint.py:87,222,411` — `_is_upstream_cca_state_dict` / `convert_upstream_cca_state_dict` / `main`（aux 分支 reroot 到 `aux_entropy_model.inner_codec.*`）
- `tests/test_models.py:1543,1615,1664` — CCA 三测（off / on / convert），P3 改动的回归落点
- `compressai/zoo/image.py:830-847`、`zoo/__init__.py:34,77` — 单 `"cca"` zoo/registry 接线（继承拆类会冲击此处）
- **`script` 对照（旧备忘形态）**：`script:compressai/entropy_models/cca.py`（独立插件）、`script:compressai/models/tcm.py`（含 `use_cca`）——`master` 均已并/删
- 同源文档：[`cca-cross-model-extension.md`](cca-cross-model-extension.md)（**已过时，描述 `script` 插件形态**）、[`auxt-embedding-abstraction.md`](auxt-embedding-abstraction.md)（姊妹篇，同从严原则）、[`model-abstraction-optimization-survey.md`](model-abstraction-optimization-survey.md)（§8 治理原则、§2 codec 工厂 won't-do）、[`lic-model-integration-template.md`](../references/lic-model-integration-template.md)（契约写入处）
