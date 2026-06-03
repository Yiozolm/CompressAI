# CCA 作为跨模型可插拔辅助熵模型

**记录日期：** 2026-05-08
**触发：** `pr-tcm-cca` 分支开发期间，发现 CCA 的辅助熵模型本质上是个独立 `nn.Module`，目前只接到了 TCM 一个模型上（通过 `use_cca=True` 开关），但其设计完全可以推广到更多 channel-slice 风格的 LIC 模型。

---

## 1. 现状

CCA 在本仓两处使用：

1. **`compressai/models/cca.py::CCAModel`** —— 独立 autoencoder（与原 LIC 论文 `LICAutoencoder` 一一对应），整套 NAF transforms + 自带辅助熵模型
2. **`compressai/models/tcm.py::TCM`** 的 `use_cca=True` 路径 —— TCM 主干不变，只在 `__init__` 里加挂一个 `CausalContextAdjustmentEntropyModel`，`forward` 时输出多一个 `aux_likelihoods` dict 供 `CCARateDistortionLoss` 消费

关键是第 2 处：这是个**纯插件式**接入，TCM 主体的 channel-slice 熵模型 / latent codec 都没改，只是在 forward 末尾多调用一次辅助熵模型。

---

## 2. 未来方向

把 `use_cca=True` 的接法推广到所有 **channel-slice 自回归熵模型** 的 LIC 模型，而不是只 TCM 一个享受。

### 适合的候选模型（按落地优先级）

| 模型 | 状态 | 备注 |
|---|---|---|
| `WACNN` | 已合入 `pr-stf-wacnn` | channel-slice + Minnen2020-style EP，结构兼容 |
| `SymmetricalTransFormer` (STF) | 已合入 `pr-stf-wacnn` | 同上 |
| `MLIC++` | fork `script` 已迁入，未上游 | channel-slice + spatial context，可挂 |
| `DCAE` | fork `script` 已迁入，未上游 | channel-slice，可挂 |
| `SAAF` | fork `script` 已迁入，未上游 | channel-slice，可挂 |
| `MambaIC` / `MambaVC` | fork `script` 已迁入 | 同上家族，可挂 |

**不适合**：纯 hyperprior（如 `bmshj2018-hyperprior`）、checkerboard 类（如 `Cheng2020AnchorCheckerboard`）—— 因为 CCA 假设 channel-wise causal 结构。

### 接入 pattern（参考 `compressai/models/tcm.py` 的 `use_cca` 实现）

任何 channel-slice 模型只需 ~30 行：

1. 构造函数加 `use_cca: bool = False` + `cca_hidden_channels` + `cca_num_layers` 参数
2. `__init__` 末尾：
   ```python
   self.cca_aux_entropy_model = (
       CausalContextAdjustmentEntropyModel(
           latent_channels=M, num_slices=num_slices,
           hidden_channels=cca_hidden_channels, num_layers=cca_num_layers,
       )
       if use_cca else None
   )
   ```
3. `forward` 末尾：如果 `cca_aux_entropy_model is not None`，多产出
   `output["aux_likelihoods"] = self.cca_aux_entropy_model(y, latent_means, latent_scales)`
4. `from_state_dict`：通过 `has_cca_aux_state` / `infer_cca_hidden_channels` / `infer_cca_num_layers`
   推断是否启用以及参数（这些 helper 已在 `compressai/entropy_models/cca.py` 提供）
5. `@property def use_cca(self): return self.cca_aux_entropy_model is not None`

无须改动主干 g_a/g_s/h_a/h_s/latent_codec，零侵入。

---

## 3. 设计要点

- **`CausalContextAdjustmentEntropyModel` 已经是 model-agnostic** —— 只依赖 `latent_channels` 和 `num_slices`，不假设宿主模型的 backbone 类型。这是它能作为插件的根本原因
- **`CCARateDistortionLoss`** 已注册为 criterion，宿主模型只要在 `output["aux_likelihoods"]` 里产出 `y_aux` / `y_cca`，loss 就能直接消费
- **state-dict 兼容**：`cca_aux_entropy_model.*` key 是命名空间隔离的，加上 / 去掉 `use_cca` 不影响主干 ckpt 的加载（用 `strict=False` + `has_cca_aux_state` 判断）

---

## 4. 已知不确定性

- **训练协议 / 是否能直接改善 RD**：CCA 论文是在 LICAutoencoder 上验证的，搬到 STF / WACNN / MLIC++ 等其他主干上能否得到论文宣称的 RD 提升，需要实证。本笔记只记**接入 API 是通的**，不承诺效果
- **`hidden_channels` / `num_layers` 默认值**：CCA 论文 + TCM 当前默认是 `224` / `4`，对小模型可能过大。其他主干可能需要按比例缩放
- **与现有 `ChannelSliceLatentCodec` 的耦合度**：目前 `CausalContextAdjustmentEntropyModel.forward` 接受 `(y, latent_means, latent_scales)` 三个 tensor，这跟 `ChannelSliceLatentCodec` 内部计算出的 means/scales 形状一致；任何**等价 layout** 的 codec 都能挂

---

## 5. 不在本笔记 scope

- 怎么改训练脚本去开启 `use_cca` —— 训练侧的工作，等到第一个尝试用 `use_cca` 开关训新模型的 PR 再写
- 是否把 CCA helper 升级为 `compressai.latent_codecs.CausalContextAdjustmentLatentCodec` —— 短期没必要，先复用现有 nn.Module 接法
- AuxT（Li et al., ICLR 2025）的类似插件化 —— 那是另一篇论文，单独 design doc

---

## 6. 引用

- 论文：Han et al., "Causal Context Adjustment Loss for Learned Image Compression", NeurIPS 2024 ([arXiv:2410.04847](https://arxiv.org/abs/2410.04847))
- 上游代码：https://github.com/LabShuHangGU/CCA
- 本仓实现入口：
  - `compressai/entropy_models/cca.py::CausalContextAdjustmentEntropyModel`
  - `compressai/losses/cca.py::CCARateDistortionLoss`
  - 接入示例：`compressai/models/tcm.py` 的 `use_cca` 路径
