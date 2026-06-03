# DeepLatticeUVEQ lattice quantizer 迁移计划

**计划日期**：2026-05-19  
**状态**：Phase 1 diamond/coset LVQ 已实现并通过目标测试  
**当前分支**：`script`  
**目标代码**：`/Users/boyce/Program/Quant/DeepLatticeUVEQ/quantizer.py`  
**建议落点**：`compressai.ops` 的独立量化 primitive，暂不接入 entropy coder symbols 路径  

**核心判断**：`DeepLatticeUVEQ/quantizer.py` 可以迁入 CompressAI，但它不是现有 `noise` / `ste` / `SGA` 这种标量量化模式的简单补充。它依赖 generator matrix、`gamma`、rate `R`、dither、block reshape、归一化和 lattice-domain clamp。结合 LIC 中三篇 lattice quantization 论文，工程上应优先实现 **diamond/coset LVQ** 这种已在 LIC 中验证、接近 scalar quantization 复杂度的路径；`lattice_quantize(...)` / `LatticeQuantizer(nn.Module)` 作为研究型通用接口保留，但不应作为第一优先级接入 bitstream。

---

## Scope 锁定

**做**：

- 新增 lattice quantization utility，保留 DeepLattice 的核心数学语义。
- 支持 `soft` differentiable rounding，对齐原实现的 `sigm()`。
- 支持 `hard` rounding，用于推理或无梯度对照。
- 可选支持 `ste` rounding，使其能和 CompressAI 现有 `quantize_ste` 对齐。
- 支持外部传入 learned `generator_matrix`，方便复现原项目由网络生成 `hex_mat` 的模式。
- 补 unit tests 覆盖 shape、padding、gradient、zero-std、device/dtype、invalid config。

**不做**：

- 不直接修改 `EntropyModel.quantize()` 的 mode 枚举。
- 不改变现有 `noise` / `ste` / `SGA` 的行为。
- 不在第一阶段实现 `compress()` / `decompress()` 的 bitstream symbols。
- 不迁移 DeepLatticeUVEQ 的训练脚本、federated learning 外壳或 `P0_cov` 分析代码。
- 不把 lattice quantizer 自动挂到所有 latent codec；先作为显式实验接口。

---

## 调研结论

### 原实现关键路径

`LatticeQuantization.__call__` 的主要步骤：

1. flatten 输入并按 `lattice_dim` 分块，不足部分 zero-pad。
2. 对每个 block 计算 mean，并用全局 norm/std 做标准化。
3. 在 orthogonal domain 采样 uniform dither，经 `gen_mat` 映射到 lattice domain。
4. 用 `inverse(gen_mat)` 映射回 orthogonal coordinates。
5. 用 `delta = 2 * gamma / (2 ** R + 1)` 做量化步长。
6. 用 soft differentiable round 或 hard round。
7. clamp 到 `[-edge, edge]`，其中 `edge = gamma - delta / 2`。
8. 映射回 lattice domain，去 dither，反标准化，去 padding，恢复原 shape。

### 与 CompressAI 当前量化体系的关系

- `EntropyModel.quantize()` 当前只服务 entropy model 内部的 `noise` / `dequantize` / `symbols`，其中 `symbols` 必须是 integer symbols。
- `compressai.ops.quantize_ste()` 是独立 op，latent codec forward path 在需要时显式调用。
- `SGAQuantizer` 的合理接入方式已经类似：训练/forward proxy 可替换，但真实 `compress/decompress` 仍走 deterministic round。

因此 lattice quantizer 的第一阶段应跟 `quantize_ste` / `SGAQuantizer` 同级，而不是塞进 `EntropyModel.quantize()`。

### LIC lattice quantization 文献调研

调研材料：

- `Zhang_LVQAC_Lattice_Vector_Quantization_Coupled_With_Spatially_Adaptive_Companding_for_CVPR_2023_paper.pdf`
- `NeurIPS-2024-learning-optimal-lattice-vector-quantizers-for-end-to-end-neural-image-compression-Paper-Conference.pdf`
- `Xu_Multirate_Neural_Image_Compression_with_Adaptive_Lattice_Vector_Quantization_CVPR_2025_paper.pdf`
- `Xu_Multirate_Neural_Image_CVPR_2025_supplemental.pdf`

三篇论文形成的技术路线：

| 论文 | 核心量化器 | 训练代理 | Entropy 建模 | 对迁移的启发 |
|---|---|---|---|---|
| CVPR 2023 LVQAC | fixed diamond lattice，两套 scalar codebook + 1-bit branch | soft-to-hard relaxation | 复用现有 scalar entropy model，额外传 codebook selection | 最容易嵌入现有 CompressAI，优先迁 |
| NeurIPS 2024 OLVQ | learned basis matrix `B` + Babai rounding | uniform noise 替代 rounding | 近似正交后，把 multivariate Gaussian 积分分解成一维 mixture 积分 | 学 lattice shape/orientation，收益大但侵入性高 |
| CVPR 2025 Adaptive LVQ | fixed mother lattice `G`，通过 `aG` 或 `AG` 变换 | STE | basis scaling 后调整 likelihood bin width；domain 用 `AQ(A^-1 y)` | 支持 multirate 和 domain adaptation，适合第二阶段 |

关键结论：

- **fixed diamond LVQ 是最稳的第一落点**：CVPR 2023 与 CVPR 2025 都围绕 diamond/coset lattice 展开，量化可分解为两次 scalar quantization + 一次 branch selection，复杂度接近 SQ。
- **adaptive companding 是独立收益来源**：LVQAC 的 A-law spatial adaptive companding 与 LVQ 解耦，适合做成 latent pre/post transform，并单独做 `LVQ only` / `AC only` / `LVQ+AC` ablation。
- **learned basis 必须配 orthogonality constraint**：NeurIPS 2024 的核心不是简单学习一个 matrix，而是用正交约束缩小 Babai rounding 与 true closest lattice point 的差距，避免训练/推理不一致。
- **multirate LVQ 比 domain adaptation 更适合先做**：CVPR 2025 的 `aG` basis scaling 与现有 variable-rate LIC 的 quantization step scaling 非常接近，能先作为可控 bitrate 实验。
- **收益集中在轻量 entropy/context model**：三篇都显示 context model 越强，LVQ 相对收益越小。Factorized、checkerboard、轻量 channel context 更适合先做；复杂 autoregressive 或 Transformer entropy model 上收益会被强 context 吃掉。

对本计划的修正：

- `lattice_quantize(inputs, generator_matrix, ...)` 仍保留为通用研究接口。
- 第一优先级应改为 `DiamondLatticeQuantizer`，而不是任意 learned `generator_matrix`。
- bitstream 方案应优先围绕 diamond/coset branch index 和 coefficient/index coding 设计，而不是直接编码 arbitrary lattice output。

---

## 建议 API

### Recommended first API：diamond/coset LVQ

```python
def diamond_lattice_quantize(
    inputs: Tensor,
    *,
    step: Union[float, Tensor] = 1.0,
    round_mode: str = "ste",
    block_axis: str = "channel",
    return_indexes: bool = False,
) -> Tensor | Tuple[Tensor, Tensor]:
    ...
```

建议语义：

- 对每个 spatial location 的 channel vector 做 LVQ，优先匹配 LVQAC / Adaptive LVQ 的 LIC 用法。
- 使用两个 coset：
  - base integer lattice
  - half-step shifted lattice
- hard path 选择距离输入更近的 coset；training path 可用 STE 或 soft branch relaxation。
- `return_indexes=True` 时返回 branch index，给后续 bitstream 设计预留接口。
- `step` 支持 scalar 或 Tensor，后续可接 CVPR 2025 的 rate adaptation。

### Research API：generic basis LVQ

这个 API 用于复刻 DeepLatticeUVEQ / NeurIPS 2024 风格的 learned basis，不作为第一阶段 bitstream 目标。

### Functional API

```python
def lattice_quantize(
    inputs: Tensor,
    generator_matrix: Tensor,
    *,
    gamma: float,
    rate: int,
    round_mode: str = "soft",
    dither: bool = True,
    normalize: bool = True,
    block_axis: str = "flat",
    eps: float = 1e-12,
) -> Tensor:
    ...
```

参数说明：

- `generator_matrix`：方阵，shape `[dim, dim]`；可为 learned tensor，允许梯度流向 matrix。
- `gamma`：dynamic range，对应原 `args.gamma`。
- `rate`：对应原 `args.R`。
- `round_mode`：
  - `"soft"`：原 DeepLattice differentiable round。
  - `"hard"`：`torch.round`。
  - `"ste"`：`quantize_ste`，作为 CompressAI-friendly 备选。
- `dither`：是否启用 uniform dither。
- `normalize`：是否启用原实现的 mean/std 标准化。
- `block_axis`：
  - 第一阶段只做 `"flat"`，严格复刻原 flatten 行为。
  - 第二阶段再扩展 `"sample"` / `"channel"`，避免 batch 之间混合。

### Module API

```python
class LatticeQuantizer(nn.Module):
    def __init__(
        self,
        generator_matrix: Tensor,
        *,
        gamma: float,
        rate: int,
        round_mode: str = "soft",
        dither: bool = True,
        normalize: bool = True,
        block_axis: str = "flat",
        eps: float = 1e-12,
    ) -> None:
        ...

    def forward(
        self,
        inputs: Tensor,
        generator_matrix: Optional[Tensor] = None,
    ) -> Tensor:
        ...
```

`forward(..., generator_matrix=...)` 用于保留原项目 `hex_mat = model(...)` 后再量化的实验方式。

---

## Revised migration route

结合论文调研后，实施优先级调整为：

1. **Phase A：fixed diamond/coset LVQ**
   - 实现 `DiamondLatticeQuantizer`。
   - 支持 two-coset quantization 和 branch index。
   - 先做 forward/RD proxy，不直接改 entropy model。
2. **Phase B：LVQAC-style spatial adaptive companding**
   - 实现 A-law companding / inverse companding。
   - 用浅层 conv 预测 per-spatial-location scale parameter。
   - 单独做 `AC only` / `LVQ only` / `LVQ+AC` ablation。
3. **Phase C：adaptive rate scaling**
   - 实现 `step=a` 或 basis scaling `aG`。
   - 先对标 CVPR 2025 的 variable-rate LVQ，不做 domain adaptation。
4. **Phase D：generic learned basis**
   - 实现 `LatticeQuantizer(B)`。
   - 增加 orthogonality regularization。
   - 重新设计 coefficient/index-space likelihood。
5. **Phase E：domain adaptation**
   - 固定 network，只学习 invertible linear map `A`。
   - 采用 `AQ(A^-1 y)` 形式保留 mother lattice 的 fast quantization。

---

## Phase 0：基线复现脚本（0.25 d）

- [ ] 在 `temp/` 写一次性 parity script，对比原 `LatticeQuantization` 与计划 API。
- [ ] 固定 seed，输入覆盖：
  - 1D tensor
  - 4D latent tensor
  - 长度不能被 `lattice_dim` 整除的 tensor
  - 常数 tensor
- [ ] 对齐输出：
  - `round_mode="soft"` 对齐原 `our_round=True`
  - `round_mode="hard"` 对齐原 `our_round=False`
- [ ] 记录 max diff，确认差异只来自 `inverse` → `linalg.solve` 或 dtype/device 处理。

完成后删除 `temp/` 临时脚本，或转为正式 test helper。

---

## Phase 1：新增 diamond/coset LVQ quantizer（0.5-0.75 d）

- [x] 在 `compressai/ops/ops.py` 新增 `diamond_lattice_quantize`。
- [x] 新增 `DiamondLatticeQuantizer(nn.Module)`。
- [x] 支持 `round_mode="ste"|"soft"|"hard"`。
- [x] 支持 `step` 为 scalar 或 Tensor。
- [x] 支持返回 branch index，后续用于 coding 侧额外 1-bit-per-vector 信息。
- [x] 默认 `block_axis="channel"`，对 latent 的每个 spatial location 做 channel vector quantization。
- [x] 保持 `compress/decompress` 不变，只用于 forward proxy。
- [x] 在 `compressai/ops/__init__.py` re-export。

---

## Phase 1b：新增 generic low-level lattice quantizer（0.5 d，后置）

- [ ] 在 `compressai/ops/ops.py` 新增：
  - `soft_lattice_round`
  - `lattice_quantize`
  - `LatticeQuantizer`
- [ ] 在 `compressai/ops/__init__.py` re-export。
- [ ] 明确文档标注为 research API，不作为第一阶段 bitstream target。
- [ ] 保持无新增第三方依赖，仅使用 `torch`。
- [ ] 用 `torch.linalg.solve(generator_matrix, x)` 替代 `torch.inverse(generator_matrix) @ x`。
- [ ] 所有随机 dither 使用 `torch.empty_like(...).uniform_(...)`，保证 device/dtype 一致。
- [ ] 对 `std=0` 做 `clamp_min(eps)`，避免常数输入 NaN。
- [ ] 对 `gamma <= 0`、`rate < 0`、非方阵 `generator_matrix` 明确报错。

---

## Phase 2：测试覆盖（0.5 d）

在 `tests/test_ops.py` 增加 `TestDiamondLatticeQuantizer`：

- [x] `test_diamond_lattice_quantize_shape`：输入输出 shape 一致。
- [x] `test_diamond_lattice_quantize_branch_index`：`return_indexes=True` 时 index shape 正确。
- [x] `test_diamond_lattice_quantize_hard_nearest_coset`：hard path 选择最近 coset。
- [x] `test_diamond_lattice_quantize_ste_grad`：STE path 梯度能回到输入。
- [x] `test_diamond_lattice_quantize_step_scaling`：不同 step 改变 quantization density。
- [x] `test_diamond_lattice_quantize_invalid_config`：非法 step / block_axis 报错。
- [x] `test_diamond_lattice_quantize_channel_step`：1D channel-wise step tensor 正确 broadcast。
- [x] `test_diamond_lattice_quantizer_module`：module wrapper 对齐 functional API。

若实施 Phase 1b，再补 `TestLatticeQuantizer`：

- [ ] `test_soft_round_has_grad`：soft round 输出可反传。
- [ ] `test_lattice_quantize_shape`：输入输出 shape 一致。
- [ ] `test_lattice_quantize_padding_removed`：不能整除 block dim 时正确去 padding。
- [ ] `test_lattice_quantize_hard_matches_expected_grid`：`generator_matrix=I`、`dither=False`、`normalize=False` 时对齐标量 grid。
- [ ] `test_lattice_quantize_ste_grad`：`round_mode="ste"` 梯度能回到输入。
- [ ] `test_lattice_quantize_zero_std_finite`：常数输入不产生 NaN/Inf。
- [ ] `test_lattice_quantizer_forward_override_matrix`：module buffer matrix 与 forward override matrix 都可用。
- [ ] `test_lattice_quantize_invalid_config`：非法 matrix / gamma / rate 报错。

目标验证命令：

```bash
.venv/bin/python -m pytest tests/test_ops.py -q
```

当前验证结果（2026-05-20）：

- `.venv/bin/python -m pytest tests/test_ops.py -q`：22 passed，4 个第三方 deprecation warning。
- `.venv/bin/python -m ruff check compressai/ops/ops.py compressai/ops/__init__.py tests/test_ops.py`：All checks passed。

---

## Phase 3：LVQAC-style companding 实验（0.5-1.0 d）

- [ ] 新增 A-law companding / inverse companding utility。
- [ ] 新增浅层 `CompandingMapNet`，输入 latent，输出 spatial scale parameter。
- [ ] 在单个 lightweight model 上做显式 wiring。
- [ ] 跑三组 ablation：
  - Scalar baseline
  - LVQ only
  - AC only
  - LVQ + AC
- [ ] 优先选择 factorized 或 checkerboard context，避免强 autoregressive context 掩盖 LVQ 收益。

---

## Phase 4：可选 latent codec 实验接线（0.5-1.0 d）

第一阶段 utility 合入后，再决定是否做实验接线。建议先只做一个最小实验模型，不泛化全库。

候选接线方式：

1. 在单个 model forward 内显式调用：
   - `y_hat = lattice_quantize(y - means_hat, hex_mat, ...) + means_hat`
   - likelihood 仍由当前 Gaussian conditional 计算。
2. 或在 `GaussianConditionalLatentCodec` 加可选 `quantizer_module: Optional[nn.Module]`：
   - `quantizer_module is None` 时保持现有 `noise` / `ste` / `sga`。
   - 非空时只影响 training/eval forward 的 reconstruction proxy。
   - `compress/decompress` 不变。

优先选择方式 1，原因：

- scope 最小。
- 不影响已有模型构造参数。
- 能快速跑 RD / gradient / speed sanity check。

---

## Phase 5：是否进入 bitstream 路径（后置决策）

只有在 forward/RD 实验证明 lattice quantizer 有明确收益后，再考虑真实 bitstream。

需要回答的问题：

- diamond LVQ 的 branch index 是否单独编码，还是并入 entropy model？
- lattice coefficient/index 是否能稳定转成 entropy coder symbols？
- `generator_matrix` 是否固定并同步到 decoder？
- dither 是随机训练代理还是 deterministic shared dither？
- clamp 后的 overload symbol 如何编码？
- 现有 `GaussianConditional` 的 likelihood 是否仍匹配 lattice quantization 后的离散分布？

若这些问题没有统一答案，不应把 lattice quantizer 接进 `compress()` / `decompress()`。

---

## 风险与缓解

| 风险 | 严重度 | 缓解 |
|---|---:|---|
| 原实现 flatten 全 tensor，batch 间会混合 | 高 | 第一阶段标注为 `block_axis="flat"` 复刻；第二阶段新增 sample/channel block 模式 |
| `std=0` 导致 NaN | 中 | `std.clamp_min(eps)` |
| `torch.inverse` 数值和性能较差 | 中 | 使用 `torch.linalg.solve` |
| dither 随机性导致测试不稳定 | 中 | 测试用 `dither=False` 或固定 seed |
| 直接改 `EntropyModel.quantize()` 破坏 symbols 路径 | 高 | 第一阶段只做 `compressai.ops` utility |
| learned `generator_matrix` 可能奇异 | 中 | 显式报错或让 `torch.linalg.solve` 抛异常；后续可加 condition regularization |
| soft round 不是 STE，梯度性质不同 | 低 | 同时支持 `soft` / `ste`，实验对比 |
| generic learned basis 训练/推理不一致 | 高 | learned basis 阶段必须加入 orthogonality regularization；优先做 diamond/coset fixed lattice |
| 强 context model 掩盖 LVQ 收益 | 中 | 先在 factorized / checkerboard / lightweight context 上验证 |

---

## 验收标准

- [ ] 不改变现有 `noise` / `ste` / `SGA` 行为。
- [ ] `tests/test_ops.py -q` 通过。
- [ ] `DiamondLatticeQuantizer` 能返回 branch index，且 hard path 选择最近 coset。
- [ ] lattice quantizer 在 CPU 上支持 float32 / float64。
- [ ] 常数输入、padding 输入、4D latent 输入均输出 finite tensor。
- [ ] `round_mode="soft"` 能对输入和 learned generator matrix 反传。
- [ ] 文档中明确说明：该工具第一阶段只用于 forward quantization proxy，不保证 bitstream compatibility。

---

## 后续建议

1. 先实现 `DiamondLatticeQuantizer`，不要从 arbitrary `generator_matrix` 起步。
2. 用一个小 hyperprior 或 checkerboard 模型比较 `noise` / `ste` / `SGA` / `diamond-lvq`。
3. 若 diamond LVQ 有收益，再接 LVQAC-style companding。
4. 若 variable-rate 需求明确，再实现 CVPR 2025 的 basis scaling `aG`。
5. Generic learned basis 和 domain adaptation 放到后续独立计划。
