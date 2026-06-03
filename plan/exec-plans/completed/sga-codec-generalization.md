# SGA codec generalization — 把 `quantizer="sga"` 推广到剩余 latent codec

**计划日期**：2026-05-11
**完成日期**：2026-05-11
**状态**：completed（commit `9cdcb05 feat(latent_codecs): generalize sga quantization`；本地 targeted tests 通过；上游无对应 ELIC checkpoint 可做 ckpt 数值回归，因此不再阻塞完成）
**前置依赖**：[`pr-mlicpp-upstreaming.md`](../active/pr-mlicpp-upstreaming.md) Phase 17 落地的 `compressai.ops.SGAQuantizer` + 已 SGA-aware 的 `EntropyBottleneckLatentCodec` / `MultiContextCheckerboardLatentCodec`
**关联设计**：[`mlic-family-reproduction.md`](../../design-docs/mlic-family-reproduction.md) §4.4

---

## ⭐ Scope

把 `quantizer="sga"` 从当前两个 codec（`EntropyBottleneckLatentCodec` + `MultiContextCheckerboardLatentCodec`）推广到剩余两个需要改实现的量化路径，并通过继承覆盖 LRP 变体。

关键决策：`CheckerboardLatentCodec` 仍在 scope 内，但只增加最小的 `quantizer="ste"|"sga"` hook。原因是 ELIC / Cheng2020 checkerboard 的最终 reconstruction `y_hat` 和 non-anchor context `y_hat` 都是在 checkerboard twopass 内部用 `quantize_ste(y - means) + means` 生成的；child `GaussianConditionalLatentCodec` 主要负责 `_chunk(params)` 和 likelihood。只改 child Gaussian 会让 rate side SGA-aware，但 ELIC 的 reconstruction / context path 仍停在 STE。

| Codec | 当前 quantizer 选项 | SGA 接入后增加 | 主要消费者 |
|---|---|---|---|
| `GaussianConditionalLatentCodec` | `"noise"`, `"ste"` | `"sga"` | Cheng2020、Mean-Scale Hyperprior、TCM/STF/CCA leaf、ELIC y leaf |
| `LRPGaussianLatentCodec` | 继承父类 `"noise"`, `"ste"` | 继承父类 `"sga"`，不单独实现 | Family 1 多个 channel-slice 模型（沿 ELIC pattern 但在 leaf 内做 LRP） |
| `CheckerboardLatentCodec` | 内部固定 `quantize_ste` | 加 `quantizer="ste"\|"sga"` + `_quantize` helper（默认 STE；SGA 仅覆盖 `twopass` / `twopass_faster`） | ELIC、Cheng2020 with checkerboard、其他用 sibling leaf 的模型 |

两个需要改实现的 codec 的改动模板与 Phase 17 已落地的两个 codec 一致，**不需要新设计决策**：

1. ctor 加 `quantizer: str = <现有默认>` + `sga: Optional[SGAQuantizer] = None`
2. 在 `__init__` 校验 `quantizer` 取值合法 + `quantizer="sga"` 时 `sga` 必填
3. `forward` 在 `quantizer == "sga"` 分支：用 `self.sga(y - means) + means` 替代 noise/STE 路径，并调 entropy module 的 `_likelihood`（绕过会自动加 noise/dequantize 的 `forward`）
4. compress/decompress 路径**不动**（真实编码用 round，与 SGA 无关）

`LRPGaussianLatentCodec` 是例外：它的 `__init__` 已把 `**gc_kwargs` 透传给 `GaussianConditionalLatentCodec`，`forward` 也先调 `super().forward()` 再应用 LRP residual。因此父类支持 SGA 后，LRP 子类天然获得 `quantizer="sga"`；本计划只为它补测试，不新增子类分支，避免 likelihood 语义漂移。

---

## Phase 1：`GaussianConditionalLatentCodec` 加 `quantizer="sga"`（~30 LoC）

`compressai/latent_codecs/gaussian_conditional.py:GaussianConditionalLatentCodec`：

- ctor 增加 `sga: Optional[SGAQuantizer] = None` + 校验「`quantizer="sga"` 时 sga 必填」
- 现有 forward：
  ```python
  y_hat, y_likelihoods = self.gaussian_conditional(y, scales, means)
  if self.quantizer == "ste":
      y_hat = quantize_ste(y - means) + means
  ```
- 加 `quantizer="sga"` 分支（仿照 `EntropyBottleneckLatentCodec._likelihood_for_quantized` 模式）：
  ```python
  if self.quantizer == "sga":
      y_hat = self.sga(y - means) + means
      y_likelihoods = self._likelihood_for_quantized(y_hat, scales, means)
  ```
- 新增 helper `_likelihood_for_quantized`：直接调 `self.gaussian_conditional._likelihood(y_hat, scales, means)` + 应用 `likelihood_lower_bound`

**测试**：扩展 `tests/test_sga.py::TestGaussianConditionalLatentCodecSGA`，3 个用例（fallback round / iter set 时 grad 流 / invalid quantizer 报错）

## Phase 2：`LRPGaussianLatentCodec` 继承覆盖验证（~0 LoC 实现，测试 only）

`compressai/latent_codecs/gaussian_conditional.py:LRPGaussianLatentCodec`（继承 GaussianConditional 的 LRP 变体）：

- **不新增实现分支**。当前子类结构已经是：
  1. `_split_ctx_params(ctx_params)` 拆出父类 Gaussian 参数与 LRP mean support
  2. `out = super().forward(y, gaussian_params)`
  3. `out["y_hat"] = self._apply_lrp(mean_support, out["y_hat"])`
- Phase 1 后，`super().forward()` 会返回 SGA-quantized pre-LRP `y_hat`，并在父类内计算 Gaussian likelihood。子类只在返回前追加 LRP residual。
- 这正好沿用 Phase 17 在 `MultiContextCheckerboardLatentCodec` 的 convention：likelihood 估计离散 latent 的概率，LRP 是 reconstruction refine 后处理，likelihood **不包含** LRP residual。
- 不要把 SGA 分支复制进子类；否则容易把 likelihood 误算到 post-LRP `y_hat` 上，并产生两套需要维护的 SGA 逻辑。

**测试**：补 `tests/test_sga.py::TestLRPGaussianLatentCodecSGA`，2 个用例：

- `quantizer="sga"` / iter unset 时，pre-LRP 部分等价 round，最终 `y_hat` 是 round 后再加 LRP residual
- iter set 时，rate term + LRP 后 reconstruction loss 都能对输入 latent 产生有效梯度

## Phase 3：`CheckerboardLatentCodec` 加 `quantizer="sga"`（~40 LoC）

`compressai/latent_codecs/checkerboard.py:CheckerboardLatentCodec`（upstream 的 sibling leaf）：

- 当前内部直接 `quantize_ste(...)`，需要改为 `_quantize` helper（与 Phase 17 给 `MultiContextCheckerboardLatentCodec` 加的 helper 同模式）
- ctor 加 `quantizer: str = "ste"` + `sga: Optional[SGAQuantizer] = None`
- `_quantize(y, means)`：`quantizer="sga"` 时返回 `self.sga(y - means) + means`，否则 `quantize_ste(y - means) + means`
- `twopass` / `twopass_faster` 替换内部直接 quantize 调用为 `self._quantize(y, means)`
- `onepass` 保持原有 noise/dequantize 近似路径；`quantizer="sga"` 明确报错，避免静默忽略 SGA
- 新增 `_likelihood_for_quantized` helper：在 SGA mode 下 likelihood 用 checkerboard 最终 `y_hat` 算，复用 child `GaussianConditionalLatentCodec._likelihood_for_quantized`，绕过内部 noise quant

**测试**：3 个 codec-level 用例（iter unset 时 SGA 与 STE 数值一致 / iter set 时梯度穿过 checkerboard `y_hat` / invalid config 报错）。ELIC / Cheng2020 路径没有上游可用 checkpoint 做 ckpt 数值回归；默认 `quantizer="ste"` 的兼容性由 targeted tests + 既有 latent codec / model tests 覆盖。

**2026-05-11 本地实施记录**：

- `GaussianConditionalLatentCodec` 已加 `quantizer="sga"`、`sga` 参数、`_likelihood_for_quantized`
- `LRPGaussianLatentCodec` 未加独立分支，通过父类继承覆盖，并补 inheritance 测试
- `CheckerboardLatentCodec` 已加 `quantizer="ste"|"sga"`、`sga` 参数、`_quantize`、SGA likelihood path；`compress/decompress` 未改
- 提交记录：`9cdcb05 feat(latent_codecs): generalize sga quantization`
- Targeted verification：`.venv/bin/python -m pytest tests/test_sga.py -q` 通过（25 passed, 1 warning）；`.venv/bin/ruff check ...` 通过
- `uv run pytest tests/test_sga.py -q` 在本机因 Python 3.13 + aarch64 macOS 触发 `scipy==1.13.1` Fortran 源码编译失败，未进入测试执行

---

## Layer B（generic refine helper，本计划之外）

如果想让任何 LIC 模型一行接 SGA refine，需要：

1. `compressai.ops.attach_sga(model, sga)` —— 扫 `model.modules()`，把所有 5 种 SGA-aware codec 的 `quantizer / sga` 切到 SGA 模式
2. 通用 `compressai.ops.refine(model, x, sga, total_iter, lr)` —— 尝试调 `model.refine_extract / refine_forward`（如 model 实现了），否则 fallback 通用接力路径（要求 model 有 `g_a / latent_codec / g_s` ELIC pattern 暴露接口）
3. 推荐 model 作者按需 override `refine_extract / refine_forward`（与 `forward / compress / decompress` 同档），保留扩展空间

**Layer B 不在本计划 scope**：涉及到通用 model API 的设计决策（不同 model 家族 latent 提取路径不同：ScaleHyperprior 用 `h_a(abs(y))`、Cheng2020 用 `h_a(y)`、TCM 有 dual-h_a 等等），需独立调研 + 收集多个 family model 的实际需求再做。建议先把 Layer A 的 2 个实现面 + LRP 继承测试做完，让用户可以手动 wiring SGA refine（已能覆盖 95% 场景），再评估 Layer B 是否值得。

---

## 总时间估算

| Phase | 工时估算 |
|---|---|
| Phase 1 (GaussianConditionalLatentCodec) | 0.2 d |
| Phase 2 (LRPGaussianLatentCodec 继承测试) | 0.05 d |
| Phase 3 (CheckerboardLatentCodec + ELIC 回归) | 0.3 d |
| **Layer A 总计** | **0.55 d**，~+110 LoC |

Layer B（如做）：~0.5 d 设计 + 0.5 d 实现，估 +100 LoC（取决于 model API 调研结果）。

---

## 风险

| 风险 | 严重度 | 缓解 |
|---|---|---|
| `CheckerboardLatentCodec` 改动影响 ELIC 行为 | 中 ✅ | 默认 `quantizer="ste"` 行为零变化；上游无对应 ELIC checkpoint 可做 ckpt round-trip，已用 targeted tests 覆盖 STE fallback、SGA gradient path、invalid config，并保留 compress/decompress 路径不变 |
| `LRPGaussianLatentCodec` 被误加独立 SGA 分支，导致 likelihood 算到 post-LRP y_hat | 低 | 不改子类实现，只测试继承路径；父类负责 pre-LRP likelihood，子类只做 reconstruction refine |
| SGA-aware codec 都用 `_likelihood_for_quantized` helper 的小幅重复 | 低 | 可考虑抽 `compressai.latent_codecs._sga_helpers` 共享，但单 helper ~10 LoC，重复代价不大；如做，留 Phase 4 |

---

## 完成记录

- [x] 已移到 `plan/exec-plans/completed/`
- [x] 更新 `plan/README.md` 索引
- [x] 在 `mlic-family-reproduction.md` §4.4 标记 Layer A codec generalization 已完成
- [ ] PR description 列举：5 个公开 SGA-aware codec 类型的接入面（其中 `LRPGaussianLatentCodec` 继承父类实现）、`CheckerboardLatentCodec` 默认 `quantizer="ste"` 的兼容策略、以及无 upstream ckpt 可做 ELIC checkpoint 回归的事实
