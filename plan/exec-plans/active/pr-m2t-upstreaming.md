# pr-m2t: M2T / Masked Transformer 上游迁入执行计划

**计划日期**：2026-05-15  
**状态**：scope 锁定 + phase checklist（第一阶段只做 forward/rate-estimation/zoo/tests；bitstream 后置）  
**当前分支**：`script`  
**目标模型**：`candidate/Masked-Transformer-For-Image-Compression/`  
**建议 registry / zoo key**：`m2t`  
**论文/方法**：M2T, *Masking Transformers Twice for Faster Decoding*, ICCV 2023  
**上游实现**：`https://github.com/wsxtyrdd/Masked-Transformer-For-Image-Compression.git`（本地 commit `2dc3919`）

**核心判断**：这个候选的 transform backbone 基本是 ELIC-style；主要 novelty 不在 `g_a/g_s`，而在 **masked Transformer latent prior + random-mask training + deterministic coding-order decoding**。因此迁移时不要大规模重构 transform backbone，重点保留 entropy/context model 语义。

---

## ⭐ Scope 锁定

**做**：
- 新增 `compressai/models/m2t.py`：`M2T(CompressionModel)`，`@register_model("m2t")`
- 新增 `compressai/layers/lic/m2t.py`：M2T 使用的 ELIC-style analysis/synthesis transform
- 新增 `compressai/layers/attn/masked_transformer.py`：M2T 专用 dynamic mask Swin/Transformer block（底层尽量复用 `swin_attention` 工具）
- 新增 `compressai/latent_codecs/masked_transformer.py`：masked Transformer latent prior / coding-order likelihood
- 接 zoo：`compressai.zoo.m2t(...)` + `candidate_model_architectures["m2t"]`
- 接 tests：forward smoke、eval likelihood deterministic smoke、state_dict round-trip、zoo factory/pretrained gate
- 记录 candidate 限制：无 root LICENSE、无 released pretrained checkpoint、无真实 ckpt forward diff

**不做**：
- **不**迁 `train.py` / `data/` / `loss/` / `utils.py` 训练外壳（含硬编码路径、wandb、DataParallel、configargparse）
- **不**迁未使用的 `AdaptiveELICSynthesis` / `ConditionalAttentionBlock` / `AdaLN`
- **不**把 `constriction` 升为 hard dependency
- **不**第一阶段实现真实 `compress` / `decompress` bitstream；先明确抛 `NotImplementedError`
- **不**修改现有公共 `RSTB` / `SwinBlock` 行为来适配 M2T，避免影响 TIC/STF/TCM 等已有模型

---

## Phase 0：文档与 TODO 锁定（已完成）

- [x] 阅读候选核心文件：
  - `candidate/Masked-Transformer-For-Image-Compression/net/mit.py`
  - `candidate/Masked-Transformer-For-Image-Compression/net/elic.py`
  - `candidate/Masked-Transformer-For-Image-Compression/layer/transformer.py`
  - `candidate/Masked-Transformer-For-Image-Compression/layer/layer_utils.py`
- [x] 确认上游 repo 状态：无 root LICENSE；README 说明 pretrained “To be released”
- [x] 在 `candidate/TODO.md` 新增 `Masked-Transformer-For-Image-Compression / M2T` 待迁移条目
- [x] 决定第一阶段范围：forward/rate-estimation/zoo/tests；bitstream 后置

---

## Phase 1：抽 ELIC-style transforms（0.5 d）

- [ ] 新增 `compressai/layers/lic/m2t.py`
- [ ] 实现 `M2TResidualUnit`
- [ ] 实现 `M2TELICAnalysisTransform`
  - 默认 channels：`[256, 256, 256, 192]`
  - 每 stage 3 个 residual units
  - 复用 `compressai.layers.GDN`, `AttentionBlock`, `conv`
- [ ] 实现 `M2TELICSynthesisTransform`
  - 默认 channels：`[192, 256, 256, 3]`
  - 复用 `GDN(inverse=True)`, `AttentionBlock`, `deconv`
- [ ] 不保留 candidate docstring 中 `[B,T,C,H,W]` video 叙述；只支持 4D image tensors
- [ ] 在 `compressai/layers/lic/__init__.py` re-export 必要类
- [ ] 写 transform-level smoke：随机输入 → `y` / `x_hat` shape 正确

---

## Phase 2：抽 M2T 专用 masked Transformer block（1.0 d）

- [ ] 新增 `compressai/layers/attn/masked_transformer.py`
- [ ] 复用底层工具：
  - `compressai.layers.attn.swin_attention.WindowAttention`
  - `window_partition`
  - `window_reverse`
  - `pad_to_window_multiple` / mask helper（如适用）
- [ ] 实现 `MaskedTransformerSwinBlock`
  - 支持 runtime `input_resolution`
  - 支持 shifted-window attention mask 动态更新
  - `attn_mask` 使用 `persistent=False` buffer，避免污染 state_dict
- [ ] 保留 `slice_size` 参数入口，但若第一阶段未使用，不暴露成 public API
- [ ] `Mlp` / `DropPath` 从 `timm.layers` 导入，不复制候选实现
- [ ] 在 `compressai/layers/attn/__init__.py` re-export 必要类
- [ ] 单测/局部 smoke：不同 H/W 下 forward shape 正确，mask device 正确（CPU/macOS 可跑）

---

## Phase 3：实现 masked Transformer latent codec（1.5 d）

- [ ] 新增 `compressai/latent_codecs/masked_transformer.py`
- [ ] 实现 `MaskedTransformerLatentCodec(LatentCodec)`
- [ ] 内部组件：
  - `mask_token: nn.Parameter`
  - `embedding_layer: nn.Linear(M, transformer_dim)`
  - `blocks: nn.ModuleList[MaskedTransformerSwinBlock]`
  - `entropy_parameters: nn.Sequential(conv1x1 -> GELU -> conv1x1 -> GELU -> conv1x1)`
  - `gaussian_conditional = GaussianMixtureConditional(K=3, scale_bound=...)`
- [ ] 实现 `forward_random_mask(y)`
  - 对齐 candidate training 语义：random mask，只对 masked positions 计 likelihood，unmasked positions likelihood = 1
  - 修掉 `.get_device()` / `.cuda()` 硬 CUDA 假设
- [ ] 实现 `estimate_likelihood(y_hat, context_mode="qlds")`
  - 按 coding order 分 stage 估计所有 positions 的 likelihood
  - 默认 `coding_steps=12`；测试小 config 可设更小
- [ ] 实现 `forward_with_given_mask(y_hat, mask, slice_size=None)`
  - 输出 CompressAI GMM layout：`scales/means/weights` 为 `(B, K*M, H, W)`
  - candidate layout `(K,B,C,H,W)` 只在临时 parity helper 中使用
- [ ] 实现 coding order helper：
  - `qlds`
  - `checkerboard2`
  - `checkerboard4`
  - `quincunx`
  - optional tensor order
- [ ] 修正/规避 candidate 已知问题：
  - Laplace fallback line-break bug（candidate 实际未加上 `+0.001*Laplace`）
  - minmax 只看正最大值的问题（bitstream 后置，先记录）
  - `torch.tensor(symbols)` 导致 device 丢失的问题（bitstream 后置，先记录）
  - `print()` debug 输出不迁
- [ ] `compress` / `decompress` 第一阶段显式 `raise NotImplementedError`
- [ ] 在 `compressai/latent_codecs/__init__.py` re-export codec

---

## Phase 4：实现 `compressai/models/m2t.py`（1.0 d）

- [ ] 新增 `compressai/models/m2t.py`
- [ ] `@register_model("m2t")`
- [ ] 构造参数建议：
  - `N: int = 256`
  - `M: int = 192`
  - `transformer_dim: int = 768`
  - `depth: int = 12`
  - `num_heads: int = 8`
  - `window_size: int = 4`
  - `context_mode: str = "qlds"`
  - `coding_steps: int = 12`
  - `eval_forward_mode: str = "coding_order"`
- [ ] `forward(x)`：
  - pad 到 64 multiple（candidate eval 也是 `p=64`）
  - `y = g_a(x_pad)`
  - train mode：`latent_codec.forward_random_mask(y)`
  - eval mode：round/dequantize 后 `latent_codec.estimate_likelihood(...)`
  - `x_hat = g_s(y_hat)` 后 crop 回原图尺寸
  - 返回 `{"x_hat": x_hat, "likelihoods": {"y": y_likelihoods}}`
- [ ] `inference(x, context_mode=None)`：显式保留 candidate 同名语义，方便后续 parity
- [ ] `compress(x)` / `decompress(strings, shape)`：第一阶段抛 `NotImplementedError`，错误信息说明 bitstream 是 Phase 7+
- [ ] `from_state_dict`：从 state_dict 推断 `N/M/transformer_dim/depth`
- [ ] 在 `compressai/models/__init__.py` re-export `M2T`

---

## Phase 5：zoo 接线（0.25 d）

- [ ] `compressai/zoo/image.py` import `M2T`
- [ ] `candidate_model_architectures["m2t"] = M2T`
- [ ] 新增 factory：
  - `def m2t(pretrained: bool = False, progress: bool = True, **kwargs)`
  - `pretrained=True` 抛 `RuntimeError("Pre-trained model not yet available")`
- [ ] `compressai/zoo/__init__.py` re-export `m2t`
- [ ] `compressai.zoo.image_models["m2t"] = m2t`
- [ ] 不写 pretrained URL；README 只有 “To be released”

---

## Phase 6：tests + parity（1.0 d）

- [ ] `tests/test_models.py::TestModels::test_m2t`
  - small config：`N=32, M=48, transformer_dim=64, depth=2, num_heads=4, coding_steps=4`
  - 输入 `1x3x64x64` 或 `1x3x128x128`
  - train forward：`x_hat` shape、`likelihoods["y"]` shape、finite/positive
  - eval forward：同输入两次 deterministic；无 `z` likelihood
  - `from_state_dict(model.state_dict())` round-trip strict load
  - `compress/decompress` 抛 `NotImplementedError`
- [ ] `tests/test_zoo.py::TestCandidateModels::test_m2t`
  - factory smoke
  - `candidate_model_architectures["m2t"] is M2T`
  - pretrained gate
- [ ] 临时 parity 脚本（放 `temp/m2t_parity_check.py`，完成后删除或转 example）
  - ELIC transform random-weight 对齐
  - coding order 对齐
  - GMM likelihood 对齐（按 candidate 实际运行公式，不加 Laplace fallback）
  - masked prior shape 对齐
- [ ] 回归命令：
  - `.venv/bin/python -m pytest tests/test_models.py -k "m2t" -q`
  - `.venv/bin/python -m pytest tests/test_zoo.py -k "m2t" -q`
  - `.venv/bin/python -m pytest tests/test_models.py -k "contextformer or tic or m2t" -q`

---

## Phase 7：bitstream 设计 spike（后置，0.5-1.0 d）

- [ ] 明确不在第一阶段阻塞 PR
- [ ] 方案 A：使用现有 entropy coder 实现 per-step masked subset CDF
  - [ ] 每个 coding step gather `encoding_locations`
  - [ ] 从 `(B,K*M,H,W)` gather 当前 step 的 symbols / scales / means / weights
  - [ ] 构造 per-symbol CDF，调用现有 `_encoder.encode_with_indexes`
  - [ ] decode 时逐 step 写回 `latent_hat`
- [ ] 方案 B：`constriction` 做 strict optional extra
  - [ ] `pyproject.toml` optional group：`m2t = ["constriction"]`
  - [ ] `compress/decompress` lazy import，缺失时报清晰错误
- [ ] 推荐先 spike 方案 A；除非现有 entropy coder 不适合 arbitrary subset CDF，否则不引入 `constriction`
- [ ] 完成后再把 `candidate/TODO.md` 状态从 `implemented-forward-and-zoo` 升级到 `implemented-and-tested`

---

## 关键风险

| 风险 | 严重度 | 缓解 |
|---|---|---|
| 上游无 root LICENSE | 中 | PR/TODO 明确记录；若要 upstream contribution，需维护者确认可接受 |
| 无公开 pretrained checkpoint | 中 | 第一阶段只做 random-weight/module parity；后续若上游 release 权重，再补 `examples/convert_m2t_checkpoint.py` |
| training/eval forward 语义不同 | 高 | class docstring 明确：train 用 random-mask objective，eval 默认 deterministic coding-order likelihood |
| eval likelihood 较慢 | 中 | CI 使用 tiny config + 小 `coding_steps`；默认大模型只做 factory smoke |
| M2T Swin block 与现有 Swin/RSTB 接口不同 | 中 | 新增 M2T 私有/dedicated block，不改公共 Swin 行为 |
| bitstream 依赖 `constriction` | 中 | 第一阶段不接 bitstream；Phase 7 决策现有 entropy coder vs optional extra |
| candidate GMM likelihood 有实现瑕疵 | 低 | 按 candidate 实际运行语义做 parity；在文档中记录 Laplace fallback 未生效 |

---

## 总时间估算

| Phase | 工时 |
|---|---:|
| Phase 1 | 0.5 d |
| Phase 2 | 1.0 d |
| Phase 3 | 1.5 d |
| Phase 4 | 1.0 d |
| Phase 5 | 0.25 d |
| Phase 6 | 1.0 d |
| **第一阶段合计** | **约 5.25 d** |
| Phase 7 bitstream spike | 0.5-1.0 d |
| 完整 bitstream 实现（若做） | 额外 1.5-2.5 d |

---

## 完成后

- [ ] 更新 `candidate/TODO.md`：
  - 第一阶段完成：`Status: implemented-forward-and-zoo`
  - bitstream 完成：`Status: implemented-and-tested`
- [ ] 若生成 PR 草稿，放到 `plan/generated/pr-m2t-draft.md`
- [ ] 若有临时 parity 脚本，按项目约定清理 `temp/`
- [ ] targeted pytest 全过后，移动本文件到 `plan/exec-plans/completed/`
- [ ] 如后续获得 checkpoint，新增 `examples/convert_m2t_checkpoint.py` 并补真实 ckpt smoke / forward diff
