# DPICT + CTC 迁移计划

> 上游 1：`candidate/DPICT/`（**CVPR 2022 Oral**, Lee et al., "DPICT: Deep Progressive Image Compression Using Trit-Planes"），未声明 license（继承 CompressAI Apache 2.0 顶部 header），需 DPICT-Main + 2 个 DPICT-Post 共 3 个 ckpt。**用户负责从 README 的 Google Drive 链接下载到 `candidate/DPICT/checkpoint/{DPICT-Main,DPICT-Post}/`**。
> 上游 2：`candidate/CTC/`（**CVPR 2023**, Jeon et al., "Context-based Trit-Plane Coding for Progressive Image Compression"），MIT license，N=192 单 ckpt（`ctc.pt`，已就位）。
> 与现仓既有迁移基线一致，目标是 `forward` `x_hat`/likelihoods diff = 0.0 + state_dict `strict=True` roundtrip + bitstream roundtrip。

> **执行顺序（已确认）**：先做 **Phase A: DPICT**，再做 **Phase B: CTC**。原因：
> 1. 命名清晰——CTC 的内嵌 baseline 是「DPICT 的简化重写」，命名为 `DPICTSimplified` 与 `DPICTMain` 兄弟并列，需要先有 `DPICTMain` 才有意义；
> 2. 公共底层 `compressai/ops/trit_plane.py` 先在 DPICT 真实 ckpt 上跑通 forward diff = 0.0，CTC 接入时算法基线已经可信；
> 3. Divide_\* slimmable 子包 + `GaussianConditionalQuantizeTensor` 这套基础设施先到位。

---

## 一、上游全景

### 1.1 DPICT（`candidate/DPICT/`，~5000 行）

DPICT 上游是 **vendored compressai 1.x fork + DPICT-Main + DPICT-Post + 独立 codec_DPICT.py**：

- **`compressai/models/models_DPICT_main.py::DPICT_main_net(Cheng2020Anchor)`**（648 行）：
  - `g_a` / `g_s` / `h_a` / `h_s` 全是 `Divide_*` 通道切分版本（`Divide_g_a` / `Divide_g_s` / `Divide_h_a` / `Divide_h_s`），由 `ResidualBlockWithStrideDivide` / `ResidualBlockUpsampleDivide` / `ResidualBlockDivide` / `AttentionBlockDivide` / `conv_divide` / `subpel_conv_divide` 构成。每层接受 `index_channel` 参数，按 `(shared_ch_start, shared_ch_end, specific_ch_starts[i], specific_ch_ends[i])` 在 weight 上做切片再调 `F.conv2d` —— **slimmable-network 风格的 multi-rate 通道切分**。
  - `entropy_bottleneck` 是 `nn.ModuleList`（每个 divide 分支一份 `EntropyBottleneck`）。
  - `entropy_parameters` 是 `GM_entropy`（3×conv1x1 + LeakyReLU），**不接 MaskedConv 上下文**——纯 hyperprior。
  - 自定义 `gaussian_conditional.quantize_dequantize_v2(y, means, scales, quantize_tensor)`：每个像素的量化步长 = `3 ** k`（k 是局部 trit-plane 截断深度），按 RD 排名决定哪些像素用 floor(k) vs ceil(k)。
  - 自定义 `gaussian_conditional.forward_with_quantize_tensor` + `_likelihood_with_quantize_tensor` + `quantize_noise(quantize_tensor)`：likelihood 的 ±0.5 边界改成 ±0.5 \* quantize_tensor。
  - 编解码入口：`compress_to_representation(x)` → `(z_strings, z_shape, y, means_hat, scales_hat)`；`compress_to_bitstream(y, means, scales)` → 调 `codec_DPICT.compress_DPICT(...)` 出 nested `y_strings`；`decompress_to_representation(y_strings_list, z_strings, z_shape)` → 调 `codec_DPICT.decompress_DPICT(...)` 出 `y_hats: List[Tensor]`（progressive 序列）；`decompress_to_image(y_hats)` → 逐个跑 `g_s` 出 `x_hats: List[Tensor]`。
- **`codec_DPICT.py`**（489 行）：trit-plane (mode=3) 渐进熵编解码核心。常量：`opt_pnum=6`、`L=20` cap on max_L、`pnum_btw_trit=48`、`_pnum_part` 表（`[24,24,24,16,8,8,3,1]` 分母 24/24/24/24/24/24/48/48）、`multiplier = -scipy.stats.norm.ppf(1e-9 / 2)`。`D_old`/`D_new` 用完整版 `E[x²] − 2m·E[x] + m²`。额外维护 `tritlevel_tensor_list`（每个 progressive 点上每个像素当前的 trit-level 数），`get_tritlevel_tensor=True` 时返回，用于训练 DPICT-Post。文件名约定：`{idx:03d}_q{maxL-i-1:02d}.bin` 或 `{idx:03d}_q{...}_{point+1:03d}.bin`。
- **`compressai/models/models_DPICT_post.py::DPICT_post_net(Cheng2020Anchor)`**（314 行）：post-processing g_s only。继承 `Cheng2020Anchor` 但只 override `g_s = Divide_g_s(...)`（10 层 ResBlock+Attn+ResBlock+Upsample 链，最后 `out = x + out` 加全局残差）；其他模块继承不用。`forward(x, index_channel=0)` → `{"x_hat": x_hat}`。**与 DPICT-Main 完全独立**：用户先跑 DPICT-Main 出 `x_hats: List[Tensor]`，再把每个 `x_hat` 喂给 DPICT-Post 出 refined `x_hats`。两个 ckpt（`_2.pth.tar`、`_3.pth.tar`）对应「post 网络 1」「post 网络 2」，差别在训练数据（不同 trit-level 区间训出来的）。
- **`compressai/layers/layers_v010.py`**（419 行）：完整 `Divide_*` 套件（`SwitchableGDN2d` / `SwitchableBatchNorm2d` / `subpel_conv_divide` / `conv_divide` / `masked_conv_divide` / `ResidualBlockWithStrideDivide` / `ResidualBlockUpsampleDivide` / `ResidualBlockDivide` / `AttentionBlockDivide` / `ResidualUnit` / `channel_mult`）。`layers_v008.py` / `layers_v009.py` 是历史版本不被使用，可丢。**这套 Divide_\* 在当前仓零复用**——整个 compressai 当前唯一的「slimmable / multi-rate channel」类设施是 `compressai/models/gained.py` 的 `GainedScaleHyperprior` 系，但那是按 *level index* 选 per-channel gain 标量，不是按 channel range 切 weight tensor，机制不同。
- **`compressai/entropy_models/entropy_models.py`**（775 行）：在 stock `EntropyModel` / `EntropyBottleneck` / `GaussianConditional` 之上加 `quantize_noise(quantize_tensor)` / `quantize_dequantize(quantize_tensor)` / `quantize_dequantize_v2(...)` / `_likelihood_with_quantize_tensor(...)` / `forward_with_quantize_tensor(...)`。其中 `quantize_dequantize_v2` 含一段 sigma-aware 截断逻辑（按 `reconstruction_level_limit = 3^ceil(log_3(scale·6.11·2)) / 2` 限制重建值），且强制 `.cuda(device=...)` —— **不是 device-agnostic，迁移时需改**。
- **License**：未声明（README 无 LICENSE 文件，源码顶部用 InterDigital Apache 2.0 header，是 fork 自 CompressAI 1.x 时带过来的）。按现仓既定政策（GLIC/CMIC/MambaIC/MambaVC/FTIC license 缺失但不阻塞）不阻塞。

### 1.2 CTC（`candidate/CTC/`，~1500 行）

`model_CTC` = **CTC 自带的 simplified-DPICT baseline + CTC 增量**，N=192：

- **`models/dpict/dpict.py::model_baseline`**（约 400 行）：继承 `MeanScaleHyperprior`，**这是 CTC 作者自己重写的 DPICT 基线，不是真正的 DPICT-Main**。仅替换 `g_a`/`g_s`/`h_a`/`h_s`/`entropy_parameters` 为 ResBlock+Attention+conv3x3 风格（**全用 compressai 现有 `ResidualBlock(WithStride/Upsample)`/`AttentionBlock`/`conv3x3`/`subpel_conv3x3`，零新层**）。提供 `repr` / `encode_dpict` / `decode_dpict` / `encode_and_save_bitstreams_dpict` / `evaluate_dpict` 一组 progressive 编解码方法。
- **`models/ctc/ctc.py::model_CTC(model_baseline)`**：仅多 6 个子模块（`CRR1/2/3`、`CDR1/2/3`），覆写 `encode_ctc` / `decode_ctc` / `encode_and_save_bitstreams_ctc` / `reconstruct_ctc`，复用 `repr`。
- **`models/ctc/util/`**：`CRR_v0`（4 路 enc + 拼接 + softmax 重估 PMF）、`CDR_v0`（4×ResBlock 残差 refine + 区间 clamp）、`post_processing_crr` 工具函数；自带的 `ResidualBlock`/`conv1x1`/`conv3x3` **与 `compressai.layers.ResidualBlock` 等价**（conv3x3+LReLU+conv3x3+LReLU+1×1 skip+identity），可直接复用，**无需 key 转换**。
- **`models/utils_trit_plane.py`**（438 行）：trit-plane (mode=3) 渐进熵编解码核心；常量 `mode=3` / `opt_pnum=5` / `pnum_btw_trit=48`，**无 `L` cap**。
- **License**：MIT。

### 1.3 DPICT vs CTC 关系一句话总结

CTC（CVPR 2023）是同一作者群（Jeon, Choi, Park, Kim）对 DPICT（CVPR 2022）的后续工作，**复用 trit-plane 渐进编码这条主线**，但：
- 把 DPICT 复杂的 `Divide_*` slimmable 通道切分 + `quantize_dequantize_v2` 扔掉，重写一个简化版 baseline（`model_baseline`，纯 `MeanScaleHyperprior` 风格）；
- 然后在 baseline 之上加 6 个 helper 模块（CRR×3 重估 PMF、CDR×3 refine reconstruction）。
- DPICT-Post（图像域后处理网络）在 CTC 中**已被完全替代**：CTC 的 CDR 是在 *latent 域* refine（输入 y_hat、entropy_params、interval_min/max → refined y_hat），不是在 *image 域* refine（DPICT-Post 是 x_hat → refined x_hat）。

`DPICTSimplified` 与 `DPICTMain` 在 compressai 中是**兄弟类**，不是父子；它们各自有自己的 state_dict 布局：
- `DPICTSimplified`：`MeanScaleHyperprior` 父类，stock 层，单 `EntropyBottleneck`，state_dict 键 `g_a.0.conv1.weight`（Sequential 索引）。
- `DPICTMain`：`Cheng2020Anchor` 父类，`Divide_*` 层带 `index_channel`，`nn.ModuleList` of `EntropyBottleneck`，stock + `GaussianConditionalQuantizeTensor`，state_dict 键 `g_a.res1.conv1.weight`（命名子模块）。

即便 `DPICTMain` 用 `shared_ratio=[0,1], specific_ratios=[1,1]` 让 Divide_\* 退化为 full-channel，**state_dict 键名仍不同**，CTC 的 ckpt 装不进 `DPICTMain`，必须用独立的 `DPICTSimplified` 类。

---

## 二、落点设计（compressai-side）

### Phase A：DPICT（CVPR 2022），约 6 人日

> 公共底层 `compressai/ops/trit_plane.py` 与 slimmable `compressai/layers/divide/` 子包先在这一阶段落地，CTC 复用。

1. **`compressai/ops/trit_plane.py`**（新增，~500 行）
   把 DPICT 的 `codec_DPICT.py` 与 CTC 的 `utils_trit_plane.py` **合并为一份参数化纯函数模块**：
   - 公开常量：`MODE = 3`、`PNUM_BTW_TRIT = 48`、`MULTIPLIER = -scipy.stats.norm.ppf(1e-9 / 2)`。
   - 公开函数：`get_nary_tensor(y, means, scales, *, opt_pnum, max_L_cap=None, with_tritlevel=False)` / `get_empty_nary_tensor(scales, *, max_L_cap=None)` / `make_pmf_table(scales, device, maxL, l_ele)` / `select_sub_interval(...)` / `get_transmission_tensor(...)` / `prepare_tped_scalable(...)` / `tp_entropy_encoding(...)` / `tp_entropy_encoding_scalable(..., *, opt_pnum)` / `tp_entropy_decoding(...)` / `tped(...)` / `tped_last_point(...)`，全部接受 `opt_pnum` 与 `max_L_cap` kwargs。
     - DPICT 调用：`opt_pnum=6, max_L_cap=20, with_tritlevel=True`。
     - CTC 调用：`opt_pnum=5, max_L_cap=None, with_tritlevel=False`。
   - 内部：`_pmf_to_cdf_tensor` / `_standardized_cumulative` / `_pnum_part` / `_get_ans_encoder` / `_get_ans_decoder`。
   - **数学形式统一**：`D_old` / `D_new` 用 DPICT 的完整版 `E[x²] − 2m·E[x] + m²`（与 CTC 的化简版 `E[x²] − m²` 数学等价，因为 `m = E[x]`；保留完整版可读性更好且不影响数值）。
   - 删掉无用的 `_pmf_to_cdf` 旧版（CTC/DPICT 实际都只用 tensor 版）和 `torch.cuda.empty_cache()` 散点。
   - **不抽到 `latent_codecs/`**：trit-plane 流程含 maxL 个外循环 + 状态化 `pmfs_list` 切割 + 多文件 sub-bitstream + `recon_level` 渐进截断，与 `LatentCodec.{forward/compress/decompress}` 三段式接口形状不匹配，硬塞会得到一个 100% pass-through 的空 codec，无收益。

2. **`compressai/layers/divide/`**（新子包，~600 行）
   - `divide_conv.py`：`DivideConv2d` / `DivideMaskedConv2d` / `DivideSubpelConv2d` —— 重写 upstream `conv_divide` / `masked_conv_divide` / `subpel_conv_divide`，每层在 weight tensor 上按 `(shared_ch_range, specific_ch_range)` 切片再调 `F.conv2d`。
   - `divide_blocks.py`：`DivideResidualBlock` / `DivideResidualBlockWithStride` / `DivideResidualBlockUpsample` / `DivideAttentionBlock` / `DivideResidualUnit`。
   - `switchable_gdn.py`：`SwitchableGDN2d`（每个 divide 分支一份 `GDN`，按 `index_channel` 选）。
   - 命名规范：所有类用 `Divide{ResidualBlock,...}` 驼峰大写形式（与 stock `ResidualBlock` 并列），消除上游 `conv_divide` snake_case 函数名风格的混乱。
   - 单测：每个层 forward / state_dict roundtrip / `index_channel` 切换数值正确性。

3. **`compressai/entropy_models/gaussian_conditional_quantize_tensor.py`**（新增，~200 行）
   `GaussianConditionalQuantizeTensor(GaussianConditional)`：扩展 `quantize_noise(qt)` / `quantize_dequantize(qt)` / `quantize_dequantize_v2(qt)` / `forward_with_quantize_tensor(qt)` / `_likelihood_with_quantize_tensor(qt)`。
   - **修上游 `.cuda(device=...)` 强制 CUDA 的问题**，改 `to(device=inputs.device)`，让 macOS / CPU-only 测试机也能跑。
   - 单测：`quantize_tensor=None` 路径必须与父类 `GaussianConditional.forward` 字字节等价（保证向后兼容）；`quantize_tensor=3` 等简单 case 与上游 forward diff = 0.0。

4. **`compressai/models/dpict_main.py`**（新增，~500 行）
   `DPICTMain(Cheng2020Anchor)`：搬 `DPICT_main_net`，使用 Divide_\* 层 + ModuleList `entropy_bottleneck` + `GaussianConditionalQuantizeTensor` + `GM_entropy`（3×conv1x1）。`@register_model("dpict-main")`，`from_state_dict` 从 `g_a.res1.conv1.weight` 推 N（与上游一致），同时按 `entropy_bottleneck.0.quantiles.shape[0]` / `Divide_g_a.res1.conv1.specific_ch_ends` 推 `shared_ratio` / `specific_ratios`。
   - 编解码 API：`compress_to_representation` / `compress_to_bitstream`（调 `compressai.ops.trit_plane.tp_entropy_encoding(opt_pnum=6, max_L_cap=20)`） / `decompress_to_representation` / `decompress_to_image`。
   - **慢路径警告**：`get_RD_ranks` 是 4-nested numpy loop 跑 B×C×H×W 次，O(B·C·H·W) 复杂度且无向量化，单张 256×256 图实测约 30~60s。**先按原样迁，标 `# TODO(perf): vectorize`**——属于训练时辅助函数，eval 才是主战场，可以接受。

5. **`compressai/models/dpict_post.py`**（新增，~150 行）
   `DPICTPost(Cheng2020Anchor)`：仅 override `g_s = Divide_g_s(...)`，`forward(x) -> {"x_hat": x + g_s(x)}`。`@register_model("dpict-post")`，两个 ckpt（`_2.pth.tar` / `_3.pth.tar`）走同一 class 不同权重，由用户在 `from_state_dict` 时选。

6. **`compressai/zoo/__init__.py`** + **`compressai/zoo/image.py`**：注册 `dpict-main` / `dpict-post`，URL 占空。

7. **测试 + CLI**：
   - `tests/test_models.py::TestModels::test_dpict_main` / `::test_dpict_post`：forward + state_dict roundtrip + 小 config bitstream（仅 main）。
   - `tests/test_zoo.py::TestCandidateModels::test_dpict_main` / `::test_dpict_post`：zoo factory smoke。
   - `examples/convert_dpict_checkpoint.py`：分 `--variant {main,post-1,post-2}`，按 ckpt 名落 zoo 入口；剥 `module.` 前缀 + state_dict 1:1 落键 + forward diff=0.0 校验。**依赖用户先把 3 份 ckpt 放到 `candidate/DPICT/checkpoint/{DPICT-Main,DPICT-Post}/`**。
   - `examples/eval_dpict_progressive.py`（demo）：跑 DPICT-Main → DPICT-Post 串接 + 输出 PSNR/MS-SSIM/bpp 曲线，与上游 `eval.py` 同输入对齐。

### Phase B：CTC（CVPR 2023），约 2.75 人日

> 复用 Phase A 已落地的 `compressai/ops/trit_plane.py`（CTC 配置 `opt_pnum=5, max_L_cap=None`）；不依赖 `compressai/layers/divide/` 与 `GaussianConditionalQuantizeTensor`。

1. **`compressai/models/dpict_simplified.py`**（新增，~250 行）
   `DPICTSimplified(MeanScaleHyperprior)` —— **CTC 内嵌的 simplified DPICT baseline**，与 `DPICTMain` 是兄弟类（不是父子）。把 CTC 的 `model_baseline` transform 与 forward/compress/decompress 整体迁过来，删掉 `decode_dpict`/`evaluate_dpict`/`pytorch_msssim` 依赖；保留 `repr`、`encode_dpict`、`encode_and_save_bitstreams_dpict` 作为 progressive API。`@register_model("dpict-simplified")`，`from_state_dict` 从 `g_a.0.conv1.weight` 推 N。class docstring 必须明确写：「这是 CTC (CVPR 2023) 内嵌的 simplified DPICT baseline，**不能加载 `dpict-main` 的 ckpt**；要用 CVPR 2022 DPICT 原始模型请用 `dpict-main`。」

2. **`compressai/layers/lic/ctc.py`**（新增，~150 行）
   - `CRR`（即 `classifier_v003_7_sc`）：4 个并行 `enc_P`/`enc_q`/`enc_y`/`enc_entropy_param` ResBlock 链 + 拼接 + 6×ResBlock + 4×conv1x1 + sigmoid sharpen + softmax，输出 PMF 重估张量。
   - `CDR`（即 `CDR_v000`）：`enc_latent`/`enc_entropy_params`/`enc` 三路 ResBlock 链 + 残差相加 + `clamp(I_min, I_max)`。
   - 顶层只复用 `compressai.layers.ResidualBlock` + `compressai.layers.conv1x1`，**不引新原语**。

3. **`compressai/models/ctc.py`**（新增，~400 行）
   `CTC(DPICTSimplified)`：加 `CRR1/2/3`+`CDR1/2/3` 子模块、覆写 `encode_ctc`/`decode_ctc`/`encode_and_save_bitstreams_ctc`/`reconstruct_ctc`。`@register_model("ctc")`，`from_state_dict` 同样按 N 推导。

4. **`compressai/zoo/__init__.py`** + **`compressai/zoo/image.py`**：注册 `dpict-simplified` / `ctc`，URL 占空。

5. **测试 + CLI**：
   - `tests/test_models.py::TestModels::test_dpict_simplified` / `::test_ctc`：forward + state_dict roundtrip + 小 config in-memory bitstream roundtrip。
   - `tests/test_zoo.py::TestCandidateModels::test_dpict_simplified` / `::test_ctc`：zoo factory smoke。
   - `examples/convert_ctc_checkpoint.py`：剥 `module.` 前缀 + state_dict 1:1 落键 + forward diff=0.0 校验 + 真实 `sample.png` PSNR/bpp smoke（基于已就位的 `candidate/CTC/ctc.pt`）。

---

## 三、关键设计决策

| 项 | 选择 | 理由 |
|---|---|---|
| Phase A vs B 顺序 | **DPICT 先（Phase A），CTC 后（Phase B）** | 用户已确认；公共 trit-plane ops 先在 DPICT 真实 ckpt 上跑通 forward diff = 0.0 后给 CTC 用，算法基线更可信；命名 `DPICTMain` ↔ `DPICTSimplified` 兄弟语义清晰；避免「先 baseline 后 main 再返工」的命名漂移 |
| `DPICTSimplified` vs `DPICTMain` 关系 | **兄弟类**，各自有独立 state_dict 布局 | 上游真实是这样：CTC 的 `model_baseline` 是 CTC 作者对 DPICT 的简化重写，不是 DPICT-Main 的子集；即便 `DPICTMain(shared=[0,1], specific=[1,1])` 让 Divide_\* 退化，state_dict 键名仍然是 `g_a.res1.conv1.weight`（命名子模块）vs `g_a.0.conv1.weight`（Sequential 索引）。强行让 CTC 继承 `DPICTMain` 只会把简单问题搞复杂 |
| `CRR` / `CDR` 命名 | 沿用论文术语 `CRR` / `CDR`（删掉上游 `_v003_7_sc` / `_v000` 版本号后缀） | 类已是「已选定的实现」，版本号是上游开发期产物 |
| `ResidualBlock`（CTC 端） | **直接用 `compressai.layers.ResidualBlock`** | 结构 + 参数命名（`conv1`/`conv2`/`skip` + `leaky_relu`）完全等价，state_dict key 一致，零迁移 |
| Trit-plane ops 模块化 | 一份 `compressai/ops/trit_plane.py`，DPICT（`opt_pnum=6`, `max_L_cap=20`, `with_tritlevel=True`） / CTC（`opt_pnum=5`, 无 cap, no tritlevel） 共用，**通过 kwargs 参数化差异** | 算法核心 99% 同源；`D_old`/`D_new` 用 DPICT 的完整版数学形式（与 CTC 化简版数学等价但可读性更好）；避免两份近 500 行的孪生代码 |
| Bitstream 接口 | DPICT 端：保留上游 `compress_to_representation` / `compress_to_bitstream` / `decompress_to_representation` / `decompress_to_image` 四步接口；CTC 端：保留上游 `encode_and_save_bitstreams_ctc` / `reconstruct_ctc` 多文件 API；**两边都额外**提供 `compress(x) -> {"strings": [...nested...], "shape": z_shape, "maxL": maxL}` 与 `decompress(strings, shape, maxL, recon_level=...)` 的 in-memory shim | compressai 主线 API 与上游 progressive API 不冲突；in-memory shim 让单元测试可跑 bitstream roundtrip 而不写盘 |
| `pytorch_msssim` | 不引入 | 只 CTC 的 `decode_dpict` / DPICT 的 `eval.py` 用，主库 forward/compress/decompress 路径不需要；compressai 已有 `compressai/utils/eval_model/` |
| `torch.autograd.set_detect_anomaly(True)` | 不迁 | `codec.py` / `eval.py` 里的开发期 debug，主库不做 |
| `recon_level` 参数 | CTC：沿用 `int ∈ [1, 160]`；DPICT：由 `len(y_strings_list)` 自然决定，无单独 cap | 与 ckpt 报告的渐进点数对应，破坏会失去与论文 RD 曲线的可比性 |
| `set_entropy_coder('ans')` 强制设置 | 不在 `__init__` 里全局副作用，改为在编解码入口处局部断言 | compressai 主线允许用户全局选 entropy coder，模型类不应静默切换 |
| Divide_\* 层落点 | `compressai/layers/divide/` 子包，命名 `Divide{ResidualBlock,...}` | 现仓 `compressai/layers/{attn,graph,ssm,wave,lic}` 已按主题分子包，新增 `divide/` 一致；不并入 `lic/` 因为 slimmable 是通用机制 |
| `quantize_dequantize_v2` 强制 CUDA | 修为 device-agnostic（`.to(device=inputs.device)`） | 上游 `.cuda(device=...)` 在 macOS / CPU-only 测试机上直接抛 `AssertionError`；现仓 `tests/` 跑在 CPU，必须改 |
| `get_RD_ranks` numpy 7-loop | 先按原样迁，标 `# TODO(perf): vectorize` | 训练时辅助函数，eval 才是主战场；先保数值等价，性能优化后置 |
| `Divide_*` 是否升格为公共 slimmable 工具 | 短期不打通；落 `compressai/layers/divide/` 自治。如未来真有第二个 slimmable 模型再升格 | 当前 `gained.py` 用 per-channel scalar gain（不同机制），直接打通会引入不必要的抽象 |

---

## 四、验证清单

### Phase A（DPICT-Main + DPICT-Post）

> 用户负责把 3 份 ckpt 放到 `candidate/DPICT/checkpoint/{DPICT-Main,DPICT-Post}/`。

1. **DPICT-Main forward 数值等价**：固定 N=192 / `shared_ratio=[0,1]` / `specific_ratios=[1,1]` 随机权重 + 256×256 → 上游 `DPICT_main_net.forward(x, index_channel=0, quantize_parameters=[0,0,0,0])` vs compressai `DPICTMain.forward(...)` diff = 0.0。
2. **DPICT-Main 真实 ckpt + bitstream roundtrip**：`candidate/DPICT/checkpoint/DPICT-Main/000.pth.tar` → load → `compress_to_representation` → `compress_to_bitstream`（trit-plane）→ `decompress_to_representation` → `decompress_to_image` → 还原序列与上游对齐 diff < 1e-5。
3. **DPICT-Post forward 数值等价**：固定 N=192 随机权重 + 256×256 → 上游 `DPICT_post_net.forward(x_hat)` vs compressai `DPICTPost.forward(...)` diff = 0.0；`x + g_s(x)` 残差路径必须保留。
4. **DPICT-Post 真实 ckpt 串接**：DPICT-Main decompress → DPICT-Post（`_2.pth.tar` / `_3.pth.tar` 两份）refine → 与上游 `eval.py` `_postprocessing` 出的 `x_hats_post2` / `x_hats_post3` 在同一输入下 diff < 1e-5。
5. **`tests/test_models.py::TestModels::test_dpict_main` / `::test_dpict_post`**：forward + state_dict roundtrip + 小 config bitstream（仅 main）。
6. **`tests/test_zoo.py::TestCandidateModels::test_dpict_main` / `::test_dpict_post`**：zoo factory smoke。
7. **Divide_\* 层单测**：每个新层 forward / state_dict roundtrip / `index_channel` 切换数值正确性；与上游 `Divide_*` 层在固定 weight + 固定 `(shared,specific)` 配置下 diff = 0.0。
8. **`GaussianConditionalQuantizeTensor` 单测**：`quantize_tensor=None` 路径与父类 `GaussianConditional.forward` 字字节等价；`quantize_tensor=3` 简单 case 与上游 diff = 0.0。

### Phase B（CTC + DPICTSimplified）

1. **Forward 数值等价**：固定 N=192 随机权重，128×128 / 256×256 输入 → 上游 `model_CTC` vs compressai `CTC` 的 `forward` `x_hat`/`y_likelihoods`/`z_likelihoods` diff = 0.0。
2. **State dict roundtrip**：`from_state_dict(net.state_dict()) → load_state_dict(strict=True)` 无 missing/unexpected。
3. **真实 ckpt 加载**：`candidate/CTC/ctc.pt` → 上游 `model_CTC.load_state_dict` → compressai `CTC.load_state_dict(strict=True)`；同输入 forward diff = 0.0。
4. **Bitstream roundtrip**（in-memory shim）：随机 64×64 输入 + N=32 小 config → `compress` → `decompress(recon_level=maxL)` → 还原 `y_hat`、`x_hat` 在 `eval()` 下 diff < 1e-5（trit-plane 量化噪声允许）。
5. **真实 PSNR**：`ctc.pt` + `candidate/CTC/sample.png` → `encode_and_save_bitstreams_ctc` → `reconstruct_ctc(recon_level=160)` → PSNR / bpp 与上游 README 报告 / 上游同输入运行结果对齐。
6. **`tests/test_models.py::TestModels::test_dpict_simplified` / `::test_ctc`**：forward + state_dict roundtrip + 小 config in-memory bitstream roundtrip（`recon_level=maxL` only）。
7. **`tests/test_zoo.py::TestCandidateModels::test_dpict_simplified` / `::test_ctc`**：zoo factory smoke。
8. **Trit-plane ops 一致性回归**：在 Phase A 已用 DPICT 配置（`opt_pnum=6, max_L_cap=20`）跑通的基础上，CTC 配置（`opt_pnum=5, no cap`）应 byte-for-byte 复现 CTC 上游 `utils_trit_plane.py` 的输出（同 scales 张量 → 同 `make_pmf_table` / 同编解码 bytes）。

---

## 五、工作分解（按建议顺序）

### Phase A：DPICT（先做）

| # | 步骤 | 输出 | 估时 |
|---|---|---|---|
| A1 | 抽 `compressai/ops/trit_plane.py`（合并 DPICT+CTC 两份算法 + 参数化 + 内部清理）+ 单测：固定 scales 张量 → `make_pmf_table` 与上游 `codec_DPICT` byte-for-byte 等价（DPICT 配置） | 1 个新模块 + 1 个 op 单测 | 0.5 d |
| A2 | 写 `compressai/layers/divide/`（`DivideConv2d` / `DivideMaskedConv2d` / `DivideSubpelConv2d` / `DivideResidualBlock*` / `DivideAttentionBlock` / `DivideResidualUnit` / `SwitchableGDN2d`） + 单测（每个层 forward/state_dict roundtrip） | 1 个新子包 + 单测 | 2 d |
| A3 | 写 `compressai/entropy_models/gaussian_conditional_quantize_tensor.py`（`GaussianConditionalQuantizeTensor` + 5 个新方法 + device-agnostic 修正） + 单测（`quantize_tensor=None` 路径必须与 `GaussianConditional` 字字节等价） | 1 个新熵模型 + 单测 | 1 d |
| A4 | 写 `compressai/models/dpict_main.py`（`DPICTMain(Cheng2020Anchor)` + 全 Divide_\* 链 + GM_entropy + `compress_to_*` / `decompress_to_*` 接 `compressai.ops.trit_plane(opt_pnum=6, max_L_cap=20)`） + zoo 注册 + tests | 1 个新模型 + 测试 | 1.5 d |
| A5 | 写 `compressai/models/dpict_post.py`（`DPICTPost`，`Divide_g_s` + `forward(x) → x + g_s(x)`） + zoo 注册 + tests | 1 个新模型 + 测试 | 0.5 d |
| A6 | 写 `examples/convert_dpict_checkpoint.py`（`--variant {main,post-1,post-2}`，剥 `module.` + ckpt → compressai 落键 + forward diff=0.0 校验） + `examples/eval_dpict_progressive.py`（demo：DPICT-Main → DPICT-Post 串接 + 输出 PSNR/MS-SSIM/bpp 曲线） + 文档（`docs/source/models.rst`） + 更新 `candidate/TODO.md` 加 `DPICT` 条目 | 2 个 CLI + docs + TODO 更新 | 0.5 d |
| **A 小计** |  |  | **6 人日 ≈ 1.5 人周** |

### Phase B：CTC（Phase A 完成后）

| # | 步骤 | 输出 | 估时 |
|---|---|---|---|
| B1 | 写 `compressai/models/dpict_simplified.py`（`DPICTSimplified(MeanScaleHyperprior)`，CTC 内嵌简化版）+ zoo 注册（`dpict-simplified`） + `tests/test_models.py::test_dpict_simplified` forward/state_dict | 1 个新模型 | 0.5 d |
| B2 | 写 `compressai/layers/lic/ctc.py`（`CRR`/`CDR`） | 1 个新模块 | 0.25 d |
| B3 | 写 `compressai/models/ctc.py`（`CTC(DPICTSimplified)` + `encode_ctc`/`decode_ctc` 搬迁 + in-memory `compress`/`decompress` shim，trit-plane ops 走 `opt_pnum=5, max_L_cap=None`） + zoo 注册（`ctc`） + `tests/test_models.py::test_ctc` + `tests/test_zoo.py` | 1 个新模型 + 测试 | 1.5 d |
| B4 | 写 `examples/convert_ctc_checkpoint.py`（剥 `module.` 前缀 + state_dict 1:1 落键 + forward diff=0.0 校验 + 真实 sample.png PSNR/bpp smoke） + 更新 `candidate/TODO.md` 加 `CTC` 条目 | 1 个 CLI + TODO 更新 | 0.5 d |
| **B 小计** |  |  | **2.75 人日** |

**总计**：Phase A 6 人日 + Phase B 2.75 人日 = **8.75 人日**。Phase A 验完 trit-plane 在 DPICT 真实 ckpt 上数值等价后，Phase B 直接复用同一 `compressai/ops/trit_plane.py`，CTC 端的算法置信度在 day 1 就高。

---

## 六、依赖与前置条件

1. **DPICT ckpt 下载**（Phase A 启动前必须）：用户从 README Google Drive 链接拉取
   - DPICT-Main：`candidate/DPICT/checkpoint/DPICT-Main/000.pth.tar`
   - DPICT-Post-1：`candidate/DPICT/checkpoint/DPICT-Post/000_2.pth.tar`
   - DPICT-Post-2：`candidate/DPICT/checkpoint/DPICT-Post/000_3.pth.tar`

2. **CTC ckpt**：`candidate/CTC/ctc.pt` 已就位，无需额外动作。

3. **新依赖**：无。`scipy.stats.norm` 现仓已直接/间接引入；`Divide_*` 层完全 PyTorch 实现；`GaussianConditionalQuantizeTensor` 复用 stock entropy_coder 接口。

---

## 七、待确认的开放点（剩余）

1. **Progressive `recon_level` 接口形状**：是否值得在 compressai 引入一个 `ProgressiveCompressionModel` mixin 抽象（DPICT-Main、CTC、未来 LIC-HPCM 的 progressive 变体都可能用上），还是先把 `recon_level` 直接挂在各模型 `decompress(...)` 的 kwargs 里？倾向后者（YAGNI），等第三个 progressive 模型出现再抽。
2. **`utils_trit_plane.py` 模块名**：放 `compressai/ops/` 还是 `compressai/entropy_models/`？倾向 `ops/`，因为它是无状态张量+ANS 工具集合，与现仓 `compressai/ops/multiplex.py` 同级；`entropy_models/` 当前都是 `nn.Module` 子类。
3. **`get_RD_ranks` 性能**：现状 4-nested numpy loop，单张 256×256 ~30~60s。是 Phase A 内做向量化（PyTorch 化 + GPU 化），还是先迁过来标 `# TODO(perf)` 后置？倾向后置——只在训练时调，eval 不走这条路径。
