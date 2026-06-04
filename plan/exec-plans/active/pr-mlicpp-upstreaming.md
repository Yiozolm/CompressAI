# pr-mlicpp: MLIC 系列上游迁入（MLIC + MLIC+ + MLIC++ + MLICv2）+ `MultiContextCheckerboardLatentCodec` 抽象执行计划

**计划日期**：2026-05-10（v2，把 scope 从「lift dedicated codec as-is」pivot 到「抽象 + lift」）；2026-05-11 v3 修订：scope 扩展到 MLIC 全系列（MLIC + MLIC+ + MLIC++ + MLICv2 合并到本 PR，原计划的 PR-5/PR-6 撤销）；2026-05-11 v3.1 修订：新增 Phase 11.5 把 MLIC++ 统一到 `_BaseMLIC` 模板（删除 `compressai/models/mlicpp.py`，4 个模型全合到 `compressai/models/mlic.py`）+ Phase 5 ckpt smoke 重排到 Phase 14 之后做（4 模型 wiring 全部到位再统一验证）
**状态**：MLIC++ Track Phase 1-4 已落地；**Phase 4.5 anchor_parity P0 代码修复已随 Phase 11 回补**；v1 / v1+ Track Phase 10-11 已落地；**Phase 11.5（MLIC++ unify）已落地**；**Phase 12（MLICv2 leaf selective hook）已落地**；**Phase 13（MLICv2 layers）已落地**；**Phase 14（MLICv2 model/factory/zoo）已落地**；**Phase 5（统一 convert script + MLIC++ 真实 ckpt smoke）已落地**；**Phase 15（联合验证）已落地，本地 macOS 仅 DDP smoke 因 rendezvous 挂起需交给 Linux CI/maintainer 环境复验**；**Phase 16 push + PR draft 已完成**（实际打开 upstream PR 等 user 时机）；SGA generic codec generalization 已完成并归档
**新执行顺序**：1-4 ✅ → 10-11 ✅ → **11.5 ✅** → **12 ✅** → **13 ✅** → **14 ✅** → **5 ✅** → **15 ✅** → **16 push+PR draft ✅ / open PR pending**
**关联设计**：[`plan/design-docs/mlic-family-reproduction.md`](../../design-docs/mlic-family-reproduction.md) —— 本 PR 内 phase 拆分与四模型 leaf 槽位映射的依据
**分支**：`pr-mlicpp`（基于 `upstream/master`；与 pr-dcae-saaf-auxt 独立，可并行 review）
**目标 PR**：本仓向上游 `InterDigitalInc/CompressAI` 提交的 Family 2 系列**第一个** PR；**v3 scope**：MLIC 系列**四模型合并提交**（MLIC + MLIC+ + MLIC++ + MLICv2），由 `MultiContextCheckerboardLatentCodec` + 应用层共享子包统一承载（[mlic-family-reproduction.md §5](../../design-docs/mlic-family-reproduction.md) 详述）
**前置依赖**：仅依赖 [`pr-tcm-cca`](../completed/codec-containerization-h-g-refactor.md) 提供的 `HyperpriorLatentCodec` + `ChannelGroupsLatentCodec`（已扩展 `max_support_slices` / `support_filter` / `side_in_context`）+ `LRPGaussianLatentCodec` + `_slice_helpers`。**不依赖** pr-dcae-saaf-auxt（MLIC++ 不用 OLP/wavelet/SSM）
**设计文档**：[`plan/design-docs/channel-slice-codec-redesign.md`](../../design-docs/channel-slice-codec-redesign.md) §2.2 Family 2 表 / §3.4 dedicated codec 段（**v2 注**：本 PR 把 MLIC++ 从「dedicated codec」family pivot 出来，改为抽象层共享）
**路线图**：[`family2-roadmap.md`](family2-roadmap.md) PR-1
**历史决策**：[`mlicpp-latent-codec-refactor.md`](../completed/mlicpp-latent-codec-refactor.md)（fork `script` 上 `MLICPlusPlusLatentCodec` monolith 抽出决策 —— 本 PR **超越**这个决策，改为更通用的 leaf 抽象）

---

## ⭐ Scope 锁定（v3，把 MLIC 家族四模型合并到本 PR）

**核心扩展（v3 vs v2）**：v2 只覆盖 MLIC++ 一个模型，v3 把 MLIC + MLIC+ + MLICv2 一并放进本 PR（详见 [`mlic-family-reproduction.md`](../../design-docs/mlic-family-reproduction.md) §5）：

- 四模型共享同一 `MultiContextCheckerboardLatentCodec`（leaf 抽象），MLICv2 启用 leaf 的 `selective_predictor` 可选 hook 处理 GSC
- 四模型共享 `compressai/layers/lic/mlic/` 应用层子包；MLICv2 因 STM transform / HGCP / CR / 2D RoPE / GSC 改动大，新增 `compressai/layers/lic/mlicv2/` 子包
- Phase 14 后 MLIC / MLIC+ / MLIC++ / MLICv2 共用 `build_mlic_slice_codec(variant=...)` 与 `compressai/models/mlic.py`
- 一次 review cycle 覆盖整个 MLIC 家族；reviewer 看到抽象层的通用性 vs 单模型 PR 看不到
- 风险：PR diff 大（~3300 LoC）；mitigation 是 commit 序列高度结构化，若 reviewer 要求拆 PR，建议在 Phase 9 / Phase 10 边界切

**核心 pivot（v1→v2，保留）**：把 codec 拆成两层：

1. **抽象层**：在 upstream 新增 sibling leaf codec `MultiContextCheckerboardLatentCodec`（作为 `CheckerboardLatentCodec` 的广义化），可被未来 Family 2 模型（首先是 MambaIC，之后可能 follow-up 上游模型）复用；
2. **应用层**：MLIC++ model 改写为 thin wrapper（沿 ELIC pattern：`g_a + g_s + HyperpriorLatentCodec(ChannelGroupsLatentCodec(K × MultiContextCheckerboardLatentCodec))`）；
3. MLIC++ 论文独有的 building blocks（`LocalContext` / `LinearGlobalInter/IntraContext` / `ChannelContext` / `EntropyParameters` / `LatentResidualPrediction` / checkerboard utils）保留在 `compressai/layers/lic/mlic/` 子包，作为应用层 helper 注入新 leaf 的参数。

**做**：
- **新增** `compressai/latent_codecs/multi_context_checkerboard.py`（~280 LoC）—— `MultiContextCheckerboardLatentCodec` sibling，generalize `CheckerboardLatentCodec` 接受：
  - **separate `entropy_parameters_anchor` / `entropy_parameters_nonanchor`** 两个 `nn.Module`（vs 上游单 head + masking）
  - 可选 `spatial_context_anchor: Optional[nn.Module]`（None 时跳过；MLIC++ 该 slot 为 None）
  - 可选 `spatial_context_nonanchor: Optional[nn.Module]`（None 时退化为 `_y_ctx_zero`；MLIC++ 用 `LocalContext`，ELIC 等价改造可用 `CheckerboardMaskedConv2d`）
  - 可选 `intra_channel_context_nonanchor: Optional[nn.Module]`（接受 `(side_params, anchor_y_hat)` 双输入；Phase 3 的 `_MlicppPriorAggregation` 先把 prior-slice 信息编码进 `side_params`）
  - 可选 `lrp_anchor` / `lrp_nonanchor`（None 时跳过；MLIC++ 两套都启用）
  - `lrp_input_builder: Optional[Callable]`（None 时退化为 `cat([side_params, y_hat])`；MLIC++ closure 可从 `side_params` 拆出 hyper/prior 信息并拼 `current_quant`）
  - 内部固定 anchor → nonanchor 双 pass + `checkerboard_split/merge` + 单一 `gaussian_conditional` 算 likelihood（沿用 upstream `CheckerboardLatentCodec` 的二 pass 模板）
- lift `compressai/layers/lic/mlic/{__init__,context.py,transforms.py,utils.py}` 子包（共 ~640 LoC，**作为 application-layer helpers**，不是 codec primitive）
- 新增 `compressai/models/_helpers/multi_context_slice.py`（Phase 11.5 后以 `build_mlic_slice_codec(variant=...)` 统一覆盖 MLIC / MLIC+ / MLIC++，side layout wrappers 包装 K × `MultiContextCheckerboardLatentCodec` 进 `ChannelGroupsLatentCodec`）
- 新增 / 收敛 `compressai/models/mlic.py`（Phase 11.5 后包含 `_BaseMLIC` + `MLIC` + `MLICPlus` + `MLICPlusPlus`，沿 STF/TCM/CCA pattern 用 `HyperpriorLatentCodec` 包外层 + `g_a` / `g_s` 在模型类；`compressai/models/mlicpp.py` 已删除）
- lift `examples/convert_mlic_checkpoint.py --variant {mlic|mlic+|mlicpp|mlicv2}`（CLI wrapper；替代早期 `convert_mlicpp_checkpoint.py` 单模型脚本）+ rename map 把上游 ckpt key 从 fork `script` 老 `MLICPlusPlusLatentCodec` 路径迁到新 `latent_codec.y.latent_codec.y{k}.{entropy_parameters_anchor,...}` 路径
- zoo 接线（`mlicpp` factory + `_LazyImport` proxy）
- tests + state_dict round-trip + 上游 ckpt smoke

**不做**：
- **不再 lift `compressai/latent_codecs/mlicpp.py`**（fork `script` 上的 monolith codec，被新 `MultiContextCheckerboardLatentCodec` + 应用层 factory 等价替代；fork `script` 仓里 ~474 LoC 净删除，**这是 v1→v2 pivot 的核心收益**）
- 不把 `LinearGlobalInter/IntraContext` / `ChannelContext` 等 MLIC++ 论文 building blocks 上升为通用 `latent_codecs/` primitive —— 它们是 application-layer helper，跟 ELIC 的 `sequential_channel_ramp` 同级；只有「双 head + 多 context + 双 LRP 的 checkerboard 内层 codec」这一层是真正可复用的 codec primitive
- 不动 ELIC、不动 upstream `CheckerboardLatentCodec`（新 leaf 是 sibling，不重写既有类，state_dict / API 零冲击）
- 不预测 MambaIC 的迁移路径 —— `MultiContextCheckerboardLatentCodec` 设计上**预留**「spatial_context_nonanchor 可被替换为 VSS Mamba block」的可能性，但 MambaIC 是否最终改用本 leaf 留到 PR-3 评估

---

## 抽象设计草稿

### 与 upstream `CheckerboardLatentCodec` 的接口差异

| 维度 | upstream `CheckerboardLatentCodec` | 新 `MultiContextCheckerboardLatentCodec` |
|---|---|---|
| EP heads | 单 `entropy_parameters: nn.Module` 共享，靠 mask 切 anchor / nonanchor | `entropy_parameters_anchor` + `entropy_parameters_nonanchor` 两 head |
| 空间上下文 | `context_prediction: CheckerboardMaskedConv2d`（mask conv on combined latent）| `spatial_context_anchor: Optional[nn.Module]`（None=skip）+ `spatial_context_nonanchor: Optional[nn.Module]`（None=degrade to `_y_ctx_zero`，传入 `anchor_y_hat`）|
| 通道间上下文 | n/a（外层 `ChannelGroupsLatentCodec` 提供）| 同（`side_params` 已经包含外层来的 channel + global_inter context）+ 新增 `intra_channel_context_nonanchor: Optional[nn.Module]` 接受 `(side_params, anchor_y_hat)` |
| LRP | n/a | 可选 `lrp_anchor` / `lrp_nonanchor` 两套，per-pass 应用，`lrp_input_builder` 可定制输入拼接 |
| Likelihood | 单 `gaussian_conditional` on merged params | 同 |
| Forward 模式 | onepass / twopass / twopass_faster | 仅 twopass（separate heads 不能 onepass） |

ELIC 等价改造（**本 PR 不做**，仅作为「设计是否过度）的回归验证想法）：把 ELIC 现有 `CheckerboardLatentCodec(entropy_parameters=mlp, context_prediction=CheckerboardMaskedConv2d)` 改写为 `MultiContextCheckerboardLatentCodec(entropy_parameters_anchor=mlp, entropy_parameters_nonanchor=mlp, spatial_context_nonanchor=CheckerboardMaskedConv2d)` 应该 likelihoods 等价（参数不共享所以训练后会发散，但 init 后 forward 数值应一致）。**不在 PR scope 内验证**，仅作为「leaf 接口确实泛化了」的设计自检题。

### 应用层 wiring sketch

```python
# compressai/models/mlicpp.py（thin model，~130 LoC）

class MLICPlusPlus(CompressionModel):
    def __init__(self, N=192, M=320, slice_num=10, context_window=5):
        super().__init__()
        self.g_a = ...  # MLIC++ analysis transform
        self.g_s = ...  # MLIC++ synthesis transform
        slice_ch = M // slice_num

        self.latent_codec = HyperpriorLatentCodec(            # ← upstream
            h_a=HyperAnalysis(M, N),
            h_s=HyperSynthesis(M, N),                          # ← single h_s（MLIC++ 不像 STF 用双 h_s）
            latent_codec={
                "z": EntropyBottleneckLatentCodec(EntropyBottleneck(N), quantizer="noise"),
                "y": build_mlicpp_slice_codec(                 # ← 新应用层 factory
                    M=M, slice_num=slice_num, slice_ch=slice_ch,
                    context_window=context_window,
                ),
            },
        )

# compressai/models/_helpers/multi_context_slice.py（实际 318 LoC）
def build_mlicpp_slice_codec(*, M, slice_num, slice_ch, context_window):
    return ChannelGroupsLatentCodec(
        groups=[slice_ch] * slice_num,
        max_support_slices=-1,
        side_in_context=True,                                  # MLIC++ side params 跨 slice 0
        channel_context={
            f"y{k}": _MlicppPriorAggregation(slice_ch, k)      # 应用层 helper：内部跑
                                                                #   ChannelContext + LinearGlobalInterContext + cat
            for k in range(slice_num)
        },
        latent_codec={
            f"y{k}": MultiContextCheckerboardLatentCodec(      # ← 新 leaf
                entropy_parameters_anchor=EntropyParameters(_anchor_in(k), 2*slice_ch),
                entropy_parameters_nonanchor=EntropyParameters(_nonanchor_in(k), 2*slice_ch),
                spatial_context_nonanchor=LocalContext(slice_ch, context_window),
                intra_channel_context_nonanchor=LinearGlobalIntraContext(slice_ch) if k > 0 else None,
                lrp_anchor=LatentResidualPrediction(M + (k+1)*slice_ch, slice_ch),
                lrp_nonanchor=LatentResidualPrediction(M + (k+1)*slice_ch, slice_ch),
                lrp_input_builder=_mlicpp_lrp_inputs,           # 应用层 closure
                gaussian_conditional=GaussianConditional(None),  # per-slice 副本（K 份，与 Family 1 一致）
            )
            for k in range(slice_num)
        },
    )
```

### state_dict 路径变化（vs fork `script` 老 layout）

| 老（fork `script`）| 新（v2 H+G after pivot）|
|---|---|
| `entropy_bottleneck.*` | `latent_codec.z.entropy_bottleneck.*` |
| `h_a.*` / `h_s.*` | `latent_codec.h_a.*` / `latent_codec.h_s.*` |
| `latent_codec.local_context.{k}.*` | `latent_codec.y.latent_codec.y{k}.spatial_context_nonanchor.*` |
| `latent_codec.channel_context.{k}.*` + `latent_codec.global_inter_context.{k}.*` | `latent_codec.y.channel_context.y{k}.{channel_part,global_inter_part}.*`（应用层 `_MlicppPriorAggregation` 内部）|
| `latent_codec.global_intra_context.{k}.*` | `latent_codec.y.latent_codec.y{k}.intra_channel_context_nonanchor.*` |
| `latent_codec.entropy_parameters_anchor.{k}.*` | `latent_codec.y.latent_codec.y{k}.entropy_parameters_anchor.*` |
| `latent_codec.entropy_parameters_nonanchor.{k}.*` | `latent_codec.y.latent_codec.y{k}.entropy_parameters_nonanchor.*` |
| `latent_codec.lrp_anchor.{k}.*` | `latent_codec.y.latent_codec.y{k}.lrp_anchor.*` |
| `latent_codec.lrp_nonanchor.{k}.*` | `latent_codec.y.latent_codec.y{k}.lrp_nonanchor.*` |
| `latent_codec.gaussian_conditional.*` | `latent_codec.y.latent_codec.y{k}.gaussian_conditional.*`（K 份，per-slice 副本）|

`convert_upstream_mlicpp_state_dict` 把上游作者发布 ckpt（非 fork-script layout，是 `JiangWeibeta/MLIC` published layout，可能更扁平）映射到新路径，rename map 加一段。

---

## Phase 0-9（v2，比 v1 多 1 个抽象设计 phase + 1 个 ELIC 等价回归）

### Phase 0：清理 working tree + 分支建立（30 分钟）

- [ ] 等 pr-tcm-cca merge 进 upstream/master
- [ ] 基于 `upstream/master` 创建 `pr-mlicpp`
- [x] 开发期可临时基于 `pr-tcm-cca` 抽 cherry-pick

### Phase 1：抽象设计 + 新 leaf codec 实现（1.5 d）✅ 完成于 2026-05-10

- [x] 落 `compressai/latent_codecs/multi_context_checkerboard.py`（**324 LoC**，原估 ~280；helper 抽出后从初版 348 行减到 311 行，Phase 2 追加 `lrp_activation` contract 后为 324 行）
  - forward / compress / decompress 三方法实现 anchor → nonanchor 双 pass
  - 所有可选 hook 默认 `None`，只有 EP heads 必填
  - **`spatial_context_*=None` 是「skip y_ctx 槽」语义**（不 zero-pad）—— 详见 Phase 1 实施差异 §Bug A
  - `__all__` 加 `"MultiContextCheckerboardLatentCodec"`
- [x] **抽出 `compressai/latent_codecs/_checkerboard_helpers.py`**（**新文件，145 LoC**）—— 7 个 module-level 纯函数：`embed` / `unembed` / `mask_all` / `mask_all_but_step` / `merge` / `write_step` / `step_parity`，全接受 `anchor_parity` 关键字参数。`compressai/latent_codecs/checkerboard.py` 与新 sibling 都改为 `from . import _checkerboard_helpers as _ckb` + 调用 `_ckb.*`，删除两边各自 6 个等价 instance method（共 ~140 行重复）
- [x] `compressai/latent_codecs/__init__.py` export `MultiContextCheckerboardLatentCodec`
- [x] 单元测试 `tests/test_multi_context_checkerboard.py::TestMultiContextCheckerboardLatentCodec`（**8 测试全过**）：
  - `test_default_forward_shapes` —— 默认参数（仅 EP heads，其他 hook 全 None）forward shape
  - `test_anchor_skips_spatial_context_when_none`（**Phase 1 后期新增，锁定 Bug A 修复语义**）—— anchor head 输入仅 `side_ch`，验证 None 时不补零
  - `test_spatial_nonanchor_forward_shapes` —— 仅 spatial_context_nonanchor 启用 forward shape
  - `test_all_hooks_forward_shapes_and_state_dict_paths` —— 全 hook 启用 forward shape + state_dict 路径自检
  - `test_lrp_activation_can_be_skipped`（**Phase 2 后补，锁定 MLIC++ LRP contract**）—— LRP module 已内置 bounded activation 时可设 `lrp_activation=None` 跳过 leaf 侧二次 `tanh`
  - `test_state_dict_round_trip` —— 两实例间 state_dict round-trip + forward 数值一致
  - `test_compress_decompress_round_trip` —— forward / compress / decompress 三路径 y_hat allclose
  - `test_matches_checkerboard_latent_codec_when_heads_are_shared` —— **ELIC 等价回归**：sibling leaf 配 `spatial_context_anchor=_ZeroContext(out_channels)` + `spatial_context_nonanchor=context_prediction` + 共享 EP head 时，forward 数值与 upstream `CheckerboardLatentCodec._forward_twopass` 完全一致
- [x] 全套回归（`test_models + test_latent_codecs + test_models_helpers + test_layers + test_init + test_multi_context_checkerboard`）**81/81 通过**，含 ELIC 在 sensetime.py 用 `CheckerboardLatentCodec` 的全部 forward / compress / decompress 路径不回归
- [x] `make static-analysis` 三步全过（ruff format / imports / lint）

### Phase 1 实施差异 / 决策记录

#### Bug A: `spatial_context_*=None` 语义从「padding zeros」修正为「skip slot」

**初版实现错误**：`_ctx_params` 在 `spatial_context_*=None` 时返回 `_mask_all(y)` 即 `(B, y.shape[1], H, W)` 零张量，并把它塞进 `ctx_parts[0]`。head 输入意外多出 `y.shape[1]` 通道。

**对 MLIC++ 的硬影响**：上游 `entropy_parameters_anchor[0]` 输入维度 = `2M`（仅 hyper_params，无空间 ctx、无 channel ctx）。容器化后 side_params width = 2M，用初版实现 anchor 输入会变成 `2M + slice_ch`，**byte-for-byte 上游 ckpt 加载失败**。

**修正**：`_ctx_params` / `_ctx_params_packed` 改为 `spatial_context_*=None` 时不塞 y_ctx 槽（`intra_channel_context_nonanchor=None` 同样跳过）；删除中间私有 helper `_spatial_context` / `_spatial_context_from_y_hat`；新增单行 `_spatial_context_module(step)` selector。类 docstring 加 contract 段明确「optional context hooks 为 None 时**omit**，不 zero-pad」+「entropy_parameters head 必须按实际启用的 hook 计算 input width」。

**ELIC 等价回归测试同步改写**：upstream `CheckerboardLatentCodec._forward_twopass` anchor pass 用 `_y_ctx_zero(y) = _mask_all(context_prediction(y))` 即 `context_prediction.out_channels` 通道零张量，**与新 sibling 的「skip slot」语义不一致**。新 leaf 要复刻 ELIC 行为必须显式传 `spatial_context_anchor=_ZeroContext(out_channels)`（测试模块新加 helper）。这条 contract 让新 leaf 比 ELIC 通用：当 `out_channels != y.shape[1]`（真实 ELIC 配 `2*slice_ch`）时新 leaf 仍正确，初版实现会通道数对不上。

#### Helper 抽出 `_checkerboard_helpers.py` 在 Phase 1 内完成（不是延后）

原计划 Phase 1 只放新 leaf；helper 抽出在风险表里作为可选缓解。实际 Phase 1 二段一并做了，**单一 source of truth** 立刻生效：未来 anchor parity 边界 bug fix 只在一个地方改，自动同步到 `CheckerboardLatentCodec` + `MultiContextCheckerboardLatentCodec`（以及未来 sibling）。

**保留为 instance method**：`CheckerboardLatentCodec.{_y_ctx_zero, quantize}`（class-private 一行 helper）；`MultiContextCheckerboardLatentCodec.embed_step`（new leaf 专用，LRP packed 路径用，是 `_ckb.unembed` 的反操作 + 对 step_index 槽以外位置补零）。

**保留为 instance attr**：两 codec 类的 `anchor_parity` / `non_anchor_parity` —— 公开 API，用户代码可能依赖。

#### `intra_channel_context_nonanchor` 签名变化对 Phase 3 的影响

签名定为 `(side_params, anchor_y_hat)`，但 MLIC++ 上游 `LinearGlobalIntraContext.forward(prev_slice_y_hat, anchor_hat)` 接受**前一个 slice 的 y_hat**（`y_hat_slices[-1]`）—— 不在 leaf 能拿到的 side_params 里。

→ **Phase 3 应用层 wiring 必须**：
1. `_MlicppPriorAggregation` 输出 side_params 时把 `prev_slice_y_hat` 拼进末尾（slice 0 占空位即 `slice_ch` 通道零）
2. 写 `_MlicppIntraWrapper(intra_ctx, side_layout)` 把 side_params 切片出 `prev_slice_y_hat` 喂给 `LinearGlobalIntraContext`
3. `lrp_input_builder` closure 同样从 side_params 拆 `hyper_means` / `prior_y_hats` 拼上 LRP 上游 input layout

替代方案 B（让 leaf forward 增加 `outer_y_hat_prev` kwarg）需要改 `LatentCodec` 接口契约，被否决。**A 方案的代价就是 Phase 3 的 helper 复杂度**——Phase 3 sketch 已记录这点，下一步 #3 把 sketch 细化时同步。

#### LoC 数据

| 文件 | 当前 LoC |
|---|---|
| `compressai/latent_codecs/multi_context_checkerboard.py` | 324 |
| `compressai/latent_codecs/_checkerboard_helpers.py`（新）| 145 |
| `compressai/latent_codecs/checkerboard.py`（trim 6 instance method 后）| 329（原 ~395，−66）|
| `tests/test_multi_context_checkerboard.py` | 285（含 `_ZeroContext` / LRP helper + 8 测试）|
| **Phase 1 净增** | **+698**（vs 原估 ~280）|

净增大于估算因为：(a) 包许可证头 + module docstring 占 ~30 行/文件；(b) `_checkerboard_helpers.py` 抽出是计划外的 Phase 1 二段；(c) ELIC 等价回归 + Bug A skip-semantics 测试引入 `_ZeroContext` helper + 1 个新测试。回报：sibling 关系结构化、上游 leaf 行为零回归。

### Phase 2：lift `compressai/layers/lic/mlic/` 子包（1 d）✅ 完成于 2026-05-10

- [x] lift `compressai/layers/lic/mlic/{__init__.py,context.py,transforms.py,utils.py}`，并新增最小 `compressai/layers/lic/__init__.py` 让子包可被 setuptools 打包
- [x] **不**在 `compressai/layers/__init__.py` 顶层 re-export `from .lic.mlic import *` —— deep-import only；测试锁定 `compressai.layers.LocalContext` / `EntropyParameters` 不存在
- [x] **类别变更**：`AnalysisTransform` / `SynthesisTransform` 是 MLIC++ g_a / g_s（应用层），不是 codec primitive；代码 docstring 标注为 MLIC++ transform
- [x] 单元测试 `tests/test_mlic_layers.py`（14 测试全过）：每个 context / transform module forward shape + state_dict round-trip + checkerboard utils 数值校验

### Phase 2 实施差异 / 决策记录

#### `LocalContext` 复用现有 attention MLP，不新增本地 MLP

按 user 评估反馈，MLIC family 归入 `[attn]` optional dependency surface。`LocalContext` 不放本地 `Mlp` 复制实现，而是 `from compressai.layers.attn.swin import Mlp` 复用现有 attention stack 里的 `timm` MLP；MLICv2 也复用 `timm.layers.LayerNorm2d`，不再在 `context.py` / `transforms.py` 各复制一份私有 `_LayerNorm2d`。`mlic` / `mlicv2` 子包仍是 deep-import only，不从 `compressai/layers/__init__.py` 暴露，因此 `import compressai.layers` 不触发可选 `[attn]` 依赖。

#### `AnalysisTransform` / `SynthesisTransform` 保留 GELU，但不改共享 `ResidualBlock*`

fork `script` 上的 `compressai.layers.ResidualBlock*` 已扩展 `act=` kwarg；当前 `pr-mlicpp` 基线没有这个改动。Phase 2 没有把 `act=` 扩到共享 `compressai/layers/layers.py`，而是在 `mlic/transforms.py` 内放 MLIC++ 私有 `_ResidualBlock*` helpers，保持 GELU 行为与权重路径，同时把对既有上游层的改动控制为 0。

**2026-05-11 cleanup pass**：该 follow-up 已在本 PR 内落地到低风险版本：

1. `compressai.layers.layers.ResidualBlockWithStride` / `ResidualBlockUpsample` / `ResidualBlock` 新增 `act: Type[nn.Module] = nn.LeakyReLU` 参数，默认仍构造 `LeakyReLU(inplace=True)`，保持既有模型行为不变
2. `compressai.layers.lic.mlic.transforms` 删除私有 `_ResidualBlockWithStride` / `_ResidualBlockUpsample` / `_ResidualBlock`，改用公开 residual blocks 并传 `act=nn.GELU`
3. `compressai.layers.lic.mlicv2.transforms` 删除私有 `_ResidualBlockWithStride` / `_ResidualBlockUpsample`，同样改用公开 residual blocks 并传 `act=nn.GELU`
4. 暂不抽象 `norm=`，因为这会把 GDN / IGDN 的调用约定扩到公共 API，影响面大于本轮 cleanup

#### LRP contract: leaf 可跳过二次 activation

fork `script` 的 `LatentResidualPrediction.forward()` 已经返回 `0.5 * tanh(raw)`，而 Phase 1 leaf 默认把 LRP module 当 raw residual predictor 处理（`lrp_scale * tanh(lrp(...))`）。Phase 2 后补 `MultiContextCheckerboardLatentCodec(lrp_activation=...)`：默认仍是 `torch.tanh`，Phase 1 既有行为不变；MLIC++ Phase 3 wiring 会设 `lrp_activation=None, lrp_scale=1.0`，直接使用 application-layer LRP module 的 bounded residual，避免 double tanh。

#### LoC 数据

| 文件 | 当前 LoC |
|---|---|
| `compressai/layers/lic/__init__.py` | 36 |
| `compressai/layers/lic/mlic/__init__.py` | 83 |
| `compressai/layers/lic/mlic/context.py` | 411 |
| `compressai/layers/lic/mlic/transforms.py` | 265 |
| `compressai/layers/lic/mlic/utils.py` | 181 |
| `tests/test_mlic_layers.py` | 211 |
| **Phase 2 净增** | **+1187** |

### Phase 3：应用层 factory（0.5 d）✅ 完成于 2026-05-10

- [x] 新增 `compressai/models/_helpers/multi_context_slice.py`（**318 LoC**）`build_mlicpp_slice_codec(...)` factory
- [x] 内部辅助 `_MlicppPriorAggregation(M, slice_ch, k)` 把 `ChannelContext + LinearGlobalInterContext` 包成 `nn.Module`，contract: `forward(cat(hyper_params, *prior_y_hat)) -> side_layout`（与 `ChannelGroupsLatentCodec(side_in_context=True)` contract 对齐）
- [x] 新增 `_MlicppSideLayout` / `_MlicppEntropyParameters` / `_MlicppIntraWrapper` / LRP input builder，恢复 fork `script` 原始输入顺序：
  - anchor EP: `hyper_params`（k=0）或 `cat(global_inter_ctx, channel_ctx, hyper_params)`（k>0）
  - nonanchor EP: `cat(local_ctx, hyper_params)`（k=0）或 `cat(local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params)`（k>0）
  - LRP: `cat(hyper_means, *prev_y_hat, current_slice)`，其中 nonanchor pass 的 `current_slice` 为完整当前 slice（anchor + nonanchor pre-LRP）
- [x] `MultiContextCheckerboardLatentCodec` 小修：`lrp_input_builder` 在 nonanchor pass 看到完整 current slice；no-LRP 返回仍只写当前 checkerboard step，保证 ELIC 等价回归不变
- [x] 单元测试 `tests/test_models_helpers.py::TestBuildMlicppSliceCodec`（5 测试全过）：factory 类型/keys、state_dict 路径、forward shape、compress/decompress round-trip、参数校验

### Phase 3 实施差异 / 决策记录

#### Factory 比 sketch 大：side layout wrapper 是必要复杂度

原 sketch 估 `multi_context_slice.py` ~60 LoC，只写 `_MlicppPriorAggregation + build_mlicpp_slice_codec`。实际落地为 318 LoC，主要因为 Phase 1 已决定不扩展 `LatentCodec` 接口，`MultiContextCheckerboardLatentCodec` 只能接收一个 `side_params` 张量。因此 Phase 3 必须在应用层显式保存 side layout，并用 wrapper 把它拆回 MLIC++ 原始模块输入顺序。

这个复杂度换来两个好处：(a) 不改 `ChannelGroupsLatentCodec` / `LatentCodec` 公共 contract；(b) 后续 checkpoint convert 只需要做路径 rename，不需要重排 `EntropyParameters.fusion.0.weight` 的输入通道。

#### Nonanchor LRP 输入修正

fork `script` 上 `lrp_nonanchor[k]` 的输入是 `cat(hyper_means, *y_hat_slices, y_hat_slice)`，其中 `y_hat_slice` 包含已 LRP 的 anchor 与尚未 LRP 的 nonanchor。Phase 1 leaf 原实现把当前 pass 的 masked `y_hat_i` 直接交给 `lrp_input_builder`，对 MLIC++ nonanchor LRP 少了 anchor 部分。Phase 3 修正为：nonanchor builder 看到完整 current slice，但 `_apply_lrp` 的返回仍 mask 到当前 step，避免 no-LRP 路径把 anchor 重复写回。`test_matches_checkerboard_latent_codec_when_heads_are_shared` 与 `test_compress_decompress_round_trip` 保持通过。

#### state_dict 路径

Phase 3 后 `ChannelGroupsLatentCodec` root 下关键路径为：

- `channel_context.y{k}.channel_part.*`
- `channel_context.y{k}.global_inter_part.*`
- `latent_codec.y{k}.entropy_parameters_anchor.fusion.*`
- `latent_codec.y{k}.entropy_parameters_nonanchor.fusion.*`
- `latent_codec.y{k}.spatial_context_nonanchor.*`
- `latent_codec.y{k}.intra_channel_context_nonanchor.*`（k>0）
- `latent_codec.y{k}.lrp_anchor.*`
- `latent_codec.y{k}.lrp_nonanchor.*`
- `latent_codec.y{k}.y.gaussian_conditional.*`

### Phase 4：lift `compressai/models/mlicpp.py`（1 d）✅ 完成于 2026-05-10

- [x] 写 thin `MLICPlusPlus` 模型类（**290 LoC**，沿 STF / TCM / CCA pattern；含 SPDX header + Apache 2.0 attribution + convert script + ctor 推断 helper）
- [x] `g_a` / `g_s` 用 `compressai.layers.lic.mlic.{AnalysisTransform, SynthesisTransform}`（Phase 2 落点）
- [x] `latent_codec` 用 `HyperpriorLatentCodec(h_a=HyperAnalysis, h_s=HyperSynthesis, latent_codec={"z": EntropyBottleneckLatentCodec, "y": build_mlicpp_slice_codec(...)})`（Phase 3 factory）
- [x] 模型基类 `CompressionModel`，**4 行 forward + 3 行 compress + 2 行 decompress** 委托给 `self.latent_codec`，没有 `SimpleVAECompressionModel` cherry-pick
- [x] `from_state_dict` 自动覆盖**三个 ckpt 时代**：(a) 当前 containerized layout、(b) fork `script` `MLICPlusPlusLatentCodec` monolith、(c) mlicpp-latent-codec-refactor.md era（root `latent_codec.X` 但 X 是单层 monolith）；任一 era 上叠加 `module.` DataParallel 前缀也支持
- [x] **4 个 ctor kwarg 全自动从 state_dict 推断**：`N` ← `g_a.analysis_transform.0.conv1.weight.size(0)`、`M` ← `g_a.analysis_transform.6.weight.size(0)`、`slice_num` ← `_infer_slice_num()` 扫描 new + legacy 两套正则、`context_window` ← `_infer_context_window()` 看 `relative_position_index` 形状（fallback `relative_position_table`）—— user 调 `MLICPlusPlus.from_state_dict(state_dict)` 不需要传任何 kwarg
- [x] **删除** `mlicpp-latent-codec-refactor.md` 时代加的 model root `entropy_bottleneck` / `gaussian_conditional` read-only 兼容属性 —— v2 全部走新路径 + convert script，不留 legacy property（测试显式 `assert not hasattr(model, "entropy_bottleneck")` 锁定）
- [x] 单元测试 `tests/test_models.py::TestMlicPlusPlus`（2 测试 / 149 行）：
  - `test_forward_and_state_dict_round_trip` —— tiny config `(N=8, M=16, slice_num=4, context_window=3)` 端到端 forward + 9 个 state_dict path 正断言（含 `latent_codec.{h_a, h_s, z.entropy_bottleneck, y.channel_context.y1.{channel_part, global_inter_part}, y.latent_codec.y0.{entropy_parameters_anchor.fusion, lrp_anchor.lrp_transform, y.gaussian_conditional}, y.latent_codec.y1.intra_channel_context_nonanchor.keys}` 全套）+ 5 个**否定**断言（旧路径 / `entropy_bottleneck` / `gaussian_conditional` root attr 全部消失）+ `from_state_dict(model.state_dict())` round-trip + 4 个 ctor kwarg 推断校验
  - `test_legacy_state_dict_conversion` —— synthetic legacy state_dict（`module.` DataParallel + `latent_codec.{name}.{k}.*` monolith）→ 14 个新路径正断言 + 3 个旧路径否定断言：覆盖 8 个 per-slice helper rename + DataParallel strip + **`gaussian_conditional` 1→K fanout**（`latent_codec.gaussian_conditional.scale_table` → `latent_codec.y.latent_codec.y{0,1}.y.gaussian_conditional.scale_table` 两份）

### Phase 4 实施差异 / 决策记录

#### 290 LoC vs plan 130 LoC：convert script 区块占 ~120 LoC

estimate 严重偏低。原因：plan 把 convert script 算进了 model class 130 LoC 估算，但实际 convert 区块（`_CURRENT_SLICE_RE` + `_LEGACY_SLICE_RE` + `_ROOT_TO_CONTAINER_PREFIXES` + `_LEGACY_LIST_RENAMES` + 4 个 helper 函数 + `_infer_context_window`）就接近 ~120 LoC。其余分布：30 行 license header、22 行 imports、110 行 `MLICPlusPlus` 类本体（其中 `__init__` 40 行含 4 项 ctor 校验、`from_state_dict` 20 行含 `relative_position_index` allowed-missing 处理、`forward`/`compress`/`decompress` 共 19 行）。**估算偏差，不是工作偏差**。

#### `_convert_mlicpp_key` 返回 `List[str]` 处理 1→K fanout

`gaussian_conditional` 上游单一共享 GC，容器化后是 K 份 per-slice 副本。convert 必须把 `latent_codec.gaussian_conditional.scale_table` 一行复制 K 份到 `latent_codec.y.latent_codec.y{0..K-1}.y.gaussian_conditional.scale_table`。把 `_convert_mlicpp_key` 签名设计成返回 `List[str]`（而不是 caller-side 展开），1→K 在 helper 内部局部化，更干净。

#### Convert 是 idempotent，对已 containerized 的 state_dict 是 no-op

对 `model.state_dict()` 自身再跑 convert：(a) `latent_codec.h_a.*` / `latent_codec.y.*` 等新路径不匹配 `_ROOT_TO_CONTAINER_PREFIXES` 的 root 前缀（不含 `latent_codec.` 前缀的项），不匹配 `_LEGACY_LIST_RENAMES`（剥 `latent_codec.` 后第一段是 `y` 不是 `local_context` 等 8 个名字之一），落到 fall-through 返回 `[key]` 不变。idempotency 由 `test_forward_and_state_dict_round_trip` 通过 `from_state_dict(model.state_dict())` 隐式覆盖。

#### 三个 ckpt 时代覆盖

| Era | 关键路径形态 | Convert 处理路径 |
|---|---|---|
| fork-script pre-refactor | 顶层 `h_a.*` + `entropy_bottleneck.*` + `local_context.{k}.*` 等 | `_ROOT_TO_CONTAINER_PREFIXES` + 走 fall-through 到 `_LEGACY_LIST_RENAMES` |
| mlicpp-latent-codec-refactor.md era | `latent_codec.h_a.*` + `latent_codec.entropy_bottleneck.*` + `latent_codec.{name}.{k}.*` | `latent_codec.h_a.*` 自然兼容（HyperpriorLatentCodec.h_a 路径）；`latent_codec.entropy_bottleneck.*` 走 `_convert_mlicpp_key` 特例分支 → `latent_codec.z.entropy_bottleneck.*`；`latent_codec.gaussian_conditional.*` 走 fanout；其它走 `_LEGACY_LIST_RENAMES` |
| 任一 era 叠加 DataParallel | `module.{any-of-above}` | `_strip_data_parallel_prefix` 先剥 `module.` |

`JiangWeibeta/MLIC` 真正 published ckpt 的命名差异留 Phase 5 验证（如有偏差再加 rename 规则）。

#### `relative_position_index` 加入 `allowed_missing` 白名单

`LocalContext.relative_position_index` 是 non-persistent buffer（构造时按 `window_size` 算一次，存在但不持久化保存），legacy ckpt 不会包含。`from_state_dict` 收集所有以 `relative_position_index` 结尾的 key，从 `incompatible_keys.missing_keys` 中扣除后再判定是否 raise。设计与 Phase 1 `LocalContext.attn_mask` non-persistent 一致。**不在白名单的任何 missing/unexpected key 仍 raise** —— 防御 silent drift。

#### `compressai/models/__init__.py` 不 re-export `MLICPlusPlus`

保持 deep-import only：`import compressai.models` 不触发 `compressai.layers.attn.swin.Mlp` 的 `[attn]` 可选依赖。user 走深路径 `from compressai.models.mlicpp import MLICPlusPlus`。zoo wiring（Phase 7）走 `_LazyImport` proxy 同样保持 lazy。

#### `decompress(strings, shape)` 类型注解已按真实 hyperprior shape 收紧

Phase 4 只覆盖 forward + state_dict round-trip，未覆盖 compress/decompress 端到端。Phase 5 追加 `TestMlicPlusPlus.test_compress_decompress_round_trip` 后确认 `HyperpriorLatentCodec.decompress` 的实际 shape 为 `{"y": List[Tuple[int, ...]], "z": Tuple[int, ...]}`，因此 `_BaseMLIC.decompress` 注解更新为 `Dict[str, Union[List[Tuple[int, ...]], Tuple[int, ...]]]`。

#### Phase 5 后续补齐项（2026-05-11 已完成）

- [x] 上游 candidate ckpt round-trip（使用 user 提供的 `candidate/MLIC/mlicpp_mse_q5_2960000.pth.tar`）
- [x] `examples/convert_mlic_checkpoint.py --variant {mlic,mlic+,mlicpp,mlicv2}` 统一 CLI wrapper（替代早期单模型 `convert_mlicpp_checkpoint.py` 设想）
- [x] compress/decompress 端到端 round-trip 测试
- [x] `JiangWeibeta/MLIC` published root-level ckpt 命名差异补到 `convert_upstream_mlicpp_state_dict`

#### LoC 数据

| 文件 | 当前 LoC |
|---|---|
| `compressai/models/mlicpp.py` | 290 |
| `tests/test_models.py::TestMlicPlusPlus`（2 测试新增段）| 149 |
| **Phase 4 净增** | **+439** |

### Phase 5：convert script + 上游 ckpt 验证（1 d）

> **2026-05-11 重排**：Phase 5 实际执行顺序推迟到 **Phase 14 之后**（与 MLICv2 published ckpt 一起做一次性批量验证）。理由：v1/v1+ 无 published ckpt，分散做没批量优势；4 个模型全部 wiring 到位再用同一 `examples/convert_mlic_checkpoint.py --variant {mlic|mlic+|mlicpp|mlicv2}` 模板覆盖。详见 [`mlic-family-reproduction.md`](../../design-docs/mlic-family-reproduction.md) §5.1.3。**任务列表保持不变**，仅执行时序调整。

- [x] **Phase 4.5 P0 修复（前置必做）**：`compressai/models/_helpers/multi_context_slice.py` 中 MLIC++ / MLIC / MLIC+ 的 `MultiContextCheckerboardLatentCodec(...)` 调用显式传 `anchor_parity="odd"`（2026-05-11 随 Phase 11 回补）。
  - **原因**：上游 `candidate/MLIC/MLIC++/utils/ckbd.py:36-39` 的 `ckbd_anchor` 把 anchor 放在 `[..., 0::2, 1::2]` + `[..., 1::2, 0::2]`（i+j **odd**）；compressai `_checkerboard_helpers.py:76-80` 的 `anchor_parity="even"` 把 anchor 放在 `[..., 0::2, 0::2]` + `[..., 1::2, 1::2]`（i+j **even**）。两套 convention 名义相反，但 ELIC / Cheng2020 等 InterDigital 自训 ckpt 跟 compressai 的 "even" 一致，因此 upstream `CheckerboardLatentCodec` 的默认值不能动；只有 MLIC 家族（v1 / v1+ / ++ / v2 共用 JiangWeibeta/MLIC `ckbd.py`）需要在应用层显式指定 `anchor_parity="odd"`。
  - **可观察症状**：当前实现 `from_state_dict` 不 fail（形状对），但 forward 数值与上游训练 ckpt 完全不匹配 —— anchor head 学到的预测被 STE 应用到错位的空间位置，likelihood 全错。
  - **回归测试**：`tests/test_models_helpers.py::TestBuildMlicppSliceCodec` 与 `TestBuildMlicSliceCodec` 断言所有 MLIC-family leaf 的 `anchor_parity == "odd"`；组合回归覆盖 MLIC++ / MLIC / MLIC+ factory forward。
  - **影响范围**：仅 `multi_context_slice.py` 单行 + 1 测试；零 ckpt convert 路径变化（原本就没 convert anchor positions）；零 ELIC 回归（独立 `CheckerboardLatentCodec` 实例继续用默认 "even"）。
- [x] 新增 `examples/convert_mlic_checkpoint.py --variant {mlic,mlic+,mlicpp,mlicv2}`（统一 MLIC-family CLI；替代早期 `convert_mlicpp_checkpoint.py` 单模型脚本设想）
- [x] `convert_upstream_mlicpp_state_dict` 处理两种 layout：
  - fork `script` `MLICPlusPlusLatentCodec` monolith → 新分层路径
  - `JiangWeibeta/MLIC` 上游作者 published ckpt（如有不同命名）→ 新路径
- [x] **上游 candidate ckpt round-trip 验证**：使用 `candidate/MLIC/mlicpp_mse_q5_2960000.pth.tar`（`state_dict` root-level published layout）→ convert → strict-load → 单图 forward → sinusoidal smoke PSNR ≥ fresh-init baseline + ε（证明权重确实参与计算，不是 dead weights）。**注意**：upstream README 提示 "use new modules" → 我们的 `LatentResidualPrediction` / `SynthesisTransform` 已对齐 post-fix 版本（详见 candidate/MLIC/MLIC++/modules/{transform,quantization} 中 New vs Old 类对比），不要误用 `LatentResidualPredictionOld`。
- [x] **额外验证项**（来自 2026-05-11 调研）：
  - `compressai/layers/lic/mlic/utils.py:67-70` `torch.meshgrid(..., indexing="ij")` 已显式 ✓（与上游 `build_position_index` 一致）
  - LocalContext `relative_position_index` 在 K 个 leaf 各自独立 lazy 算（vs 上游 `_update_local_contexts` 共享），数值等价但首次 forward 多 K-1 次计算开销；如成本不可接受，在 `build_mlicpp_slice_codec` factory 内做 mask 预热并 share buffer（Phase 9 PR description 已列为 follow-up）
  - leaf 内部 `merge` 是 cat（params packing），与上游 `ckbd_merge` 的 add（means/scales 合并到完整空间张量）非同义；但 leaf 通过 `embed` 直接写回空间张量后再走 `gaussian_conditional`，等价 ✓

#### Phase 5 验证记录

- ckpt layout inspection：`candidate/MLIC/mlicpp_mse_q5_2960000.pth.tar` 顶层 keys 为 `epoch` / `state_dict` / `loss` / `optimizer` / `aux_optimizer` / `current_step`；`state_dict` 共 1023 keys，命名为 `h_a.*` / `h_s.*` / `local_context.*` / `gaussian_conditional.*` 等 `JiangWeibeta/MLIC` published root-level layout。
- strict-load：`MLICPlusPlus.from_state_dict(sd)` 成功，推断 `N=192, M=320, slice_num=10, context_window=5`；convert 后 key 数 `1023 -> 1086`；参数量 `83,501,408`。
- CLI smoke：`.venv/bin/python examples/convert_mlic_checkpoint.py --src candidate/MLIC/mlicpp_mse_q5_2960000.pth.tar --variant mlicpp --smoke --smoke-size 64` 输出 `PSNR=38.24dB`、`y_bpp=1.0242`、`z_bpp=0.0099`、`total_bpp=1.0341`。
- fresh-init baseline 对照：loaded PSNR `38.2366`，fresh PSNR `5.0608`，delta `33.1759`。
- targeted tests：`.venv/bin/pytest tests/test_models.py::TestMlicPlusPlus tests/test_models.py::TestMlicFamily tests/test_models.py::TestMlicv2 tests/test_models_helpers.py::TestBuildMlicppSliceCodec tests/test_models_helpers.py::TestBuildMlicSliceCodec -q`：16 passed。
- MLIC family regression：`.venv/bin/pytest tests/test_mlic_layers.py tests/test_mlicv2_layers.py tests/test_models_helpers.py tests/test_multi_context_checkerboard.py tests/test_multi_context_checkerboard_selective.py tests/test_models.py::TestMlicPlusPlus tests/test_models.py::TestMlicFamily tests/test_models.py::TestMlicv2 tests/test_zoo.py::TestMlicZoo -q`：74 passed。
- static / lock / audit：`PATH=".venv/bin:$PATH" make static-analysis` passed；`uv lock --check` resolved 231 packages；`git diff --check` clean；import audit 确认 `timm_loaded=False`、`mlic_loaded=False`、`mlicv2_loaded=False`，`model_architectures["mlicpp"]` / `["mlicv2"]` 均为 `_LazyImport`。

### Phase 6：model tests（0.5 d）

- [x] `tests/test_models.py::TestMlicPlusPlus`：forward / state_dict round-trip / current layout path self-check / fork-script-legacy conversion（Phase 4 已完成；真实 published ckpt conversion smoke 已在 Phase 5 完成）
- [x] `tests/test_models.py::TestMlicPlusPlus.test_compress_decompress_round_trip`：model-level compress/decompress 端到端测试，覆盖 `strings` / `shape` contract 与 `_BaseMLIC.decompress` 类型注解
- [x] `tests/test_models_helpers.py::TestBuildMlicppSliceCodec`：factory 单元测试（Phase 3 已完成）

### Phase 7：清理 + zoo 接线（0.5 d）

- [x] `compressai/zoo/__init__.py` 加 `mlicpp` 到 `image_models`（Phase 11.5 完成）
- [x] `compressai/zoo/image.py` `mlicpp()` factory + `_LazyImport` proxy（Phase 11.5 完成；target 指向 `compressai.models.mlic.MLICPlusPlus`）
- [x] 验证 `import compressai` + `import compressai.zoo` 仍 0 timm 加载（Phase 11.5 import audit 通过）

### Phase 8：全量验证（0.5 d）

- [x] `make static-analysis` 全过（Phase 5 复验：`PATH=".venv/bin:$PATH" make static-analysis`）
- [x] `pytest tests/ -q --deselect tests/test_eval_model_video.py --deselect tests/test_zoo.py` 全过（合并到 Phase 15 联合验证；本地 macOS 仅 DDP smoke 因 rendezvous 挂起需 Linux CI 复验）
- [x] state_dict 路径自检：构造 small MLIC++，verify §state_dict 路径变化 表里所有新 key 存在 + 老 key 不存在
- [x] Import audit：`import compressai` + `import compressai.zoo` 后 optional-dep 模块保持 lazy
- [x] `uv lock --check` 一致（Resolved 231 packages）

### Phase 9：提交 + push（0.5 d）

- [ ] 按 logical 分组打 commit（建议 7-8 commits）：
  - `refactor(latent_codecs): extract checkerboard helpers to _checkerboard_helpers.py`（Phase 1 助攻段，单独 commit 让 reviewer 看清「`CheckerboardLatentCodec` 行为零变化」）
  - `feat(latent_codecs): add MultiContextCheckerboardLatentCodec sibling of CheckerboardLatentCodec`（Phase 1 主段）
  - `feat(layers/lic/mlic): lift MLIC++ application-layer building blocks`（Phase 2）
  - `feat(models/_helpers): add build_mlicpp_slice_codec factory`（Phase 3）
  - `feat(models): add MLICPlusPlus with containerized codec`（Phase 4）
  - `feat(examples): add convert_mlic_checkpoint.py`（Phase 5）
  - `chore(zoo): wire mlicpp zoo entry with lazy import`（Phase 7）
  - 测试可分摊进 Phase 1 / 4 / 7 commits 或单独 commit
- [ ] 写 PR description draft → `plan/generated/pr-mlicpp-draft.md`
  - 重点说明：(a) `MultiContextCheckerboardLatentCodec` 是 `CheckerboardLatentCodec` 的 sibling/广义化，**不动 ELIC 等现有用户**；(b) `_checkerboard_helpers.py` 是 sibling 共享的 single source of truth，未来 anchor parity 边界 bug fix 自动同步；(c) MLIC++ model 沿 ELIC pattern；(d) `compressai/layers/lic/mlic/` 是 MLIC++ application-layer building blocks，不是 codec primitive；(e) 历史 `mlicpp-latent-codec-refactor.md` 决策（model root 保留 `entropy_bottleneck` 兼容属性）已被 v2 容器化方案替代，convert script 处理 ckpt 兼容；(f) `from_state_dict` 4 个 ctor kwarg (N/M/slice_num/context_window) 全自动从 state_dict 推断 + 三个 ckpt 时代覆盖（fork-script pre-refactor / mlicpp-latent-codec-refactor.md era / DataParallel-prefixed any era）
  - **必含 「## Follow-up Recommendations」段**：列两条独立 follow-up PR：
    - **(1) ResidualBlock `act=` / `norm=` kwargs refactor**：明确建议未来独立 PR 重构 `compressai.layers.layers.ResidualBlock*` + `compressai.models.sensetime.ResidualBottleneckBlock`，加 `act: Type[nn.Module] = nn.ReLU` 关键字参数（默认 ReLU 保持现有行为不变）+ 评估 `norm: Optional[Module] = None` 抽象 GDN/IGDN/LayerNorm2d。理由：本 PR 因为不能侵入上游 layers API surface，**被迫**在 `compressai/layers/lic/mlic/transforms.py` 内放 GELU-硬编码的私有 `_ResidualBlockWithStride` / `_ResidualBlockUpsample` / `_ResidualBlock`（详见 §Phase 2 实施差异），跟 upstream ReLU 版本几乎逐行重复。Reviewer 应该意识到我们认得这个 tech debt + 提出了明确的解决方向，避免它被悄悄塞进 codebase。详见 §Phase 2 实施差异第 2 节
    - **(2) LocalContext attn_mask 跨 slice 共享**：fork-script `_update_local_contexts` 让 K=10 个 `LocalContext` 实例共享同一份 `attn_mask`（首个实例算一次，其余复用）。容器化后每个 leaf 独立 lazy 算 mask（K 次首次 forward 开销 vs fork-script 1 次）。**不影响数值正确性**（`attn_mask` 是 non-persistent buffer），但首次 forward 微 perf trade-off 应该向 reviewer 透明。可作为未来 follow-up：在 `LocalContext` 加 class-level mask cache（shape-keyed），或在 `build_mlicpp_slice_codec` factory 内显式预热第一个 leaf 的 mask 然后赋给其余 leaf
  - **Phase 5 已解决的验证点**：(a) `decompress(strings, shape)` 类型注解已按实际 hyperprior shape 收紧为 `Dict[str, Union[List[Tuple[int, ...]], Tuple[int, ...]]]`；(b) MLIC++ published ckpt strict-load + smoke + fresh-init baseline 对照已通过；(c) `downsampling_factor = 64`（`2**(4+2)` = 4 g_a stride-2 layers + 2 h_a stride-2 layers）暴露为 property 给 `compressai.utils.eval_model` 等工具算 padding，跟 fork-script behavior 一致
- [ ] push origin/pr-mlicpp（不 push upstream，等 user 决定时机）

---

## Phase 10-16（v3 新增，MLIC + MLIC+ + MLICv2 合并进同一 PR）

详见 [`mlic-family-reproduction.md`](../../design-docs/mlic-family-reproduction.md) §5.1。本节是该设计的 phase 级镜像，便于本 PR 内连贯追踪。

> **执行顺序调整（2026-05-11）**：原顺序要求 Phase 10+ 在 Phase 5 上游 ckpt round-trip 通过之后再启动。**新决策**：先推进所有 4 个 MLIC family 模型的 wiring（Phase 10 → 11 → 11.5 → 12-14），最后再统一做 ckpt convert + smoke（Phase 5 重排到 Phase 14 之后）。理由 / 风险 / 缓解详见 [`mlic-family-reproduction.md`](../../design-docs/mlic-family-reproduction.md) §5.1.3。
>
> **新执行顺序**：1-4 ✅ → 10-11 ✅ → **11.5 ✅** → 12-14 → **5 (deferred)** → 6 / 7 / 15 / 16
>
> Phase 4.5 的 `anchor_parity="odd"` 代码修复已随 Phase 11 回补；Phase 11.5 已把 MLIC++ 统一到 `_BaseMLIC` 模板（删除 `compressai/models/mlicpp.py`，MLIC / MLIC+ / MLIC++ 合到 `compressai/models/mlic.py`）。

### Phase 10：MLIC + MLIC+ 应用层 building blocks（1 d）✅ 完成于 2026-05-11

- [x] 在 `compressai/layers/lic/mlic/context.py` 追加：
  - `StackedCheckerboardConv(dim, kernel=5, num_layers=3)` —— MLIC v1 的 local context，3 层奇数 conv5×5 + GELU；输出 `(B, 2*dim, H, W)` 给 nonanchor EP head；用作 `spatial_context_nonanchor` 槽位
  - `VanillaGlobalIntraContext(dim)` —— `LinearGlobalIntraContext` 的 quadratic 版本，签名同 `(prev_y_hat, anchor_y_hat)`；MLIC / MLIC+ 用
  - `VanillaGlobalInterContext(in_dim, out_dim, num_heads)` —— `LinearGlobalInterContext` 的 quadratic 版本；MLIC+ 用
  - `WindowCheckerboardAttn(dim, window_size)` —— MLIC+ 的 overlapped window attention（vanilla softmax + mask 避免 anchor↔anchor 信息泄漏；论文 §3.4.2 + fig 7-8）；用作 `spatial_context_nonanchor` 槽位
- [x] 单元测试 `tests/test_mlic_layers.py` 追加每个新 block 的 forward shape + state_dict round-trip + checkerboard mask 正确性（`tests/test_mlic_layers.py` 19/19 通过）

### Phase 10 实施差异 / 决策记录

- `WindowCheckerboardAttn` 继承现有 `LocalContext`：MLIC+ 的 overlapped window checkerboard attention 与 Phase 2 已 lift 的 MLIC++ `LocalContext` 在当前复现范围内共享同一 window attention + checkerboard mask contract；保留单独 class 名，便于 Phase 11 factory 用模型语义注入。
- `VanillaGlobalIntraContext` / `VanillaGlobalInterContext` 使用 quadratic scaled dot-product attention，保留现有 linear global context 的输入/输出 contract 与 conv/MLP refinement pattern。Intra 版本显式构造 nonanchor-query × anchor-key 的 checkerboard attention mask，并默认屏蔽局部半径 2 的 key 以避免退化成 local context。
- `StackedCheckerboardConv` 校验 `kernel` / `num_layers` 必须为正奇数，匹配论文中 odd-layer 信息传递要求；最后一层无激活，输出 `2*dim` 通道。
- `compressai/layers/lic/mlic/__init__.py` deep-import 暴露四个新 block；仍不从 `compressai/layers/__init__.py` re-export，保持 `[attn]` 可选依赖 lazy。
- 执行顺序备注：Phase 10 按 user 指令先落地 application-layer blocks；Phase 4.5 的 `anchor_parity="odd"` 代码修复已在 Phase 11 回补；MLIC++ published ckpt smoke 已在 Phase 5 完成。

### Phase 11：MLIC + MLIC+ slice factory + thin model（1.5 d）✅ 完成于 2026-05-11

- [x] 在 `compressai/models/_helpers/multi_context_slice.py` 追加 `build_mlic_slice_codec(*, variant: Literal["mlic", "mlic+"], M, slice_num, slice_ch, ...)` factory；复用 90% `_MlicppPriorAggregation` / `_MlicppSideLayout` 基础设施，只换 context block 注入（`LocalContext`→`StackedCheckerboardConv` 或 `WindowCheckerboardAttn`；`LinearGlobalIntra/InterContext`→`VanillaGlobalIntra/InterContext`）
- [x] `compressai/models/mlic.py` 含 `MLIC` + `MLICPlus` 两 class（共享 thin model 模板，沿 STF/TCM pattern）；`from_state_dict` 简化（无 fork-script monolith 兼容，因为没有官方 v1/v1+ ckpt）
- [x] zoo 接入：`mlic` / `mlicplus` lazy factory（`_LazyImport` proxy）；`pretrained=True` raise `RuntimeError`（无 published ckpt）
- [x] 单元测试 `tests/test_models.py::TestMlicFamily` + `tests/test_models_helpers.py::TestBuildMlicSliceCodec` + `tests/test_zoo.py::TestMlicZoo`：forward + state_dict round-trip + factory 参数校验 + state_dict 路径自检 + zoo lazy factory

### Phase 11 实施差异 / 决策记录

- `build_mlic_slice_codec` 采用单一 `variant: Literal["mlic", "mlic+"]` factory，而不是拆 `build_mlic_slice_codec` / `build_mlicplus_slice_codec` 两个函数。原因是两者 side layout、LRP input layout、channel context 和 entropy-parameter head wrapper 完全共享，仅 local/global context 注入不同；单 factory 更容易锁定两模型的共同 contract。
- MLIC 默认构造参数按论文 Table 2：`N=192, M=192, slice_num=6`；MLIC+ 默认 `N=192, M=320, slice_num=10`。测试使用 tiny config 覆盖 forward / state_dict round-trip。
- `MLIC` 使用 `StackedCheckerboardConv` + `VanillaGlobalIntraContext`，没有 `global_inter_part`；`MLICPlus` 使用 `WindowCheckerboardAttn` + `VanillaGlobalIntraContext` + `VanillaGlobalInterContext`。测试显式断言 `MLIC` 不产生 `channel_context.y{k}.global_inter_part.*` 路径。
- Phase 4.5 的 `anchor_parity="odd"` 修复随本 phase 一起落地：`build_mlic_slice_codec` 的 MLIC / MLIC+ / MLIC++ variants 都显式传入 odd parity，避免继承 CompressAI 默认 even convention。真实 MLIC++ published ckpt round-trip 已在 Phase 5 完成。
- `compressai.models.__init__` 仍不 re-export `MLIC` / `MLICPlus`；用户和 zoo 都走 deep import / lazy import，保持 `[attn]` optional dependency lazy。

### Phase 11 验证记录

- `env PATH=.venv/bin:$PATH pytest tests/test_models_helpers.py::TestBuildMlicSliceCodec tests/test_models_helpers.py::TestBuildMlicppSliceCodec tests/test_models.py::TestMlicFamily tests/test_models.py::TestMlicPlusPlus tests/test_zoo.py::TestMlicZoo -q`：17 passed。
- `env PATH=.venv/bin:$PATH pytest tests/test_mlic_layers.py tests/test_models_helpers.py tests/test_multi_context_checkerboard.py tests/test_models.py::TestMlicFamily tests/test_models.py::TestMlicPlusPlus tests/test_zoo.py::TestMlicZoo -q`：57 passed。
- `env PATH=.venv/bin:$PATH make static-analysis`：ruff format / import order / lint 全过。
- `git diff --check`：clean。
- Import audit：`import compressai` / `import compressai.models` / `import compressai.zoo` 后 `timm` 均未加载；deep import `from compressai.models.mlic import MLIC, MLICPlus` 可用，且 factory leaf `anchor_parity == "odd"`。

### Phase 11.5：MLIC++ 统一到 `_BaseMLIC` 模板（0.5 d）

> **目标**：4 个 MLIC family 模型共一个文件 `compressai/models/mlic.py`，共一个 `_BaseMLIC` 模板，共一个 `build_mlic_slice_codec(variant=...)` factory。删除 `compressai/models/mlicpp.py`（不留 shim，import 路径 breaking change：`from compressai.models.mlicpp import ...` → `from compressai.models.mlic import ...`）。详细设计 + 风险评估见 [`mlic-family-reproduction.md`](../../design-docs/mlic-family-reproduction.md) §5.1.2。

#### 任务

- [x] **factory 扩展**：`build_mlic_slice_codec(variant=Literal["mlic","mlic+","mlicpp"])` 加 `"mlicpp"` 分支：
  - local context: `LocalContext(dim, window_size)`（与 `WindowCheckerboardAttn` 是同一类）
  - global inter factory: `_build_linear_global_inter_context`
  - intra wrapper: `_MlicppIntraWrapper(LinearGlobalIntraContext)`
- [x] **删除** `build_mlicpp_slice_codec`（test 调用点改 `build_mlic_slice_codec(variant="mlicpp")`）
- [x] **`_BaseMLIC` 加 `_legacy_convert: Optional[Callable] = None` hook**：`from_state_dict` 在 N/M/slice_num 推断前调 `cls._legacy_convert(state_dict)` 如果 set
- [x] **`MLICPlusPlus` 内联到 `compressai/models/mlic.py`**：
  - `class MLICPlusPlus(_BaseMLIC): _variant = "mlicpp"; _legacy_convert = staticmethod(convert_upstream_mlicpp_state_dict)`
  - `__init__` 默认 `N=192, M=320, slice_num=10, context_window=5`
  - `__all__` 加 `MLICPlusPlus, convert_upstream_mlicpp_state_dict`
- [x] **`convert_upstream_mlicpp_state_dict` + `_CURRENT_SLICE_RE` + `_LEGACY_SLICE_RE` + `_ROOT_TO_CONTAINER_PREFIXES` + `_LEGACY_LIST_RENAMES` + `_strip_data_parallel_prefix` + `_convert_mlicpp_key`** 整体从 `compressai/models/mlicpp.py` 搬到 `compressai/models/mlic.py`
- [x] **`_infer_context_window`（mlic.py 中已有）**：核对 mlicpp.py 版本与 mlic.py 版本是否完全等价；若有差异（mlicpp.py 多 `relative_position_table` fallback 分支）合并到 mlic.py 共用
- [x] **删除文件** `compressai/models/mlicpp.py`
- [x] **zoo 更新**：
  - `compressai/zoo/image.py::model_architectures`：`"mlicpp": _LazyImport("compressai.models.mlicpp", "MLICPlusPlus")` → `_LazyImport("compressai.models.mlic", "MLICPlusPlus")`
  - `compressai/zoo/image.py::mlicpp(...)` 函数体内 `from compressai.models.mlicpp import MLICPlusPlus` → `from compressai.models.mlic import MLICPlusPlus`
- [x] **测试 import 改**：
  - `tests/test_models.py::TestMlicPlusPlus`：`from compressai.models.mlicpp import MLICPlusPlus, convert_upstream_mlicpp_state_dict` → `from compressai.models.mlic import MLICPlusPlus, convert_upstream_mlicpp_state_dict`
  - `tests/test_models_helpers.py::TestBuildMlicppSliceCodec`：`build_mlicpp_slice_codec(...)` → `build_mlic_slice_codec(variant="mlicpp", ...)`；保持所有 path 断言不变
- [x] **回归验证**：
  - Phase 4 `TestMlicPlusPlus.test_forward_and_state_dict_round_trip` + `test_legacy_state_dict_conversion` 全过（验证 wiring + legacy convert 零回归）
  - Phase 11 `TestMlicFamily` + `TestBuildMlicSliceCodec` + `TestMlicZoo` 全过
  - 组合回归 `pytest tests/test_mlic_layers.py tests/test_models_helpers.py tests/test_multi_context_checkerboard.py tests/test_models.py::TestMlicFamily tests/test_models.py::TestMlicPlusPlus tests/test_zoo.py::TestMlicZoo -q` 不引入新失败
  - `make static-analysis` 三步全过
  - Import audit：`import compressai.zoo` 触发 0 timm 加载；`from compressai.models.mlic import MLICPlusPlus, MLIC, MLICPlus` 全部可 deep-import；`from compressai.models.mlicpp import ...` **应该报 ModuleNotFoundError**（验证 shim 没误留）
- [ ] **回滚 tag**：`git tag pr-mlicpp-pre-unify HEAD` 在 Phase 11.5 commit 之前，万一 Phase 12+ 发现问题可定向回退。**2026-05-11 未执行**：当前 Phase 1-11.5 仍是未提交工作树改动，给 `HEAD` 打 tag 不能覆盖 pre-unify 工作树状态；待 commit 分组前再按实际 commit 边界补 tag / backup。

#### Phase 11.5 验证记录

- `env PATH=.venv/bin:$PATH pytest tests/test_models_helpers.py::TestBuildMlicppSliceCodec tests/test_models_helpers.py::TestBuildMlicSliceCodec tests/test_models.py::TestMlicPlusPlus tests/test_models.py::TestMlicFamily tests/test_zoo.py::TestMlicZoo -q`：17 passed。
- `env PATH=.venv/bin:$PATH pytest tests/test_mlic_layers.py tests/test_models_helpers.py tests/test_multi_context_checkerboard.py tests/test_models.py::TestMlicFamily tests/test_models.py::TestMlicPlusPlus tests/test_zoo.py::TestMlicZoo -q`：57 passed。
- `env PATH=.venv/bin:$PATH make static-analysis`：ruff format / import order / lint 全过。
- `git diff --check`：clean。
- `env PATH=.venv/bin:$PATH uv lock --check`：resolved 231 packages，lockfile 一致。
- Import audit：`import compressai` / `import compressai.models` / `import compressai.zoo` 后 `timm` 均未加载；`from compressai.models.mlic import MLIC, MLICPlus, MLICPlusPlus` 可用；`compressai.models.mlicpp` spec 为 `None`。

#### LoC 影响

- `mlicpp.py` 删 -290
- `mlic.py` +170（`MLICPlusPlus` 类 ~30 + convert helpers + regex ~140）
- `multi_context_slice.py` 净 -40（删 `build_mlicpp_slice_codec` -50 + 加 `variant="mlicpp"` 分支 +10）
- 测试 imports 改 ±0
- **Phase 11.5 净 -160 LoC**

#### 风险（详见 design doc §5.1.2）

| 风险 | 严重度 | 缓解 |
|---|---|---|
| Linear vs Vanilla module submodule 名 / shape 差异 | 低 ✅ 已验证 | grep 验证 `_pointwise_then_dwconv` / `reprojection` / `mlp` 等 submodule 名 + shape 完全一致 |
| 删 `mlicpp.py` 后 user 代码 breaking | 低 | mlicpp.py 没 push origin（仅本地 pr-mlicpp 分支），无下游依赖；upstream 也没合 |
| `TestMlicPlusPlus.test_legacy_state_dict_conversion` 跑挂 | 低 | `convert_upstream_mlicpp_state_dict` 函数体不变，只换 import 来源；test 内部断言 path-string 也不变 |
| zoo `_LazyImport` 还引用旧 path | 低 | task 列表显式列了改 zoo 这条；CI 跑 `TestMlicppZoo.test_factories` 校验 |

#### 与 Phase 12-14 的关系

Phase 14 完成后，4 个 MLIC family 共用 `_BaseMLIC` 模板：MLIC / MLIC+ / MLIC++ / MLICv2。`build_mlic_slice_codec` 已支持第 4 个 variant 路径 `variant="mlicv2"`。MLICv2 不新建 `compressai/models/mlicv2.py`，直接加 `class MLICv2(_BaseMLIC): _variant = "mlicv2"` 到 `mlic.py`。

### Phase 12：MLICv2 leaf hook（0.5 d）

- [x] `MultiContextCheckerboardLatentCodec` 加可选 `selective_predictor: Optional[nn.Module]` 构造参数 + `_selective_checkerboard.py` 内部 helper（含 `apply_selective_compression` / `apply_selective_decompression`）
- [x] forward / compress / decompress 三路径：默认 `None` 时与 Phase 1 行为完全等价；启用时，对每个位置预测 `s ∈ {0,1}`，`s==1` 走正常 arithmetic coding，`s==0` 跳过且 y_hat 填 means
- [x] 新增 `tests/test_multi_context_checkerboard_selective.py`：`test_selective_predictor_skip_semantics`（启用时压缩比 vs 不启用对比）+ `test_selective_predictor_none_is_identity`（None 时与 Phase 1 测试结果数值一致）

#### Phase 12 验证记录

- `env PATH=.venv/bin:$PATH pytest tests/test_multi_context_checkerboard.py tests/test_multi_context_checkerboard_selective.py -q`：10 passed。
- `env PATH=.venv/bin:$PATH pytest tests/test_mlic_layers.py tests/test_models_helpers.py tests/test_multi_context_checkerboard.py tests/test_multi_context_checkerboard_selective.py tests/test_models.py::TestMlicFamily tests/test_models.py::TestMlicPlusPlus tests/test_zoo.py::TestMlicZoo -q`：59 passed。
- `env PATH=.venv/bin:$PATH make static-analysis`：ruff format / import order / lint 全过。
- `git diff --check`：clean。
- `uv lock --check`：Resolved 231 packages，lockfile 一致。
- Import audit：`import compressai` / `import compressai.models` / `import compressai.zoo` 后 `timm_loaded_after_lazy_imports=False`；`from compressai.models.mlic import MLIC, MLICPlus, MLICPlusPlus` 正常；`compressai.models.mlicpp` spec 为 `None`。

### Phase 13：MLICv2 子包 `compressai/layers/lic/mlicv2/`（2 d）

- [x] 新建 `compressai/layers/lic/mlicv2/{__init__.py,transforms.py,context.py}`
- [x] `transforms.py`：`SimpleTokenMixing(dim)`（LN + DepthRB + DWConv5×5 + Conv1×1 + Gate；论文 §3.3 + eq 6）；`STMAnalysis` / `STMSynthesis`（g_a / g_s 用 STM 替 ResidualBlock，其他 stage 跟 MLIC++ 一致）
- [x] `context.py`：
  - `HGCPModule(M, slice_ch)` —— slice 0 用 hyperprior anchor/nonanchor 互注意力预测 anchor-side global context（论文 §3.4.1 + eq 7）；放进 leaf 的 `spatial_context_anchor` 槽位（仅 k==0）
  - `ContextReweighting(dim)` —— channel-wise softmax(Q·K)·V + Gate（论文 §3.4.2 + fig 5 + eq 8）；wrap 在 `LinearGlobalIntra/InterContext` 后
  - `RoPE2D(dim, learnable_thetas=True)` —— 2D Rotary Position Embedding（论文 §3.4.3 + eq 9-10）；θ_x = θ_y = 10000 可学；替代 LocalContext / GlobalContext 里的 `relative_position_index` + RPE bias
  - `GSCModule(slice_ch, threshold=0.3)` —— 后训练 skip map 预测器（论文 §3.4.4 + eq 11-13）；输入 scale + prev slices + hyperprior；输出 sigmoid 二分类
- [x] 单元测试 `tests/test_mlicv2_layers.py`：每个 block 的 forward shape + state_dict round-trip；关键不变量（RoPE 位置可比性、CR channel attention 形状、HGCP slice-0 anchor context shape、GSC skip-rate 在合理范围）

#### Phase 13 验证记录

- `env PATH=.venv/bin:$PATH pytest tests/test_mlicv2_layers.py -q`：11 passed。
- `env PATH=.venv/bin:$PATH pytest tests/test_mlic_layers.py tests/test_mlicv2_layers.py tests/test_models_helpers.py tests/test_multi_context_checkerboard.py tests/test_multi_context_checkerboard_selective.py -q`：63 passed。
- `env PATH=.venv/bin:$PATH make static-analysis`：ruff format / import order / lint 全过。
- `git diff --check`：clean。
- Import audit：`import compressai` / `import compressai.layers` / `import compressai.layers.lic` 后 `mlicv2_loaded=False` 且 `timm_loaded=False`；安装 `[attn]` 后 deep import `from compressai.layers.lic.mlicv2 import SimpleTokenMixing, HGCPModule` 正常并加载 `timm`。

### Phase 14：MLICv2 model + factory（1.5 d）

- [x] `compressai/models/_helpers/multi_context_slice.py` 追加 `variant="mlicv2"` factory 路径：
  - 注入 `HGCPModule` 到 k==0 leaf 的 `spatial_context_anchor` 槽位
  - 包 `ContextReweighting` + `RoPE2D` 在现有 `LinearGlobalIntra/InterContext` 外（应用层 wrapper，不改 MLIC++ 既有 module）
  - 注册 `GSCModule` 到所有 K 个 leaf 的 `selective_predictor` 槽位
- [x] `compressai/models/mlic.py` 加 `class MLICv2(_BaseMLIC): _variant = "mlicv2"`（**不**新建 `compressai/models/mlicv2.py`；4 个 MLIC family 模型全部内联到同一 `mlic.py`，与 Phase 11.5 决策一致）；用 `STMAnalysis` / `STMSynthesis` 替 `AnalysisTransform` / `SynthesisTransform`
- [x] zoo 接入：`mlicv2` lazy factory；`pretrained=True` raise（无 published ckpt）
- [x] 单元测试 `tests/test_models.py::TestMlicv2`：forward + state_dict round-trip + GSC skip-rate sanity check + state_dict 路径自检（关键路径含 `latent_codec.y.latent_codec.y0.spatial_context_anchor.*` 验证 HGCP 注入正确）

执行记录：
- `tests/test_models_helpers.py::TestBuildMlicv2SliceCodec`：2 passed。
- `tests/test_models.py::TestMlicv2`：1 passed。
- `tests/test_zoo.py::TestMlicZoo`：3 passed。
- MLIC family targeted regression（layers + helpers + leaf + model + zoo）：73 passed。
- `make static-analysis`：passed。
- `git diff --check`：clean。
- `uv lock --check`：resolved 231 packages。
- Import audit：`import compressai` + `import compressai.zoo` 后 `timm_loaded=False`、`mlic_loaded=False`、`mlicv2_loaded=False`；`model_architectures["mlicv2"]` 是 `_LazyImport`；4 个 thin model `downsampling_factor == 64`。

### Phase 15：联合验证（0.5 d）

- [x] `make static-analysis` 全过
- [x] `pytest tests/ -q --deselect tests/test_eval_model_video.py --deselect tests/test_zoo.py` 全套不回归（含新增 MLIC v1 / MLIC+ / MLICv2 测试）
- [x] Import audit：`import compressai` + `import compressai.zoo` 仍 0 timm 加载 + 0 mlicv2 子包加载（deep-import only）
- [x] `uv lock --check` 一致（本 PR 不引入新 hard / optional dependency）
- [x] 4 个 thin model 的 `downsampling_factor` property 自检（MLIC / MLIC+ / MLIC++ / MLICv2 都 = 64）

#### Phase 15 验证记录

- `PATH=".venv/bin:$PATH" make static-analysis`：ruff format / import order / lint 全过。
- `uv lock --check`：resolved 231 packages。
- Import audit：`import compressai` + `import compressai.zoo` 后 `timm_loaded=False`、`compressai.models.mlic_loaded=False`、`compressai.layers.lic.mlic_loaded=False`、`compressai.layers.lic.mlicv2_loaded=False`；`model_architectures["mlic"]` / `["mlicplus"]` / `["mlicpp"]` / `["mlicv2"]` 均为 `_LazyImport`。
- 4 个 thin model 自检：`MLIC.downsampling_factor=64`、`MLICPlus.downsampling_factor=64`、`MLICPlusPlus.downsampling_factor=64`、`MLICv2.downsampling_factor=64`。
- MLIC family targeted regression：`.venv/bin/pytest tests/test_mlic_layers.py tests/test_mlicv2_layers.py tests/test_models_helpers.py tests/test_multi_context_checkerboard.py tests/test_multi_context_checkerboard_selective.py tests/test_models.py::TestMlicPlusPlus tests/test_models.py::TestMlicFamily tests/test_models.py::TestMlicv2 tests/test_zoo.py::TestMlicZoo tests/test_sga.py -q`：88 passed, 1 warning。
- Broad regression（本地 macOS/Codex 非 sandbox，排除计划中的 video/zoo 以及本机会挂起的 DDP smoke）：`.venv/bin/pytest tests/ -q --deselect tests/test_eval_model_video.py --deselect tests/test_zoo.py --deselect tests/test_train.py::test_train_example_ddp`：286 passed, 4 skipped, 36 deselected, 1 warning。
- 原计划 broad regression 的本地限制：sandbox 内原命令得到 283 passed / 4 skipped / 35 deselected / 4 failed，失败均为环境限制：`torch_shm_manager ... Operation not permitted`（DataLoader worker shared memory）与 `torch.distributed` localhost rendezvous timeout；非 sandbox 原命令通过 shared-memory 失败点后停在 `tests/test_train.py::test_train_example_ddp` 的 macOS `torch.distributed.run --standalone` rendezvous，已手动中止。该 DDP smoke 与 MLIC family 改动无关，需在 Linux CI/maintainer 环境复验。
- `git diff --check`：clean。

### Phase 16：提交 + push（0.5 d）

- [x] 本地 commit 已完成（按当前 review 粒度压成 2 个逻辑 commit，文档未纳入提交）：
  - `b0924fc feat(models): add mlic family` — MLIC / MLIC+ / MLIC++ / MLICv2、`MultiContextCheckerboardLatentCodec`、MLICv2 selective hook、MLIC-family SGA refine API、convert/refine examples、tests
  - `9cdcb05 feat(latent_codecs): generalize sga quantization` — `GaussianConditionalLatentCodec` / `CheckerboardLatentCodec` SGA quantizer hook、`LRPGaussianLatentCodec` inheritance coverage、`tests/test_sga.py`
- [x] push 当前分支到 remote（`origin/pr-mlicpp` 已在 `9cdcb05`）
- [x] 更新 PR description draft `plan/generated/pr-mlicpp-draft.md`：
  - 标题改为「Add MLIC family (MLIC + MLIC+ + MLIC++ + MLICv2) with `MultiContextCheckerboardLatentCodec` abstraction」
  - 重点说明四模型共抽象的设计动机 + 上游 ckpt 验证只对 MLIC++ 一档（v1/v1+/v2 fresh-init only）+ leaf 抽象 7 个可选 hook（5 个原 + `selective_predictor` 新 + `lrp_activation`）的可选语义
  - **Phase 11.5 unify 段**：解释「MLIC++ 从独立 `mlicpp.py` 内联到 `mlic.py`，4 个 family 模型共一个 `_BaseMLIC` 模板」是为了 single source of truth；`from compressai.models.mlicpp import ...` breaking 但 mlicpp.py 没 push 到 origin 也未 upstream，无下游影响
  - **Phase 5 重排段**：解释「ckpt smoke 集中在 4 模型全部 wiring 到位后做」+ `examples/convert_mlic_checkpoint.py --variant {mlic|mlic+|mlicpp|mlicv2}` 单脚本覆盖 4 个 model
  - 若 reviewer 要求拆 PR，建议拆分边界：**Phase 11.5 / Phase 12 之间**（MLIC + MLIC+ + MLIC++ self-contained PR，MLICv2 作为 follow-up PR 复用同一抽象 + 新加 `selective_predictor` hook）—— 比初版「Phase 9 / Phase 10 之间」更优，unify 后边界更干净

### Phase 17：MLICv2+ — SGA 推理时 latent re-optimization（实际 ~0.4 d，已落地）✅ 完成于 2026-05-11

`Improving Inference for Neural Image Compression` (Yang, Bamler, Mandt, NeurIPS 2020, [arxiv:2006.04240](https://arxiv.org/abs/2006.04240)) 的 Stochastic Gumbel Annealing；MLICv2 论文 §3.5 把它套到 v2 上得到 v2+。本 Phase 把 SGA 作为通用 inference utility 落到 ops + codec quantizer hook + model refine API。

- [x] **`compressai/ops/sga.py`**（132 行含 license header）—— `SGAQuantizer(nn.Module)`：`set_iter(it, total_iter)` 设置退火状态；状态未设置时 fallback `torch.round`；启用后用 `RelaxedOneHotCategorical` 在 floor/ceil 之间采样，温度 `T = 0.5 * exp(-1e-3 * (it - t0))`、`t0 = 0.35 * total_iter`。`compressai/ops/__init__.py` 导出 `SGAQuantizer`，与 `quantize_ste` 同级
- [x] **`EntropyBottleneckLatentCodec` 加 `quantizer="sga"` 模式**（净 +27 LoC）—— 新增 `sga: Optional[SGAQuantizer]` ctor 参数；当 `quantizer="sga"` 时 forward 走 `_likelihood_for_quantized(y_hat_sga)` —— 绕过 `entropy_bottleneck()` 的 noise/dequantize 路径直接调 `_likelihood`，让 rate term 沿 SGA-quant y_hat 反传梯度。默认 `"noise"` 行为零变化
- [x] **`MultiContextCheckerboardLatentCodec` 加 `quantizer="sga"` hook**（净 +18 LoC）—— `quantizer: str = "ste"` + `sga: Optional[SGAQuantizer]` ctor 参数；新增 `_quantize` helper 替代直接 `quantize_ste(...)`；新增 `_likelihood_for_quantized` 在 SGA mode 下绕过内部 `GaussianConditionalLatentCodec` 的 noise quant，调 `gaussian_conditional._likelihood(y_hat_sga, scales, means)`
- [x] **`_BaseMLIC` 加 refine API**（净 +56 LoC）—— `refine_extract(x) -> (y, z)` 跑一次 `g_a + h_a` 取初始 latent；`refine_forward(y, z) -> {x_hat, likelihoods}` 跳过 `g_a/h_a` 接 HyperpriorLatentCodec 内部接力（z_codec → h_s → y_codec → g_s），让 caller 把 y/z 包成 nn.Parameter 直接 backprop；`set_sga_mode(sga | None)` 把内部 z_codec + 全部 K 个 y leaves 的 quantizer 切到 SGA（共享同一 `SGAQuantizer` 实例），`None` 时 revert 到默认 noise/STE
- [x] **`examples/refine_with_sga.py` CLI**（180 LoC）—— 加载 MLIC family ckpt + 单图 → SGA refine（默认 total_iter=2000, lr=5e-3, lambda=0.025）→ 打印 init vs post 的 bpp/PSNR
- [x] **`tests/test_sga.py`**（14 测试全过）：
  - SGAQuantizer：fallback round / set_iter+reset / 梯度可微 / 温度退火
  - EntropyBottleneckLatentCodec(sga)：iter unset 时 round 等价 / iter set 时梯度流 / invalid quantizer 报错 / sga without module 报错
  - MLIC refine：set_sga_mode 把所有 leaf 与 z_codec 切到同一 SGA 实例 / set_sga_mode(None) 还原 / refine_extract 形状 / refine_forward 在 SGA 关时数值与 forward(x) 一致 / 50 iter SGA refine loop 在 MLIC++ 上 RD loss 真实下降
  - **MLICv2 仅接口测试**（不验证 grad）：fresh-init MLICv2 的 GSCModule 输出全 False mask（GSC 没训过）→ `apply_selective_y_hat` 把 y_hat 全替成 means → 解耦于 y。这是 v2 fresh-init 的预期行为；trained ckpt 上 GSC 输出 mixed mask 时 SGA 梯度会正常流。PR description 须显式说明这点
- [x] 关键设计决策：(a) SGA 作为 quantizer 选项放进 codec（与 `quantize_ste` / "noise" 同位），不走 monkey-patch；(b) SGA module 全 codec 共享同一实例，set_iter 一次全员同步；(c) refine_forward 复用 model 的 latent_codec 内部接力，不复制实现；(d) MLICv2+ ≈ MLICv2 + 调用 refine_with_sga.py，没有新 model class

**Phase 17 净增**：~+460 LoC（ops/sga.py 132 + entropy_bottleneck.py +27 + multi_context_checkerboard.py +18 + mlic.py +56 + examples/refine_with_sga.py 180 + tests/test_sga.py 230 - import 调整 ~10）

**Layer A 泛化完成记录**：独立执行计划 [`sga-codec-generalization.md`](../completed/sga-codec-generalization.md) 已完成并移入 completed，commit `9cdcb05` 把 `quantizer="sga"` 推广到 `GaussianConditionalLatentCodec` / `CheckerboardLatentCodec`，并通过父类继承覆盖 `LRPGaussianLatentCodec`。上游无对应 ELIC checkpoint 可做 ckpt 数值回归，因此完成标准改为 targeted tests + default STE 行为不变；`.venv/bin/python -m pytest tests/test_sga.py -q`：25 passed, 1 warning。

---

## 关键风险

| 风险 | 严重度 | 缓解 |
|---|---|---|
| `MultiContextCheckerboardLatentCodec` 抽象失败 / 无第二消费者 → 看起来过度设计 | 中 | (a) 设计上预留 `spatial_context_nonanchor` 可被 VSS Mamba block 替代的 hook，留 MambaIC 复用空间；(b) 即便短期只有 MLIC++ 一个用户，sibling leaf 比 monolith 仍更 pedagogical，符合 user 教学需求；(c) PR description 显式说明这是「shared between MLIC++ now, MambaIC reconsidered in PR-3」 |
| 上游 reviewer 反对新增 sibling leaf（觉得跟 `CheckerboardLatentCodec` 重复）| 中 | 准备好 fallback：把 sibling 的 optional hooks 加到 upstream `CheckerboardLatentCodec` 作为可选 kwargs（默认全 None 时 100% 等价旧行为）；这是 fewer-files 但 wider-API-surface 的折衷，等 reviewer 真有反对意见再切 |
| `_MlicppPriorAggregation` 应用层 helper 把 `ChannelContext + LinearGlobalInterContext` 强行拼成单 `nn.Module` 显得 ad-hoc | 低 | 跟 ELIC `channel_context` 字典里的 module 同 contract（plain `nn.Module`，无新 abstract class），ELIC 类似情况也用 inline 工厂；reviewer 应能接受 |
| `compressai/latent_codecs/checkerboard.py` 内部 helper 暴露给 sibling 用 | ~~低~~ ✅ | **Phase 1 已解决**：`_checkerboard_helpers.py` 抽 7 个 module-level 纯函数（`embed` / `unembed` / `mask_all` / `mask_all_but_step` / `merge` / `write_step` / `step_parity`），两 codec 都 import；`CheckerboardLatentCodec` 公开 API（`anchor_parity` / `non_anchor_parity` attr，`_y_ctx_zero` / `quantize` method）零变化；ELIC / Cheng2020 等用户全套测试无回归 |
| **anchor_parity 与上游 MLIC++ ckbd 约定反向** | ~~高~~ ✅ | **Phase 4.5 代码修复已随 Phase 11 回补，Phase 5 真实 ckpt smoke 已通过**：上游 `ckbd_anchor` 用 i+j odd 位置作 anchor，compressai default "even" 用 i+j even 位置；`build_mlic_slice_codec` 的 MLIC / MLIC+ / MLIC++ variants 已显式传 `anchor_parity="odd"`。ELIC 等用 upstream `CheckerboardLatentCodec` 的模型不受影响（保持默认 "even"，对齐 InterDigital 自训 ckpt convention）。|
| **现有 fork `script` `MLICPlusPlusLatentCodec` 仓内被本仓的训出 ckpt（如有）失效** | 低 | 本仓 fork `script` 上 MLIC++ 本来就没在 origin 用，convert script 处理路径迁移；无生产 ckpt 受影响 |

---

## 总时间估算

| Phase | 工时 | 状态 |
|---|---|---|
| Phase 0 | 30 min | 部分完成（开发期 cherry-pick 已就位）|
| Phase 1 | 1.5 d → 实际 ~0.7 d（抽象设计 + 新 leaf + helper 抽出 + 8 测试 + Bug A/LRP contract 修复 + 全套回归）| ✅ 完成于 2026-05-10 |
| Phase 2 | 1 d → 实际 ~0.3 d（mlic 子包 + GELU residual 适配 + 14 测试）| ✅ 完成于 2026-05-10 |
| Phase 3 | 0.5 d → 实际 ~0.3 d（slice factory + side layout wrappers + 5 测试 + LRP current-slice 修正）| ✅ 完成于 2026-05-10 |
| Phase 4 | 1 d → 实际 ~0.2 d（thin model + current/legacy state_dict tests）| ✅ 完成于 2026-05-10 |
| Phase 5 | 1 d → 实际 ~0.2 d | ✅ 完成于 2026-05-11（统一 `examples/convert_mlic_checkpoint.py --variant {mlic,mlic+,mlicpp,mlicv2}`；MLIC++ 真实 ckpt strict-load + smoke + fresh-init baseline 对照；Phase 4.5 anchor_parity 修复已随 Phase 11 回补）|
| Phase 6 | 0.5 d | ✅ 完成于 2026-05-11（Phase 4 / 11 / 14 各自加 model forward / state_dict / legacy mapping tests；Phase 5 追加 MLIC++ model-level compress/decompress round-trip）|
| Phase 7 | 0.5 d | ✅ 完成于 2026-05-11（mlic / mlicplus / mlicpp / mlicv2 zoo lazy factories；无 hosted weights 的 pretrained=True 均 raise）|
| Phase 8 | 0.5 d | 与 Phase 15 合并 |
| Phase 9 | 0.5 d | 与 Phase 16 合并 |
| **MLIC++ Track 小计** | **~3.5 d** | Phase 1-8 + Phase 5/6/15 已落地；剩余只剩与四模型共享的 Phase 16 |
| Phase 10 | 1 d → 实际 ~0.2 d（MLIC / MLIC+ application-layer blocks + 5 个新增 layer 测试）| ✅ 完成于 2026-05-11 |
| Phase 11 | 1.5 d → 实际 ~0.4 d（MLIC + MLIC+ slice factory + thin model + zoo lazy factory + tests）| ✅ 完成于 2026-05-11 |
| **Phase 11.5** | **0.5 d → 实际 ~0.2 d** | ✅ 完成于 2026-05-11（MLIC++ 统一到 `_BaseMLIC` 模板：删 `mlicpp.py`、扩展 `build_mlic_slice_codec(variant="mlicpp")`、MLIC / MLIC+ / MLIC++ 同文件 `mlic.py`；rollback tag 因未提交工作树状态暂缓）|
| Phase 12 | 0.5 d → 实际 ~0.2 d | ✅ 完成于 2026-05-11（leaf `selective_predictor` hook + forward/compress/decompress skip semantics + 2 个测试）|
| Phase 13 | 2 d → 实际 ~0.4 d | ✅ 完成于 2026-05-11（MLICv2 子包：STM + HGCP + CR + 2D RoPE + GSC + 11 个 layer 测试）|
| Phase 14 | 1.5 d → 实际 ~0.3 d | ✅ 完成于 2026-05-11（MLICv2 `variant="mlicv2"` factory + HGCP/CR/RoPE/GSC 接线 + `MLICv2` thin model + zoo lazy factory + tests）|
| Phase 15 | 0.5 d → 实际 ~0.5 d | ✅ 完成于 2026-05-11（静态检查 / lock / import audit / downsampling / MLIC targeted regression / broad regression 已过；本地 macOS 仅 DDP smoke 因 rendezvous 挂起需 Linux CI 复验）|
| Phase 16 | 0.5 d | ✅ 完成于 2026-05-12（本地 commits `b0924fc` / `9cdcb05` 已创建且 `origin/pr-mlicpp` 已在 `9cdcb05`；PR draft 写入 `plan/generated/pr-mlicpp-draft.md`；实际打开 upstream PR 等 user 时机）|
| **v1 + v1+ + v2 Track 小计** | **~7.5 d** | Phase 10-16 已落地；实际打开 upstream PR 等 user 时机 |
| **PR 总剩余** | **按 user 时机** | 实际打开 upstream PR |

比 v2 估算（5.5 d，仅 MLIC++）多 7.5 d 的增量来自 v3 scope 扩展：把 MLIC + MLIC+ + MLICv2 合并进同 PR。回报：一次 review cycle 覆盖整个家族 + 抽象层通用性当场验证 + reviewer 看到 four-model coverage 而非孤立 sibling leaf。

比 v1 估算（3 d）多 8 d 的增量来自：
- Phase 1 抽象设计 + 新 leaf 实现 + 单元测试 + ELIC 等价回归 + helper 抽出（+1.5 d）
- Phase 3 应用层 factory + helper class（+0.5 d，已落地；实际 LoC 高于 sketch，因 wrapper 保持 checkpoint 输入顺序）
- Phase 4 model thin wrapper 重写（v1 是直接 cherry-pick，v2 是按 ELIC pattern 重写）（+0.5 d）

本 PR 总 LoC（v3，含全家族四模型）：
- **MLIC++ Track**（Phase 1-9）：
  - Phase 1 净增 +698（new leaf 324 + helpers 145 + test 285 − checkerboard.py trim 66 + 杂项 10）
  - Phase 2 净增 +1187（mlic 子包 976 + tests 211；其中 `context.py` 411 行略高于原估，主要来自 license header + 保留 upstream LocalContext 结构）
  - Phase 3 净增 +560（slice factory 318 + 5 测试约 90 + Phase 3 期间小幅扩展现有测试 ~150；factory LoC 5× 超 plan，原因是 `_MlicppSideLayout` dataclass + `_MlicppEntropyParameters` split/reorder 不是 trivial closure）
  - Phase 4 净增 +439（model class 290 + 2 model tests 149）
  - 后续 Phase 估增 ~+250（CLI convert script ~50 + ckpt round-trip 测试 ~80 + zoo wiring ~30 + extras 测试 ~90）
  - **MLIC++ Track 累计 +3134**（实际 1-4 落地 + 后续估）
- **v1 + v1+ Track**（Phase 10-11）：
  - Phase 10 净增 ~+236（mlic/context.py 追加 quadratic/global/local context blocks + `mlic/__init__.py` export + 5 个 layer 测试）
  - Phase 11 实际增 ~+430（factory +151 / thin model +262 / zoo +46 / 测试增量并入现有测试文件）
- **v2 Track**（Phase 12-14）：
  - Phase 12 估增 ~+150（leaf selective_predictor hook + helper + 2 测试）
  - Phase 13 估增 ~+700（mlicv2 子包 STM/HGCP/CR/RoPE/GSC + 测试）
  - Phase 14 估增 ~+450（factory ~300 + thin model ~140 + zoo ~30 + 测试 ~20）
- **预计总 +5300 ~ +5500**（vs v2 估 +3100，扩展四模型后翻倍；vs v1 估 +1300 已翻 4×）。如 reviewer 反对 PR 体量，按 Phase 9/10 边界拆出后半 Track 到 follow-up PR

---

## 完成后

- 移动本文件到 `plan/exec-plans/completed/`
- 更新 `plan/README.md` 索引
- 在 design doc `channel-slice-codec-redesign.md`：
  - §3.4 表把 MLIC++ 的 codec 列从 `MLICPlusPlusLatentCodec` 改为 `ChannelGroupsLatentCodec + MultiContextCheckerboardLatentCodec(× K slices)`（**已在 2026-06-04 重构落地**）
  - 加新条目：`MultiContextCheckerboardLatentCodec` 在 §3.4 codec 类完整分类表占一行（family 2 / leaf / 用户 = MLIC++）（**已落地**）
  - dead-end 结论「MLIC++ 内部不容器化」的修订（改为「部分容器化：双 head + 多 context + 双 LRP 抽成 sibling leaf」）现记录在 [`codec-containerization-h-g-refactor.md`](../completed/codec-containerization-h-g-refactor.md#design-rationale)「设计依据」段 D.8（原 design-doc §10.12 已并入该处）
- 在 [`family2-roadmap.md`](family2-roadmap.md) §1 / §3 / §4 表更新 MLIC++ 相关行
- **MambaIC PR (PR-3) 启动时**重新评估：是否把 `MambaICLatentCodec` 也改写为 `MultiContextCheckerboardLatentCodec` 的应用层装配（spatial_context_nonanchor=VSSBlock）；如可，PR-3 LoC 会进一步压缩
- **新增 follow-up exec plan `plan/exec-plans/active/residual-block-act-norm-refactor.md`**（独立 PR，不绑定本 PR）：把 `compressai.layers.layers.ResidualBlock*` + `compressai.models.sensetime.ResidualBottleneckBlock` 加 `act=` (and optionally `norm=`) kwargs，删除 `compressai/layers/lic/mlic/transforms.py` 私有 `_ResidualBlock*`，避免每个论文 fork 一份私有 GELU/SiLU/LayerNorm 变体。先收集 DCAE / SAAF / 其他 follow-up 模型的实际需求再做设计决策。详细动机见 §Phase 2 实施差异第 2 节
