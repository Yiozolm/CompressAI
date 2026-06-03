# LIC Model Integration Template

> 用途：后续把 `candidate/` 下单个 LIC 模型迁入 CompressAI 时，统一代码落点、注册方式、zoo 暴露和验证口径。

## 1. 文件落点

- 模型主体：`compressai/models/<model_name>.py`
- 共享层：优先复用 `compressai/layers/lic/`，仅模型独有模块放在同目录新文件
- entropy / latent codec：优先复用 `compressai/entropy_models/` 与 `compressai/latent_codecs/`
  - **Minnen2020-style 通道自回归熵模型**（按通道等大切片 + 双头 `cc_mean_transforms` / `cc_scale_transforms` + 内置 LRP + 单 RANS string）：直接用 `compressai.latent_codecs.ChannelSliceLatentCodec`，不要再手写 `chunk(num_slices)` 循环；可参考 `compressai/models/stf_support.py::SliceEntropyCompressionModel` 的薄壳写法
  - **He2022/ELIC-style 不等大 channel groups + 内层 checkerboard**：用 `ChannelGroupsLatentCodec` + `CheckerboardLatentCodec`，参考 `compressai/models/sensetime.py::Elic2022Official`
- 临时对照脚本或分析记录：`temp/`，完成后删除

## 2. 模型类模板

```python
from typing import Any, Dict

from torch import Tensor

from compressai.models import CompressionModel
from compressai.registry import register_model


@register_model("<model-arch-name>")
class ModelName(CompressionModel):
    def __init__(self, N: int = 192, M: int = 320, **kwargs: Any) -> None:
        super().__init__(entropy_bottleneck_channels=N)
        self.N = int(N)
        self.M = int(M)

    def forward(self, x: Tensor) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Tensor]) -> "ModelName":
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N=N, M=M)
        net.load_state_dict(state_dict)
        return net
```

## 3. Zoo 暴露清单

迁入模型后同步更新：

- `compressai/models/__init__.py`：re-export 新模型类
- `compressai/zoo/image.py`：补 `model_architectures`、`cfgs`、`model_urls`
- `compressai/zoo/__init__.py`：补 `image_models` 映射和函数 re-export
- 无公开权重时，`model_urls[architecture] = {}`，`pretrained=True` 应抛出 “not yet available”

## 4. 验证清单

- Import smoke：缺少严格可选依赖时，`import compressai` 仍成功
- Forward smoke：小尺寸随机输入，检查 `x_hat` 和 `likelihoods` key
- Compression smoke：仅在 entropy coder / optional dependency 可用时跑 `compress` / `decompress`
- State dict：候选 checkpoint 可通过 `from_state_dict` 恢复结构
- 数值等价：同一输入下，与 candidate 输出 key、shape、主要中间张量误差对齐

## 5. TODO 更新

每迁完一个模型：

- 勾选 `candidate/TODO.md` 对应模型
- 标注未解决项：license、缺权重、缺 optional dependency、数值差异
- 若新增共享层，回填 Phase 0 对应条目
