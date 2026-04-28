# 我们在做什么？

`CompressAI` 是 `Learned Image Compression（LIC）` 领域经典的Infra，但是其内部官方支持的模型截止到了2022年的 `ELIC`。我们正在尝试集成那之后的模型。

# 注意事项

1. 待集成的模型代码在 `candidate/` 目录内，请你在该目录下写一个 `TODO.md`。完成一个模型，修改一个TODO。
2. 原库的模型写在 `compressai/models/` 内。原库的机器学习层写在 `compressai/layers/` 内。
3. 尽可能优雅的完成迁移，将模型的各层拆分为公用的最大子集。
4. 部分模型的代码是基于旧版本的compressai，你可能需要迁移到现版本的风格。
5. python环境可使用当前目录下的.venv。
6. pytest不要跑zoo的pretrained部分。
