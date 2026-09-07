# PPKO发布与验证

[PPKO发布目录](../1_training/ppko/README.md)包含两个独立模型：

- V10B：给定基线、观测掩码、靶点和作用方向，使用完整图先验，预测8192个原生磷酸肽的变化。
- ChemState-MODZ：给定化合物结构、基线和实验条件，预测45个P100分析物的变化。发布52药和118药两份权重。

`PPKO_MODELS.json`列出权重、架构、训练与推理入口。V10B训练300轮，ChemState-MODZ的52药、118药部署模型分别训练30轮、10轮。

从`organized_code/1_training/ppko/`执行：

```bash
python test_ppko_release.py
```

该测试加载三份权重，检查架构参数一致、输出维度、有限值和重复推理一致性。V10B另检查总输出等于三路输出之和。测试使用模拟输入，仅验证软件与权重配套；实验性能按各自验证表及支持集报告。
