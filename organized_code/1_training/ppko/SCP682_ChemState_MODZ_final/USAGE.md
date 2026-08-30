# 使用方法

## 环境

```powershell
python -m pip install -r requirements.txt
```

## 构建 MODZ 标签

```powershell
python scripts\build_p100_modz_labels.py `
  --replicate-dir <P100重复水平输入目录> `
  --cohort-dir <P100队列目录> `
  --output-dir <MODZ标签输出目录>
```

## 训练 KI 模型

```powershell
python scripts\train_chemstate_modz.py `
  --cohort-dir <P100队列目录> `
  --modz-dir <MODZ标签目录> `
  --include-classes direct_kinase,pi3k_mtor `
  --epochs 30 `
  --seed 682 `
  --device cuda:0 `
  --output-dir <训练输出目录>
```

## 训练 All 模型

```powershell
python scripts\train_chemstate_modz.py `
  --cohort-dir <P100队列目录> `
  --modz-dir <MODZ标签目录> `
  --include-classes direct_kinase,pi3k_mtor,other,phosphatase `
  --epochs 10 `
  --seed 682 `
  --device cuda:0 `
  --output-dir <训练输出目录>
```

## 推理输入

压缩数组文件必须包含：

- `baseline`：样本数×45
- `input_mask`：样本数×45
- `fingerprint`：样本数×512
- `cell_ids`：样本数
- `condition`：样本数×2

`condition` 的两列依次为摩尔剂量的常用对数和处理小时数加一后的自然对数。

## 推理

```powershell
python scripts\infer_chemstate_modz.py `
  --checkpoint models\scp682_chemstate_modz_ki.pt `
  --input-npz <输入文件> `
  --output-npy <预测输出>
```

## 权重自检

```powershell
python scripts\test_frozen_models.py `
  models\scp682_chemstate_modz_ki.pt `
  models\scp682_chemstate_modz_all.pt
```
