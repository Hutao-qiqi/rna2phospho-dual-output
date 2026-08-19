# B0 残差研究 · τ 内层交叉验证 · 潜变量变换器（decoder-retrieval）

## 定位

本模块承接轴向动态超图模型之后的下一个实验阶段（2026-08）。核心目标是把磷酸化
预测拆成 **母蛋白基线 + 神经残差** 两层，并**在训练折内部**（而非仅外部留出）系统
评测「研究 × 位点」收缩参数 τ 的选择敏感性。同步引入一个轻量 **潜变量变换器
（decoder-retrieval）** 作为磷酸化残差算子，替代上一阶段的轴向超图图传播主干
（图传播退化为参考检索，不参与梯度）。

测试/推理/推理边界沿用轴向阶段的固定 916/229/286 划分数据契约，但本阶段
允许在 τ 内层 CV 的折内以 0 封存样本运行（`--expected-sealed-samples 0`）。

## 目录

```
code/   模型、训练、目标函数与数据工具（全部 --arg 参数化，无硬编码路径）
run/    运行脚本（路径用环境变量 SCP682_ROOT / SCP682_PYTHON，不硬编码）
```

### code/ 新文件

| 文件 | 作用 |
|---|---|
| `decoder_retrieval_model.py` | 潜变量变换器：RNA+蛋白测量 → 磷酸潜 token → 固定位点坐标查询 |
| `train_decoder_retrieval.py` | 锁定划分上的主训练入口（全 --arg） |
| `reference_retrieval.py` | 无泄漏的训练标签检索（模块级回退摘要） |
| `reliable_objective.py` | 覆盖率感知相关目标 + 确定性 Huber 回退 |
| `site_ridge_floor.py` | 训练期按位点岭下限（chunked 位点模型专用） |
| `experiment_contract.py` | 配置、划分隔离、报告工具 |
| `manual_dp_runtime.py` | 双 Windows CUDA worker 共享内存梯度同步 |
| `chunk_manifest.py` / `chunked_centering.py` | 位点分块清单、训练残差尺度拟合与全面板中心化 |
| `cancer_training.py` / `cophee_prior_bundle.py` | 癌症适配器与 CoPhee 先验打包 |
| `audit_*.py` / `build_crossfit_protein_reliability.py` | 输入与蛋白可靠性审计 |
| `evaluate_external_neural_operator.py` / `spawn_decoder_retrieval_windows.py` | 外部推理与窗口并发 |

### run/

运行脚本把绝对路径抽成两个环境变量，缺省即报错，避免任何内网路径/用户名泄漏：

```bash
export SCP682_ROOT=/path/to/SCP682-main        # 数据根
export SCP682_PYTHON=/path/to/project/python    # 运行 Python
```

- `run_inner_tau_cv_u32_20260819.sh`：5 折内层 CV × τ∈{0,2,5,10,20,50} 主循环
- `build_inner_tau_cv_contracts_20260819.py`：按研究分层 5 折内层契约（`--input --output`）
- `aggregate_tau_inner_cv_20260819.py`：聚合结果（`--input --cvroot --results`）
- `run_b0_*` / `run_a0b0_*` / `run_old_reference_*`：单次 B0 / 基线消融运行
- `inner_tau_preflight.sh`：τ CV 预检

CUDA 驱动兼容 shim 的库目录可用 `CUDA_LIB_DIR` 覆盖（默认
`/usr/lib/x86_64-linux-gnu`），版本号保留为脚本内常量。

## 数据契约

沿用轴向阶段：916 训练（总蛋白全为折外 `cross_fitted`）/ 229 选择验证（只用
916 拟合）/ 286 封存；磷酸化标签永不进入图输入；τ 内层各折按研究分层，
训练折内 fit 研究 × 位点中心，验证仅应用训练参数。

## 脱敏说明

所有新增 `.py` 已把本地 `/data/lsy`/`/data/fcr` 硬编码路径改为命令行参数；
`.sh` 改为 `SCP682_ROOT`/`SCP682_PYTHON` 环境变量。若有残留本地路径，见
`predict_scp682main1_phosphosite_centered_external.py`（上一阶段遗留，本分支未改）。