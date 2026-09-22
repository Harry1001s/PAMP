# Residual CondPool

## 模型

`预测 log2(kcat) = 冻结原 predictor(ESM2 mean, substrate) + γ × local correction(ESM2 residues, substrate)`

原 predictor 使用当前固定的 CLS / MLP / compact ensemble。mean embedding 直接传给原模型；局部分支仅做底物条件残基汇总，将 `[local, substrate projection]` 输入官方 TabM（K=16），预测残差。

- γ 是可学习标量，初始为 **0**；局部输出乘训练集标签标准差以恢复 log2 单位，最后不再加标签均值。
- 只训练 local branch 和 γ。原 predictor 的权重、BatchNorm 状态和 dropout 模式保持冻结。
- epoch 0 的原模型作为候选 checkpoint。只有 validation RMSE 更低才更新 best；这保证所选模型的验证 RMSE 不劣于该基线，测试表现仍需实测。
- 优化目标是各 TabM 成员的最终预测 MSE 后取平均，推理时才平均成员输出。

## 数据与训练

固定 train/validation/test = **22,126 / 2,766 / 2,766**，沿用 `experiment_mean/split_indices.npz`。

复用 `../catapro_tabm_small/cache/esm2_residue/` 的全量特征，ESM2 冻结。全局训练输出复用已核验的冻结预测缓存，避免重复计算原模型。

**SCREENING ONLY — legacy substrate cache**。两个分支使用同一 UniKP substrate cache。输入 scaler 与标签标准差只用训练集拟合。对原模型训练集内残差进行拟合仍可能过拟合；较小 γ 本身不保证验证或测试集改善。

配置见 `config.json`：seed 42、AdamW lr 5e-4、batch 64、最多 200 轮、patience 20。只进行本次单 seed 初筛，结束后自动保存预测、训练曲线、γ 变化、注意力诊断及对比报告。

## 结果与进度

- 状态：`reports/status.json`
- 训练日志：`logs/training.csv`、`logs/run.log`
- 最佳权重：`checkpoints/best.pt`
- 最终报告：`reports/final_report.md`
- 逐样本预测：`predictions/train.csv`、`val.csv`、`test.csv`

需要提前停止时，在本目录创建 `STOP` 文件。训练循环会结束并评估已保存的最佳权重；SIGTERM 同样用于请求停止训练后完成评估。

## Python 推理

```python
import sys
sys.path.insert(0, '/root/rivermind-data/experiments/catapro_residual_condpool/scripts')
from residual_model import load_predictor

model = load_predictor(
    '/root/rivermind-data/experiments/catapro_residual_condpool/checkpoints/best.pt',
    device='cpu',
)
# h_mean: [B,1280]，传入现有 mean embedding
# H: [B,L,1280]，逐残基 embedding
# substrate: [B,1024]；mask: [B,L] bool
y_log2 = model.predict_log2(h_mean, H, substrate, mask)
```

直接传入原始 embedding，模型内部已有 scaler。加载器验证原 predictor manifest 的哈希，并加载固定的原模型权重。精确复现缓存全局输出使用 CPU float32；局部分支训练和指标评估使用 GPU bfloat16 混合精度，CPU 推理可能有小幅数值差异。
