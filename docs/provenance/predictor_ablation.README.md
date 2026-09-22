# Predictor ablation — implementation in progress

用户已排除 Extra Trees 和 ESMC；使用 ESM-2 layer 33、UniKP 1024 维底物表示。重点为 raw concat、单/双分支投影 concat，以及等参数 sum/gated/low-rank bilinear。R² 0.65 是期望目标，选型指标仍为 validation RMSE。

当前已完成输入哈希、历史 split 映射、mean/max 特征派生、容量参数核算和 CPU 正确性/断点恢复 smoke。正式训练次数为 0。完整 S0–S9 调度、current_arch_controlled adapter、搜索、最终评价和报告尚未实现完成，不能把本目录当作已完成的实验。

## 数据前置条件

历史 train/val、train/test、val/test 分别有 9、14、2 组 organism/enzyme/smiles/log2Kcat 完全相同的记录。数量是每对划分的共享键数，不能直接相加当作独立测量数。四列表没有 measurement ID，无法确认是同一测量还是独立重复。

原始标签列及 NumPy 标签一致，尺度为 log2Kcat；物理单位待数据来源确认。旧代码曾把单位直接标记为 s⁻¹，现已撤回该未经证实的标注。正式协议锁定和训练入口对此设有阻断。

证据在 `runs/predictor_v1/audit/`，状态在 `runs/predictor_v1/completion_status.json`。CPU smoke 位于 `runs/predictor_v1/smoke/`；其指标来自历史 train 内的小子集，不得作为模型性能结论。

## 当前可执行命令

在本目录、使用已有 `kcat_env` 环境运行：

```bash
python -m unittest discover -s tests -v
python scripts/audit_data.py --manifest configs/data_manifest.json --experiment predictor_v1 --resume
python scripts/count_parameters.py --experiment predictor_v1 --all-required
python scripts/run_smoke_test.py --experiment predictor_v1 --device cpu --resume
```

`prepare_protocol.py` 会拒绝未解决数据前置条件的正式锁定。

## 待完成的关键工作

训练模块目前仅验证固定物理 batch 的 concat 系列短训练；尚需验证梯度累积、完整资源上限、代码/特征 hash 的恢复校验、完整 epoch train 指标、CPU/GPU 计时及失败状态管理。锁定与训练授权必须由审计和阶段文件联合校验，不能手工把 manifest 标志改为 true 跳过审计。S4 的最终参考模型依赖 S2 验证结果，当前配置清单只是 A0 上的候选示例。

全部原始数据与历史 checkpoint 保持原样。新派生 mean/max 向量共约 162 MiB；当前尚未删除任何恢复 checkpoint。
