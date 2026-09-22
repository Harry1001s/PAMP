# CataPro TabM predictors

独立实验目录。固定划分：train 22,126 / validation 2,766 / test 2,766，直接读取 `../../experiment_mean/split_indices.npz`。

**SCREENING ONLY — legacy substrate cache**

## 模型

- Mean-TabM：ESM2 layer33 mean 1280 + UniKP substrate 1024 → 官方 TabM（K=16，3 层，宽度 512）。
- CondPool-TabM：逐残基 ESM2 → 底物条件加权汇总 → [global, local, substrate] → 相同 TabM。
- 输出均为 `log2(kcat)`。训练集拟合的输入及目标 scaler 随权重保存。

结果见 `reports/final_report.md`；逐样本结果见 `predictions/`；完整训练配置见 `configs/`。

## Python 调用

```python
import sys
sys.path.insert(0, '/root/rivermind-data/experiments/catapro_tabm_small/scripts')
from predictor import load_predictor

model = load_predictor(
    '/root/rivermind-data/experiments/catapro_tabm_small/checkpoints/tabm_mean/best.pt',
    device='cpu',
)
# protein_mean: float tensor [B,1280]; substrate: float tensor [B,1024]
pred_log2_kcat = model.predict_log2(protein_mean, substrate)
```

CondPool 同样加载 `checkpoints/tabm_condpool/best.pt`，输入改为 `[B,L,1280]` 的逐残基特征，并传入 `[B,L]` 的布尔 `mask`。BOS/EOS 不包含在特征中；padding 对应 `False`。模型支持对输入特征求导。

直接传入原始 embedding，模型已包含 scaler。特征必须符合训练缓存约定。历史 UniKP 缓存依赖样本在编码批次中的位置；新 SMILES 的编码需保留相同生成约定。当前推理接口接收预计算 embedding。

报告中的新模型指标使用 GPU bfloat16 混合精度推理；命令行默认 CPU float32，两者可能有小幅数值差异。抽查训练样本的差异记录在 `reports/inference_cli_verification.json`。

## 命令行推理

```bash
/opt/conda/envs/kcat_env/bin/python scripts/predict.py \
  --checkpoint checkpoints/tabm_mean/best.pt \
  --protein new_protein_mean.npy --substrate new_substrate.npy \
  --output new_predictions.csv
```

CondPool 的 `--protein` 使用 NPZ，每个样本一个 `[Li,1280]` 数组，键为字符串 `0`、`1`……。不使用全局 padding 文件。

## 训练和复现

依赖：现有 `/opt/conda/envs/kcat_env/bin/python`；官方 `tabm==0.0.3` 与 `rtdl_num_embeddings==0.0.12` 保存在本目录 `vendor/`，来源 <https://github.com/yandex-research/tabm>。

```bash
/opt/conda/envs/kcat_env/bin/python scripts/audit_existing_experiment.py
/opt/conda/envs/kcat_env/bin/python scripts/sanity_check.py
/opt/conda/envs/kcat_env/bin/python -u scripts/train.py --kind mean
/opt/conda/envs/kcat_env/bin/python -u scripts/extract_residue_cache.py
/opt/conda/envs/kcat_env/bin/python -u scripts/train.py --kind condpool
/opt/conda/envs/kcat_env/bin/python scripts/compare_results.py
```

已完成训练不会重新执行测试评估；发现未完成但已存在的 checkpoint 时拒绝覆盖。重新训练需复制到新的独立实验目录，并保持对应数据路径。提取残基缓存支持按已完成的序列继续。

训练只按 validation RMSE 选 epoch；test 在每个最佳 checkpoint 冻结后评估一次。seed42 初筛后，只有 CondPool validation RMSE 优于 Mean，或 Mean validation RMSE 比当前基线低至少 0.02，才为两种模型追加 seed3407 与 seed2026。

## 追加的 32 成员实验

用户追加要求：测试 TabM 的 32 个 MLP 成员。使用 `scripts/train.py --kind mean --k 32`，checkpoint 单独保存在 `checkpoints/tabm_mean_k32/best.pt`。其余训练设置不变，与 K=16 的结果分别保留。`reports/selected_mean_predictor.json` 按 validation RMSE 指向较好的 mean 模型。

## CataPro full embedding

`cache/esm2_residue/` 中每个数字命名的 NPY 对应一条唯一蛋白质序列，形状 `[序列长度,1280]`、float16。`index.json` 保存序列 SHA256、长度及完整 27,658 行的映射；`completion.json` 存在时表示全部 13,211 条唯一序列已生成并检查。

```python
from cache_reader import CataproResidueCache
cache = CataproResidueCache()
residue_features = cache[123]  # 原始 CataPro CSV 第 123 行，零起始
```

不同底物对应同一蛋白质时共享序列特征。没有删除数据行，train/validation/test 仍读取原有 split。长序列按原 pipeline 的 1022 残基分段计算后拼接；BOS/EOS 不计入特征。
