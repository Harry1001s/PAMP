# 多 predictor 的 PAMP 攻击流程

入口：`run.py`。模型权重冻结，保留原始实验结果。

## 当前支持

- 原 predictor、Residual CondPool 分别作为白盒攻击目标。
- 每个候选同时由原 predictor、Residual CondPool、Extra Trees 评价。
- Extra Trees 当前是迁移评价器，不是独立黑盒搜索目标，不能称为已对树模型执行白盒攻击。
- 六种方法：A3_fw_avg_pamp、A0_pamp、A2_fw_avg、A1_hotflip、B0_esm_lm、B2_random。
- 第一轮 Top5：原序列上的五个排序候选，每个位点最多两个候选。保留全部结果，另报告由攻击源模型选择的 best-of-five。
- 两轮 Top1：第一轮固定排序第 1；在其突变序列上重新计算第二轮排名。本次使用 `--site-policy distinct`，第二轮排除第一突变位置，路径上的候选顶点与最终排序都遵守约束，确保两个不同位点、禁止回复突变。始终执行，不因预测下降而退回。可选旧模式 `allow` 不用于本次运行。
- Extra Trees 评价同一冻结候选，不用树预测重新选择 Top5。

## Residual 适配

全局输入为 `mean0 + mean(H_live) - mean(H_live_WT)`。

局部分支输入为 `H0 + H_live - H_live_WT`，H0 与现有 fp16 残基缓存精度一致。原 mean 分支被保留。底物在整个突变搜索中固定。

ESM2 按原流程使用至多 1022 残基的非重叠块；完整残基共同经过条件汇聚。采用链式法则先求完整 predictor 对残基的梯度，再分块回传到 ESM2 输入，以控制长序列显存。

FP32、TF32 关闭、MHA fastpath 关闭；与之前 bf16 推理可能有微小数值差别。`test_adapter.py` 在实际训练样本上验证零扰动恒等及分块梯度与直接反向传播一致。

## 运行方式

两条 BRENDA 样本的完整流程检查：

```bash
/opt/conda/envs/kcat_env/bin/python -u run.py --dataset brenda --out smoke_brenda --limit 2 --methods all
```

完整 BRENDA（4835 条，包含长序列）并启用 Extra Trees 迁移评价：

```bash
/opt/conda/envs/kcat_env/bin/python -u run.py --dataset brenda --out brenda_full --methods all --extra-trees
```

完整原 CataPro 测试集（2766 条）：

```bash
/opt/conda/envs/kcat_env/bin/python -u run.py --dataset catapro --out catapro_test_full --methods all
```

`--methods a3` 仅运行 A3。省略 `--limit` 才是完整队列；smoke 选最短两条仅用于检查流程，不能用于推断总体攻击效果。

每条样本独立保存 selections、rows，支持在相同冻结契约下恢复。汇总：candidate_results.csv、endpoint_results.csv、summary.csv。第一轮全部候选、Top1、best-of-Top5、第二轮 Top1 均可追溯。输出为模型预测变化，不能解释为实测活性提升；保留旧底物编码批次依赖限制。

## 本次确定范围

用户最终指定原 CataPro 测试集中长度 80–1024 的 2697 条记录，随后明确只攻击完整 Residual Predictor。Extra Trees 评价暂不做。同一蛋白搭配不同底物保留各记录，不按蛋白序列去重。仅运行 A3_fw_avg_pamp（三点梯度平均＋距离惩罚），第一轮 Top5 和两轮 Top1。A3 目标为 global + gamma × local 的最终预测，梯度同时经过 mean/global 分支与完整残基/local 分支；局部分支没有被 detach。旧 Predictor 仅作为完整模型内部的冻结全局分支，不再作为独立攻击目标。Extra Trees 默认不加载、不评价，只有显式传入 `--extra-trees` 才启用。

本次命令：

```bash
/opt/conda/envs/kcat_env/bin/python -u run.py --dataset catapro --out catapro_test_2697_residual_a3_distinct --methods a3 --min-length 80 --max-length 1024 --sources residual --site-policy distinct
```

三点为起点梯度与两次步长 0.5 的路径更新后的梯度，三者取平均后使用距离惩罚排序；不运行其余五种方法。

若要单独最大化局部修正项，而不最大化最终预测，那是另一个攻击目标，需要单独定义及报告；目前尚未启用这组额外实验。

筛选前完整测试集有 8 条含 U/X 的记录；长度筛选后仍保留符合范围的记录，仅对标准 20 种氨基酸位点生成替换。保留全部残基并按既有 1022 aa 分块策略处理。2432 aa 的长序列试跑已通过，但该条不在最终 80–1024 队列中。旧的 2766 条全量任务已停止；旧 2476 条开发攻击队列不用于本次任务。

## 2026-09-17 长序列扩展

新增原始 2766 条测试记录中全部长度 >1022 aa 的 58 条记录（47 个唯一序列，1024–2432 aa）。入口 `run_long58.py`，队列与结果目录 `catapro_test_long58_residual_distinct/`。沿用完整 Residual Predictor、五种方法 A0/A1/A2/A3/B2（不含 ESM-LM）、首轮 Top5 和两轮不同位点 Top1，完整序列按 1022 aa 分块。按长度降序执行以优先验证最长序列。

运行完成后自动校验并合并到 `results_no_esm_lm_2754/`，同时保留长序列独立汇总。原 2697 条与这 58 条重叠 row_id=10055，合并时该行统一采用本次长序列重测结果，最终 2754 条。只有该目录的 `verification.json` 为 PASS 时才表示合并完成；运行状态查看长序列目录中的 `status.json`、`run.log` 和可能的 `pipeline_error.json`。

## A3 在第二轮基础上增加三轮（总五轮）

入口 `run_a3_round5.py`，结果目录 `catapro_test_2754_residual_a3_round5_distinct/`。从 2754 条合并结果中的 A3 第二轮 Top1 继续，每轮重新求梯度并排除全部历史修改位点，最终恰好五个不同位点。每轮固定 Top1、始终接受，保留负增量。输出逐轮累计变化和相邻轮次增量，及按原蛋白序列聚类的 bootstrap 均值置信区间；另列 58 条长序列子组。原 WT 锚定保持不变，并逐条检查第二轮预测可复现。`status.json` 为 complete 且 `verification.json` 为 PASS 后才可视为完成。
