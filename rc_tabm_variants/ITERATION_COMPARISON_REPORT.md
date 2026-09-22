# Residual 与模型迭代历史的预测性能对比

本次整理 161 条评价记录、主表 16 个模型版本；逐种子结果、训练/验证/测试指标、外部和不同划分结果均见 Excel。来源与 SHA256 见 provenance.json，原始结果表见 source_tables。不是按时间连续改进的单一路径，包含探索失败与旁支。

## 同一固定测试集：2766 条

| model | R2 | PCC | Spearman | MSE | RMSE | MAE |
| --- | --- | --- | --- | --- | --- | --- |
| Residual Predictor | 0.64737 | 0.80754 | 0.79310 | 8.66345 | 2.94337 | 2.06624 |
| Extra Trees (2304 features) | 0.63966 | 0.79982 | 0.78544 | 8.85303 | 2.97540 | 2.06813 |
| Extra Trees (2471 features) | 0.63758 | 0.79853 | 0.78010 | 8.90401 | 2.98396 | 2.08258 |
| Three-family ensemble + MACCS | 0.61875 | 0.78876 | 0.77277 | 9.36654 | 3.06048 | 2.17098 |
| Original Predictor (no MACCS) | 0.61862 | 0.78907 | 0.77329 | 9.36989 | 3.06103 | 2.16916 |
| Two-head average | 0.60875 | 0.78469 | 0.76933 | 9.61237 | 3.10038 | 2.17203 |
| Mean-TabM K16 | 0.60495 | 0.77804 | 0.76188 | 9.70570 | 3.11540 | 2.24831 |
| Mean-TabM K32 | 0.60101 | 0.77541 | 0.76007 | 9.80248 | 3.13089 | 2.25079 |
| Compact MLP (3-seed prediction ensemble) | 0.59955 | 0.77681 | 0.75879 | 9.83844 | 3.13663 | 2.27383 |
| Compact MLP + MACCS (3-seed prediction ensemble) | 0.59769 | 0.77534 | 0.75684 | 9.88416 | 3.14391 | 2.29201 |
| Interaction MLP | 0.57567 | 0.77465 | 0.76056 | 10.42496 | 3.22877 | 2.20935 |
| CLS+MLP Hybrid | 0.57431 | 0.76554 | 0.74819 | 10.45856 | 3.23397 | 2.31262 |
| CLS head | 0.57311 | 0.76209 | 0.74609 | 10.48787 | 3.23850 | 2.32456 |
| Standalone CondPool-TabM K16 | 0.56337 | 0.75180 | 0.73198 | 10.72735 | 3.27526 | 2.35541 |
| Low-rank interaction (3-seed prediction ensemble) | 0.55642 | 0.75136 | 0.72962 | 10.89810 | 3.30123 | 2.36123 |
| Low-rank interaction + MACCS (3-seed prediction ensemble) | 0.54881 | 0.74773 | 0.72658 | 11.08504 | 3.32942 | 2.37964 |

主表 Compact/Low-rank 使用三种子预测平均，不是选择测试最优种子，也不是三种子指标均值。所有单种子和历史数值精度版本均保留于 All_matched_records。缺失指标留空，MSE=RMSE²；误差单位为 log2(kcat)。

## Residual 相对各模型的差值

| reference | Residual_minus_R2 | Residual_minus_PCC | Residual_minus_Spearman | Residual_minus_RMSE | Residual_minus_MAE | RMSE_reduction_percent |
| --- | --- | --- | --- | --- | --- | --- |
| Extra Trees (2304 features) | 0.00772 | 0.00772 | 0.00767 | -0.03203 | -0.00190 | 1.07652 |
| Extra Trees (2471 features) | 0.00979 | 0.00902 | 0.01300 | -0.04059 | -0.01634 | 1.36013 |
| Three-family ensemble + MACCS | 0.02862 | 0.01879 | 0.02034 | -0.11711 | -0.10475 | 3.82644 |
| Original Predictor (no MACCS) | 0.02875 | 0.01847 | 0.01982 | -0.11765 | -0.10292 | 3.84359 |
| Two-head average | 0.03862 | 0.02285 | 0.02377 | -0.15701 | -0.10579 | 5.06418 |
| Mean-TabM K16 | 0.04242 | 0.02950 | 0.03122 | -0.17202 | -0.18208 | 5.52171 |
| Mean-TabM K32 | 0.04636 | 0.03213 | 0.03303 | -0.18752 | -0.18455 | 5.98925 |
| Compact MLP (3-seed prediction ensemble) | 0.04783 | 0.03074 | 0.03431 | -0.19326 | -0.20759 | 6.16124 |
| Compact MLP + MACCS (3-seed prediction ensemble) | 0.04969 | 0.03220 | 0.03626 | -0.20053 | -0.22577 | 6.37851 |
| Interaction MLP | 0.07170 | 0.03289 | 0.03254 | -0.28540 | -0.14312 | 8.83919 |
| CLS+MLP Hybrid | 0.07307 | 0.04200 | 0.04491 | -0.29060 | -0.24638 | 8.98574 |
| CLS head | 0.07426 | 0.04545 | 0.04701 | -0.29512 | -0.25833 | 9.11302 |
| Standalone CondPool-TabM K16 | 0.08401 | 0.05574 | 0.06113 | -0.33189 | -0.28917 | 10.13323 |
| Low-rank interaction (3-seed prediction ensemble) | 0.09096 | 0.05618 | 0.06348 | -0.35785 | -0.29500 | 10.84002 |
| Low-rank interaction + MACCS (3-seed prediction ensemble) | 0.09857 | 0.05981 | 0.06653 | -0.38605 | -0.31340 | 11.59501 |

## 训练与验证

| model | split | N | R2 | PCC | Spearman | MSE | RMSE | MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Residual Predictor | train | 22126 | 0.91947 | 0.95969 | 0.95422 | 2.08492 | 1.44393 | 0.98044 |
| Residual Predictor | val | 2766 | 0.64566 | 0.80598 | 0.79336 | 9.18843 | 3.03124 | 2.11739 |
| Original Predictor (no MACCS) | train | 22126 | 0.87135 | 0.93557 | 0.93137 | 3.33082 | 1.82505 | 1.21579 |
| Original Predictor (no MACCS) | val | 2766 | 0.61465 | 0.78635 | 0.77835 | 9.99246 | 3.16108 | 2.22364 |
| Mean-TabM K16 | train | 22126 | 0.84524 | 0.92232 | 0.91607 | 4.00688 | 2.00172 | 1.35128 |
| Mean-TabM K16 | val | 2766 | 0.60111 | 0.77561 | 0.76793 | 10.34363 | 3.21615 | 2.31481 |
| Mean-TabM K32 | train | 22126 | 0.84612 | 0.92392 | 0.91714 | 3.98391 | 1.99597 | 1.36084 |
| Mean-TabM K32 | val | 2766 | 0.60499 | 0.77784 | 0.76991 | 10.24315 | 3.20049 | 2.30716 |
| Standalone CondPool-TabM K16 | train | 22126 | 0.80161 | 0.90196 | 0.89229 | 5.13631 | 2.26634 | 1.62827 |
| Standalone CondPool-TabM K16 | val | 2766 | 0.57511 | 0.75889 | 0.74491 | 11.01795 | 3.31933 | 2.39941 |

## 外部评估

| model | split | N | R2 | PCC | Spearman | MSE | RMSE | MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Two-head average | pair_novel/rows | 4835 | 0.24470 | 0.53804 | 0.54011 | 18.99221 | 4.35801 | 3.15701 |
| Two-head average | sequence_novel/rows | 3794 | 0.22599 | 0.52381 | 0.51718 | 19.87415 | 4.45804 | 3.24699 |
| Original Predictor (no MACCS) | pair_novel/rows | 4835 | 0.25722 | 0.54263 | 0.54561 | 18.67729 | 4.32172 | 3.14608 |
| Extra Trees (2304 features) | pair_novel/rows | 4835 | 0.27024 | 0.52002 | 0.53071 | 18.35006 | 4.28370 | 3.19417 |
| Residual Predictor | pair_novel/rows | 4835 | 0.25947 | 0.54639 | 0.54839 | 18.62076 | 4.31518 | 3.13678 |
| Original Predictor (no MACCS) | sequence_novel/rows | 3794 | 0.23497 | 0.52684 | 0.52201 | 19.64354 | 4.43210 | 3.23710 |
| Extra Trees (2304 features) | sequence_novel/rows | 3794 | 0.25069 | 0.50116 | 0.50259 | 19.24008 | 4.38635 | 3.28185 |
| Residual Predictor | sequence_novel/rows | 3794 | 0.23641 | 0.52934 | 0.52313 | 19.60666 | 4.42794 | 3.23103 |
| Two-head average (historical evaluation) | pair_novel_all/BRENDA_rows | 4835 | 0.24468 | 0.53804 | 0.54011 | 18.99272 | 4.35806 | 3.15706 |
| Three-family ensemble + MACCS | pair_novel_all/BRENDA_rows | 4835 | 0.26127 | 0.54364 | 0.54737 | 18.57545 | 4.30992 | 3.13554 |
| Two-head average (historical evaluation) | sequence_novel_all/BRENDA_rows | 3794 | 0.22597 | 0.52381 | 0.51717 | 19.87472 | 4.45811 | 3.24705 |
| Three-family ensemble + MACCS | sequence_novel_all/BRENDA_rows | 3794 | 0.24091 | 0.52923 | 0.52513 | 19.49102 | 4.41486 | 3.22270 |

外部配对未见与序列未见不是独立样本集，后者是前者子集。MACCS 集成和 Original Predictor（无 MACCS）不是同一权重版本；历史 Two-head 数值精度版本也保留命名区分，不能当成架构变化收益。

## 其他划分：不纳入主表排名

| model | N | R2 | PCC | Spearman | MSE | RMSE | MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| predictor_h512_h8 | 1638 | 0.23051 | 0.52617 | 0.52443 | 18.32215 | 4.28044 | 3.08779 |
| predictor_h512_h8_randomsplit | 1658 | 0.49697 | 0.71529 | 0.70759 | 12.79907 | 3.57758 | 2.49227 |
| predictor_r2_lr1e3 | 1658 | 0.52577 | 0.72574 | 0.70298 | 11.58239 | 3.40329 | 2.51236 |
| predictor_r2_mse_adam | 1658 | 0.51118 | 0.72074 | 0.70403 | 11.93880 | 3.45526 | 2.46563 |
| predictor_unikp1024_global_local | 1658 | 0.58058 | 0.76637 | 0.75469 | 10.24386 | 3.20060 | 2.23766 |
| predictor_unikp1024_global_local_drop005 | 1658 | 0.59482 | 0.77296 | 0.76406 | 9.89591 | 3.14578 | 2.25027 |
| predictor_unikp1024_v2_s42 | 1658 | 0.54935 | 0.74899 | 0.72689 | 11.00647 | 3.31760 | 2.34501 |
| ragged_kcat_predictor_cuda_outputs | 1638 | 0.33322 | 0.60824 | 0.60736 | 15.87662 | 3.98455 | 2.79861 |
| ragged_random_split | 1658 | 0.50931 | 0.71835 | 0.71405 | 12.48516 | 3.53343 | 2.44437 |
| Early global-local two-model ensemble | 1658 | 0.61010 | 0.78206 | 0.77021 | 9.52276 | 3.08590 | 2.17811 |

| model | group | N | R2 | PCC | Spearman | MSE | RMSE | MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Two-head average | official_10fold | 27658 | 0.17430 | 0.41927 | 0.41130 | 21.27179 | 4.61214 | 3.53217 |
| mean_cls_transformer | official_10fold | 27658 | 0.15275 | 0.39614 | 0.39120 | 21.82715 | 4.67195 | 3.57455 |
| mean_interaction_mlp | official_10fold | 27658 | 0.13252 | 0.38741 | 0.38263 | 22.34825 | 4.72739 | 3.63526 |
| CLS+interaction MLP, fixed 0.5/0.5, no MACCS | official_development | 3444 | 0.62610 | 0.79406 | 0.78892 | 9.75523 | 3.12334 | 2.13754 |
| Exact 0.618618 reference: CLS + interaction MLP + 3 esm_plain heads; no MACCS | official_development | 3444 | 0.63784 | 0.80027 | 0.79456 | 9.44908 | 3.07394 | 2.11586 |

## 结论与解释范围

在上述同划分主表中，Residual 的五项原始指标均为最优；MSE 与 RMSE 排名等价。它相对独立 CondPool 的收益较大，相对 Extra Trees 的收益较小。外部结果不支持全面超过 Extra Trees；MACCS 集成在部分外部误差指标上也更好。训练集拟合优度不代表泛化。

沿用既有 UniKP 底物缓存，其位置编码沿样本轴应用的问题已在原报告记录；本表是既有流程下的结果，不是修正缓存后重训的验证。项目已多次查看测试集并迭代模型，不可描述成完全未接触的前瞻性盲测。大多数模型仅一个训练种子，主表没有进行配对显著性检验。早期 1638/1658 测试、官方十折、官方 dev、可靠性划分分别列出；不能用其 R² 与 2766 测试直接比较。

未完成/仅 smoke 的分支见 Not_final_models；没有将训练中或未保存最终指标的分支填入估计结果。Excel 保留 source 列，可逐条追溯。
