# Extra Trees 快速实验

依据用户要求，停止原 500 棵树 / 6 参数组合 / 3 种子的长实验，保留其已有记录。本实验使用标准 sklearn ExtraTreesRegressor，100 棵树、seed=42、CPU n_jobs=16。参数在测试前固定：max_features=0.3、min_samples_leaf=1、bootstrap=False、max_depth=None；无超参数搜索。验证集只用于描述性评价，不据测试结果调整配置。

先训练 2,471 维 ESM2 + UniKP + MACCS，与完整固定集成匹配特征；再训练 2,304 维 ESM2 + UniKP，与单独 CLS/交互 MLP 匹配特征。复用既有的 22,126/2,766/2,766 训练/验证/测试记录，核对来源 SHA-256，不重新划分数据。

查看 STATUS.json 和 pipeline.log 获取进度。完成后 RESULTS.md 与 comparison_summary.csv 包含对照结果；每个 features_* 目录保存配置、验证指标、测试指标、逐行测试预测以及 model_seed_42.joblib。validation.json 记录测试行、pair key、标签和指标复算检查。

运行入口：/opt/conda/envs/kcat_env/bin/python -u run_pipeline.py。输出目录必须新建，不能覆盖已训练目录。

结论只能用于单种子快速基线，不声称最优树模型性能或统计显著性。原随机划分含跨分区重复序列，UniKP 缓存缺陷保持不变；这不是 PAMP 攻击的独立验证，也不是测量酶活改进。
