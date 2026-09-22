"""Build a source-grounded manuscript revision snapshot without changing experiments."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path('/root/rivermind-data')
TAB=ROOT/'experiments/catapro_tabm_small'
RES=ROOT/'experiments/catapro_residual_condpool'
EXT=ROOT/'experiment_brenda_external/residual_condpool_v1'
PAMP=ROOT/'experiments/pamp_multi_predictor_v1'
ACTIVE=PAMP/'catapro_test_2697_residual_a3_distinct'
OLD=Path('/root/paper_revision/20260910T034935Z/pamp_protein_substrate_v1')
SCMR=ROOT/'experiment_scmr_kcat_v0'
OUT=ROOT/'RESULTS_FOR_PAPER_REVISION_20260915.md'


def read(p):return json.loads(p.read_text())


def link(p,label=None):return f'[{label or p.name}]({p})'


def table(rows,columns):
    def value(v):
        if isinstance(v,(float,np.floating)):return f'{v:.6f}'
        return str(v).replace('|','/')
    text='| '+' | '.join(title for key,title in columns)+' |\n'
    text+='| '+' | '.join('---' for _ in columns)+' |\n'
    for row in rows:text+='| '+' | '.join(value(row[key]) for key,title in columns)+' |\n'
    return text


def main():
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    r=read(RES/'reports/metrics.json');cfg=read(RES/'config.json')
    assert read(RES/'reports/verification.json')['status']=='PASS'
    ext=pd.read_csv(EXT/'metrics.csv');assert read(EXT/'verification.json')['status']=='PASS'
    comp=pd.read_csv(RES/'reports/comparison.csv')
    oldtab=pd.read_csv(TAB/'reports/comparison.csv')
    names={'tabm_mean':'Mean-TabM K16','tabm_mean_k32':'Mean-TabM K32','tabm_condpool':'Standalone CondPool-TabM K16'}
    tabruns={name:read(TAB/f'reports/{name}_metrics.json') for name in names}
    status=read(ACTIVE/'status.json');contract=read(ACTIVE/'contract.json')
    cohort=pd.read_csv(ACTIVE/'cohort.csv')
    assert len(cohort)==2697 and contract['sources']==['residual'] and contract['site_policy']=='distinct'
    assert contract['source_split_counts']=={'train':0,'val':0,'test':2697}
    assert contract['methods']==['A3_fw_avg_pamp'] and contract['evaluators']==['residual']
    cache=read(TAB/'cache/esm2_residue/completion.json')
    sources=[RES/'reports/metrics.json',RES/'reports/comparison.csv',RES/'reports/verification.json',RES/'config.json',
        TAB/'reports/comparison.csv',EXT/'metrics.csv',EXT/'verification.json',EXT/'contract.json',
        ACTIVE/'contract.json',ACTIVE/'status.json',ACTIVE/'cohort.csv',
        SCMR/'results/response_degeneracy_check.json',OLD/'attack/summary.csv',OLD/'attack_top1_twice/summary.csv']
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (ROOT/'RESULTS_FOR_PAPER_REVISION_20260915_sources.json').write_text(json.dumps({'snapshot_utc':stamp,'files':source_hashes,'attack_status_snapshot':status},indent=2))
    parts=[]
    parts.append(f'''# CataPro / Residual Predictor / BRENDA / PAMP：论文修改成果总表

**快照时间：{stamp}。** 本文是本轮相关实验的可追溯成果汇总，不代表扫描了整个工作区所有历史研究项目。所有数字来自本地已保存评估文件；未新增训练、未修改论文原文件。

**阅读顺序：** 第 1 节看结论；第 3–6 节用于模型及数据 Methods/Results；第 7–8 节用于攻击实验；第 10 节提供英文替换段落；第 12 节提供文件入口。

## 1. 可以写进论文的结论与边界

1. **内部固定测试集上，Residual Predictor 的五项点估计均优于旧 Predictor 与 Extra Trees。** 测试 R²=0.647372，PCC=0.807545，Spearman=0.793102，RMSE=2.943374，MAE=2.066238。
2. **BRENDA 外部优势不一致。** 4835 条配对未见记录上，Residual Predictor 的 PCC 高于 Extra Trees，但 R² 更低、RMSE 更高；不能写“外部总体优于 Extra Trees”。
3. **Mean-TabM 与独立 CondPool-TabM 未超越旧 Predictor。** K32 相比 K16 的验证 RMSE 略低，但测试表现未改善。这里 K 是 TabM 成员数，不是网络层数。
4. **ESM2 完整残基特征已提取并校验。** Residual Predictor 保留旧 mean/global 预测，新增底物条件化残基修正分支。
5. **新 A3 攻击正在运行，尚无最终全队列结论。** 当前对象是完整 Residual Predictor，使用 2697 条原测试记录，第二轮禁止重用第一突变位置。运行状态见第 7 节。
6. **SCMR 仅完成方案审查。** 原公式的响应均值恒为零，已做数值验证；尚未训练，不能列入性能对比。

所有新神经模型仅完成 seed=42。没有对这些模型差异完成置信区间或显著性检验，不应写“显著提高”或“多种子稳定提升”。

## 2. 统一命名与术语

| 论文统一名称 | 本地别名 | 准确定义 |
|---|---|---|
| 旧 Predictor / Original predictor | PAMP predictor、原 predictor | 冻结的旧可微集成预测模型，输入 ESM2 mean 与底物 |
| Residual Predictor | Residual CondPool | 完整模型：旧 Predictor + γ × local correction |
| 局部残基分支 | LocalCorrection、local branch | Residual Predictor 的预测分支，接收 ESM2 逐残基表征；不是 ESM2 内部结构 |
| Standalone CondPool-TabM | tabm_condpool | 独立承担全部预测的早期模型；不能与 Residual Predictor 混用名称 |
| Mean-TabM K16/K32 | tabm_mean / tabm_mean_k32 | mean 特征上的 TabM；K 表示内部预测成员数 |
| PCC / Pearson r | ppc、pearson | 同一指标，论文只保留一列 |
| Spearman ρ | Spearman | 秩相关系数，与 PCC 不同 |
| 首轮 Top5 | round1_top5 | 五个独立单点替换候选，不是同时改五个位点 |
| 首轮 best-of-Top5 | round1_best_of_top5 | 在五个候选的真实重编码预测中，由攻击目标模型选择最高者 |
| 两轮 Top1 | round1_top1 → round2_top1 | 先执行固定排名第一的单点突变，再重算第二轮排名；不是从首轮 best-of-Top5 继续 |

## 3. 数据、划分与特征

### 3.1 固定内部预测划分

| 划分 | 记录数 |
|---|---:|
| Train | 22126 |
| Validation | 2766 |
| Test | 2766 |
| 合计 | 27658 |

原始数据：{link(Path('/root/kcat-data_0.4simi-10fold.csv'))}。划分：{link(ROOT/'experiment_mean/split_indices.npz')}。行级清单：{link(ROOT/'experiment_mean/data_manifest.csv')}。

目标为 y = log₂[kcat/(1 s⁻¹)]。训练集目标均值为 1.727250987598611，标准差为 5.088255379714057。特征及目标 scaler 只在训练集拟合。这里的固定划分不能自动解释为蛋白序列或同源簇互斥划分；相同/相似蛋白跨集合的限制需要保留。

### 3.2 ESM2 特征成果

- 同一冻结 ESM2 t33_650M_UR50D，第 33 层，残基维度 1280。
- mean 特征为每条记录的既有 1280 维缓存；Residual Predictor 明确保留该输入。
- 完整残基缓存：{cache['unique_sequences']} 个唯一序列，{cache['residues']} 个残基，覆盖全部 27658 条记录。
- 存储 {cache['storage_bytes']} bytes，约 {cache['storage_bytes']/2**30:.2f} GiB；每个序列独立 fp16 数组，不对整个数据集全局 padding。
- 超长序列按非重叠至多 1022 aa 分块；去除 BOS/EOS，残基顺序保留。
- 提取推理使用 fp32；存储 fp16。存储后重算 mean 与旧 mean 的最大绝对差 {cache['max_mean_error']:.9f}，在既定 0.002 容差内。
- 该资源不包含结构或活性位点真实标注；注意力位置不能自动解释为催化机制。

来源：{link(TAB/'cache/esm2_residue/completion.json')}、{link(TAB/'reports/cache_alignment_check.json')}。

### 3.3 底物特征的重要限制

当前已完成预测实验均沿用旧 UniKP 1024 维底物缓存。旧位置编码存在沿样本轴应用的问题，使底物特征依赖 batch slot。**这些结果不是修正位置轴后重训所得。** BRENDA 也复用了同一表示类型的既有外部缓存，batch size=64；这不等于消除了顺序与邻居依赖。

因此，内部结果可表述为“在同一既有特征流程下的比较”；外部结果应表述为“冻结旧流程的外部评估”。将来若修正底物特征，需在修正缓存上重训兼容基线，不能把缓存修正收益全部归因于模型架构。

来源：{link(TAB/'reports/substrate_reencoding_cuda.json')}、{link(ROOT/'experiment_brenda_external/smiles_metadata.json')}、{link(EXT/'contract.json')}。
''')
    cols=[('model','模型'),('R2','R² ↑'),('RMSE','RMSE ↓'),('MAE','MAE ↓'),('PCC','PCC ↑'),('Spearman','Spearman ↑')]
    internal=[]
    for split in ['val','test']:
        rows=[]
        for model in ['Original predictor','Extra Trees']:
            z=comp[(comp.model==model)&(comp.split==split)].iloc[0].to_dict();rows.append(z)
        for key,name in names.items():rows.append(dict(model=name,**tabruns[key]['metrics'][split]))
        rows.append(dict(model='Residual Predictor',**r['metrics'][split]));internal+= [{**z,'split':split} for z in rows]
        parts.append(f"## 4{'A' if split=='val' else 'B'}. {'验证集' if split=='val' else '测试集'}结果（N=2766）\n\n"+table(rows,cols))
    parts.append('''所有误差指标在原始 log₂(kcat) 单位上计算；R² 和相关系数无量纲。主模型对比采用相同固定划分。Mean-TabM/Standalone CondPool 是本轮筛查模型，未完成多种子重复。

### 4.1 Residual Predictor 相对改变量

''')
    changes=[]
    for name in ['Original predictor','Extra Trees']:
        b=comp[(comp.model==name)&(comp.split=='test')].iloc[0]
        m=r['metrics']['test']
        changes.append(dict(reference=name,delta_R2=m['R2']-b.R2,delta_PCC=m['PCC']-b.PCC,delta_Spearman=m['Spearman']-b.Spearman,
            delta_RMSE=m['RMSE']-b.RMSE,RMSE_reduction_percent=100*(b.RMSE-m['RMSE'])/b.RMSE,delta_MAE=m['MAE']-b.MAE))
    parts.append(table(changes,[('reference','测试参考模型'),('delta_R2','ΔR²'),('delta_PCC','ΔPCC'),('delta_Spearman','ΔSpearman'),('delta_RMSE','ΔRMSE'),('RMSE_reduction_percent','RMSE下降 %'),('delta_MAE','ΔMAE')]))
    parts.append('''MAE 相对 Extra Trees 的改善很小。这里是单次实验点估计，不是显著性检验结论。K32 的验证 RMSE 比 K16 更低，但测试 R²/PCC/Spearman 更低，因此不能说成员数增加带来一致收益。

### 4.2 训练结果及过拟合边界

''')
    training=[]
    for key,name in names.items():
        v=tabruns[key];training.append(dict(model=name,best=v['best_epoch'],end=v.get('epochs',53),params=v['parameters'],train_rmse=v['metrics']['train']['RMSE'],val_rmse=v['metrics']['val']['RMSE'],gap=v['generalization_gap']))
    training.append(dict(model='Residual Predictor（新增部分）',best=r['best_epoch'],end=r['last_completed_epoch'],params=r['local_parameters'],train_rmse=r['metrics']['train']['RMSE'],val_rmse=r['metrics']['val']['RMSE'],gap=r['generalization_gap']))
    parts.append(table(training,[('model','模型'),('best','最佳轮'),('end','末完整轮'),('params','本轮训练参数'),('train_rmse','训练RMSE'),('val_rmse','验证RMSE'),('gap','验证−训练RMSE')]))
    parts.append(f'''Residual Predictor 的最佳检查点为第 {r['best_epoch']} 轮，训练完成第 {r['last_completed_epoch']} 轮后按用户要求停止，最终评估读取最佳检查点。验证集用于检查点选择；没有用测试集重新选择 epoch。但整个项目已经反复查看测试结果并迭代模型，不能把本轮描述为完全未接触的前瞻性盲测。

Residual Predictor 训练 RMSE 为 {r['metrics']['train']['RMSE']:.6f}，验证−训练差距 {r['generalization_gap']:.6f}，大于旧 Predictor 的 1.336032。不能根据 γ 小或测试集改善宣称“没有过拟合”。Standalone CondPool 同样是人工提前停止（最佳第 49 轮、末完整第 53 轮）；Mean K16/K32 分别在第 95/142 轮结束。

## 5. Residual Predictor 的可复现方法

### 5.1 模型公式

令 H∈R^(L×1280) 为冻结 ESM2 的逐残基表示，h_mean 为既有 mean 缓存，s∈R^1024 为底物特征。

```text
                       冻结 ESM2
                      /         \\
               既有 mean       完整残基 H
                   |               |
底物 s → 旧 Predictor     底物 s → 条件注意力 → TabM K16
                   |               |
                y_global       Δy_local
                      \\         /
                    y = y_global + γ Δy_local
```

旧 Predictor 为三个模型家族的等权平均：旧 CLS 头、interaction MLP、三个 compact MLP 的平均。它接收 protein mean 1280 + substrate 1024，不含额外 MACCS 输入。具体权重以 {link(OLD/'model_manifest.json')} 为准。

局部模块将按训练集统计标准化的残基和底物分别投影到 256 维；使用 128 维 additive tanh 条件注意力，并对 padding 做 mask。注意力加权残基与底物投影拼接成 512 维输入，进入 TabM：K=16，3 blocks，每 block 宽度 512，dropout=0.1。残基特征 scaler 沿用训练集 protein mean 各维统计，不是在验证/测试残基上重新拟合。

对第 k 个 TabM 成员：

\\[
\\hat y_k=f_{{global}}(h_{{mean}},s)+\\gamma\\,\\sigma_{{train}}\\,t_k(H,s),
\\qquad
\\hat y=\\frac1K\\sum_{{k=1}}^K\\hat y_k.
\\]

局部输出乘训练目标标准差，不再添加目标均值。训练损失是逐成员标准化均方误差，先算各成员误差再平均；不是先平均成员再训练：

\\[
\\mathcal L=\\frac1{{BK}}\\sum_{{b,k}}
\\left(\\frac{{\\hat y_{{bk}}-y_b}}{{\\sigma_{{train}}}}\\right)^2.
\\]

γ 初始为 0，因此初始化精确恢复旧 Predictor。最佳 γ={r['gamma']:.9f}，允许负值。γ 的符号可与局部输出符号互相补偿，不能单独解释成抑制机制或预测下降。

### 5.2 训练与计算

| 项目 | 设置 |
|---|---|
| 训练参数 | local branch + γ，共 {r['local_parameters']:,} |
| 冻结旧 Predictor 参数 | {r['global_parameters_frozen']:,} |
| 完整预测头参数 | {r['local_parameters']+r['global_parameters_frozen']:,}，不含冻结 ESM2 |
| ESM2 | 约 650M 级预训练编码器，冻结；精确参数量不从简称推断 |
| 优化器 | AdamW，lr=0.0005；local weight decay=0.001；γ weight decay=0 |
| Batch / clipping | 64 / 梯度范数上限 1.0 |
| Seed | 42 |
| 训练数值 | local bfloat16 autocast；保存特征 fp16；误差计算 fp32 |
| 停止规则 | 上限 200，patience 20；本次实际为用户提前停止 |
| 检查点选择 | 最小验证 RMSE，含 γ=0 的第 0 轮候选 |
| 本次训练耗时 | {r['training_seconds']:.1f} s，约 {r['training_seconds']/60:.1f} min |
| 峰值 GPU allocated memory | {r['peak_gpu_memory_mib']:.1f} MiB |

记录的耗时属于本次运行，不能直接与并行提取缓存时的其他实验耗时解释为公平速度基准。

### 5.3 完成验证与导出修复

已验证 γ=0 恢复旧模型、冻结全局权重、padding 排除、γ 与局部梯度、checkpoint 回读及完整 split/预测对齐。最终导出时曾因 float32 残差列的 CSV 小数精度导致严格加和校验中断；最大误差不足 4.8×10⁻⁷。随后以绝对容差 10⁻⁶ 复核加和关系，并从导出的 y_pred 重算全部指标，与 metrics.json 在 10⁻¹² 内一致。没有重复测试推理，也没有修改已选权重。额外 Residual attention diagnostics 尚未生成，不能写已完成该可解释性分析。

来源：{link(RES/'reports/model_checks.json')}、{link(RES/'reports/verification.json')}、{link(RES/'reports/final_report.md')}。
''')
    parts.append('## 6. BRENDA 外部冻结评估\n\n未在 BRENDA 上训练、校准或选择检查点。配对未见定义为规范化蛋白序列＋RDKit canonical isomeric SMILES 的精确组合不在全部 CataPro 中；标签和 organism 不用于判定重复。序列未见进一步要求精确序列未出现，不等于远同源或新家族外推。\n\n')
    extcols=[('cohort','队列'),('model','模型'),('N','N'),('R2','R²'),('PCC','PCC'),('Spearman','Spearman'),('RMSE','RMSE'),('MAE','MAE')]
    rows=ext[ext.unit=='rows'].to_dict('records')
    for z in rows:z['cohort']={'pair_novel':'配对未见','sequence_novel':'序列也未见'}[z['cohort']];z['model']=z['model'].replace('Residual CondPool','Residual Predictor')
    parts.append(table(rows,extcols))
    parts.append('### 6.1 唯一配对汇总敏感性结果\n\n同一配对的重复 log₂ 标签及预测取均值：配对未见为 4783 个唯一配对；序列也未见为 3751 个唯一配对。\n\n')
    unique=ext[ext.unit=='unique_pairs'].to_dict('records')
    for z in unique:z['model']=z['model'].replace('Residual CondPool','Residual Predictor')
    parts.append(table(unique,extcols))
    parts.append(f'''Residual Predictor 在外部数据上的相关系数优于 Extra Trees，但平方误差解释能力仍不及 Extra Trees。这两个观察必须同时保留；相关性更强不保证绝对数值预测误差更低。相对旧 Predictor 的外部提升较小。

本次原模型必须使用当前冻结 manifest 的重算值：配对未见 R²=0.257223。目录顶层早期 RESULTS.md 的 R²=0.261273 属于早期运行，不能替代本次同批比较。这里优先使用 {link(EXT/'metrics.csv')} 与 {link(EXT/'contract.json')}。没有借用早期其他模型的 bootstrap CI 来描述新 Residual Predictor。

## 7. 当前 A3 攻击：配置已验证，完整结果待完成

### 7.1 队列与状态

- 原 Test 2766 条，按包含端点的 80–1024 aa 筛选，得到 **2697 条记录**；实际最短 103 aa、最长 1024 aa。
- 包含 **{cohort.sequence_clean.nunique()} 个唯一蛋白序列**，不是 2697 个互不相同蛋白。保留所有选中的原测试行，同蛋白不同底物分别计算梯度；未进一步对相同配对重复行去重。
- 来源分布：train=0，validation=0，test=2697；标签不参与突变选择。
- 当前仅攻击 **完整 Residual Predictor**。旧 Predictor 是其内部冻结 global branch；不再单独作为攻击源。Extra Trees 不评价。
- 快照状态：**{status['stage']}，完成 {status.get('completed',0)}/2697 条**；该数字会继续变化。
- 当前目录：{link(ACTIVE,'当前攻击目录')}；实时状态：{link(ACTIVE/'status.json')}。

本文件没有汇总中途攻击均值或成功率，避免把按固定顺序完成的部分样本误作全队列结果。完成后读取 summary.csv、endpoint_results.csv、candidate_results.csv 和 verification.json，再补论文攻击 Results。

### 7.2 实际攻击目标与链式梯度

攻击优化最终输出，而不是只优化全局分支或单独局部项：

\\[
J(W)=f_{{global}}(h_{{mean}}(W),s)+\\gamma f_{{local}}(H(W),s),
\\quad
\\nabla_WJ=\\nabla_W f_{{global}}+\\gamma\\nabla_W f_{{local}}.
\\]

W 是 ESM2 的输入氨基酸 embedding。ESM2 和 predictor 参数冻结，但输入梯度保留。条件注意力与完整残基参与梯度计算，两分支梯度允许不同或方向相反。所有候选最后都转换成离散氨基酸序列，再实际重编码评价。

为了在零扰动时对齐已有缓存，使用固定原序列锚点：

\\[
H_{{anchor}}(Z)=H_0+[H_{{live}}(W+Z)-H_{{live}}(W)],
\\]
\\[
h_{{anchor}}(Z)=h_0+[\\operatorname{{mean}}H_{{live}}(W+Z)-\\operatorname{{mean}}H_{{live}}(W)].
\\]

底物固定。第二轮始终保留原 WT 锚点，不把第一个突变重新当作原始缓存中心。推理为 fp32、TF32 off、MHA fastpath off，因此与训练/评估中的 bf16 local 推理可存在小幅数值差别。

### 7.3 A3 三点梯度平均与距离惩罚

每一轮取起点梯度 g₀，进行两次步长 0.5 的路径更新，得到 g₁、g₂；使用 (g₀+g₁+g₂)/3 排序。路径顶点由未加惩罚的一阶替换分数选择；最终候选使用原 PAMP 距离惩罚：

\\[
S_{{i,a}}=\\frac{{2\\epsilon}}{{\\|\\bar g\\|_F}}
\\bar g_i^T(e_a-e_{{x_i}})-\\|e_a-e_{{x_i}}\\|_2^2,
\\quad
\\epsilon=2d_{{median}}\\sqrt L.
\\]

d_median 为标准氨基酸输入 embedding 两两欧氏距离中位数。实现有分母下界避免除零。三点梯度指三个连续路径采样点，不是三个突变位点。

### 7.4 Top5 与两轮 Top1 的严格语义

1. 首轮从 WT 生成 5 个候选，最多每个位点保留 2 个替换；不是要求 5 个不同位点。
2. 首轮 Top1 是固定排序第 1，不是在重编码的 5 个结果中挑最好的。
3. 首轮 best-of-Top5 可以另行报告，但不能用它替代两轮 Top1 的第一步。
4. 第二轮在首轮固定 Top1 的突变序列上重新计算三点梯度，**路径顶点与最终候选都屏蔽第一突变位点**。
5. 最终必须有两个不同位点的替换；禁止回复原氨基酸，也禁止第一位点再换成第三种氨基酸。
6. 始终执行所选候选，保留负增益结果，不做 best-so-far 回退。
7. 模型预测增益不等于真实酶活提高。没有 WT/mutant 实测标签时，不能把“预测增益”为正称作实验证明的成功率。

验证成果：{link(PAMP/'adapter_checks.json')} 中直接反向传播与分块链式梯度最大差为 0；零扰动预测一致。{link(PAMP/'smoke_distinct/verification.json')} 中两个实例通过严格不同位点验证，其中包含旧规则下发生同位点修改的样本。2432 aa 长序列完整流程也已试跑通过，但其不属于当前长度队列。

### 7.5 最终攻击表格建议

| 方法 | 队列 N | 终点 | 平均 Δlog₂ | 几何平均预测倍数 | 预测正增益比例 | 95% CI |
|---|---:|---|---|---|---|---|
| A3 / Residual Predictor | 2697 | 首轮固定 Top1 | 待完成 | 待完成 | 待完成 | 尚未计算 |
| A3 / Residual Predictor | 2697 | 首轮 best-of-Top5 | 待完成 | 待完成 | 待完成 | 尚未计算 |
| A3 / Residual Predictor | 2697 | 两轮不同位点 Top1，最终相对 WT | 待完成 | 待完成 | 待完成 | 尚未计算 |

建议同时提供第二轮相对第一轮的增量，并区分记录平均与蛋白平均。当前自动 summary 主要是记录平均，不会自动产生蛋白/同源簇 bootstrap CI；若正文需要推断统计，需另行完成。几何平均预测倍数为 2^(平均 Δlog₂)，不是算术平均倍数。
''')
    old1=pd.read_csv(OLD/'attack/summary.csv');old2=pd.read_csv(OLD/'attack_top1_twice/summary.csv')
    hist=[]
    for z in old1[old1.method=='A3_fw_avg_pamp'].to_dict('records'):
        hist.append(dict(endpoint=f"首轮 {'Top1' if z['budget']==1 else 'best-of-Top5'}",mean=z['mean'],cluster=z['cluster_mean'],positive=z['positive']))
    for z in old2[(old2.method=='A3_fw_avg_pamp')&old2.endpoint.isin(['second_delta_log2','total_delta_log2'])].to_dict('records'):
        hist.append(dict(endpoint='第二轮增量' if z['endpoint']=='second_delta_log2' else '两轮累计相对WT',mean=z['sequence_mean'],cluster=z['cluster_mean'],positive=z['positive_rate']))
    parts.append('''## 8. 旧 2476 条攻击成果：历史结果，不能冒充新实验

旧队列筛选过程：全部 27658 → 80–512 aa 且标准氨基酸 21475 → 排除历史攻击开发涉及簇 11932 → 再排除含原测试集样本的簇 4064 → 完全相同蛋白序列去重、用固定哈希选一个底物记录 → 2476。最后有 1422 个检测到的序列簇，train=2278、val=198、test=0；实际长度 86–512。

簇来自既有 MMseqs 相似边图（identity≥40%，双向 coverage≥80%）的连通分量。这只是检测到的相似性关系，不保证排除所有局部或远同源。它避免复用先前攻击开发暴露簇，却没有排除 predictor 的训练暴露。

旧 A3 对旧 Predictor 的结果如下，数值均为模型自身预测 Δlog₂；旧两轮允许同位置与回复，不能与当前严格不同位点队列直接视为受控比较。

'''+table(hist,[('endpoint','历史终点'),('mean','序列平均Δlog₂'),('cluster','簇等权平均Δlog₂'),('positive','预测正增益比例')]))
    parts.append(f'''旧两轮同位点比例 0.029887、回复比例 0.004847。来源：{link(OLD/'attack/summary.csv')}、{link(OLD/'attack_top1_twice/summary.csv')}。如保留旧攻击结果，应在论文单列“历史开发队列”，注明模型、队列来源、去重方式、位点规则与聚合权重，不能冠以“原测试集”。

## 9. SCMR 方案成果：数学审查，未训练

提出的 SCMR 将残基投影后中心化 c_i=z_i−mean(z)，再施加对同一样本所有位置相同的底物条件线性算子 A(s)。因此：

\\[
m_1=\\operatorname{{mean}}[c_iA(s)]=0,
\\qquad
m_2=\\operatorname{{diag}}[A(s)^T\\operatorname{{Cov}}(z)A(s)].
\\]

数值验证：m₁ 最大绝对值 1.903×10⁻¹⁶；协方差恒等式误差 1.332×10⁻¹⁵；移除方差后条件算子梯度接近机器精度。有效信号是条件化的二阶统计，不能把 no_variance 解释为有信息响应均值对照。

建议的最小修正为响应头使用 [m₂,g,q]，并重新解释消融；若希望非零响应均值，需要显式引入非线性或非均匀汇聚。SCMR 尚未实现完整训练，不应出现在已完成性能表或摘要结论中。来源：{link(SCMR/'results/method_review.md')}、{link(SCMR/'results/response_degeneracy_check.json')}。

## 10. 可用于论文的英文替换段落

以下段落仅重述已核验方法与结果，不替代作者的科学论证。参考文献编号、图表编号应与原稿对应；没有新增或虚构外部文献。

### 10.1 Methods: residual predictor

We retained the original predictor as a frozen global branch receiving mean ESM2 and substrate embeddings. A substrate-conditioned local branch used residue-level representations from the same frozen ESM2 encoder. Masked additive attention pooled projected residue features, which were concatenated with substrate features and passed to a 16-member TabM head. The final prediction added the gated local correction to the global prediction. The scalar gate was initialized to zero, exactly recovering the original predictor at initialization. Only the local branch and gate were optimized using member-wise squared errors normalized by the training target standard deviation. All normalization statistics were fitted on the training partition, and checkpoints were selected by validation RMSE.

### 10.2 Results: internal predictive comparison

On the fixed CataPro test partition, the Residual Predictor achieved an R² of 0.6474 and a Pearson correlation of 0.8075. The corresponding Extra Trees values were 0.6397 and 0.7998, respectively. RMSE decreased from 2.9754 to 2.9434, whereas the MAE improvement was small. Mean-TabM and standalone CondPool-TabM did not outperform the original predictor under the evaluated configurations. These results were obtained with one training seed and the existing feature caches. They establish point-estimate improvements within this experimental setting, without demonstrating statistical significance or stability across training seeds.

### 10.3 Results: external evaluation and qualification

External evaluation used 4,835 BRENDA records whose exact protein–substrate pairs were absent from the CataPro dataset. All predictors were frozen, with no BRENDA-specific training, calibration or checkpoint selection. The Residual Predictor achieved a higher Pearson correlation than Extra Trees, at 0.5464 versus 0.5200. However, its R² was lower, at 0.2595 versus 0.2702, and its RMSE was higher. The subset containing previously unseen exact protein sequences showed the same qualitative pattern. Thus, the external results supported improved correlation but did not establish uniformly better predictive accuracy than Extra Trees.

### 10.4 Methods: current attack protocol

The attack cohort comprised 2,697 records from the original test partition, restricted to protein lengths of 80–1,024 residues. Records sharing a protein sequence were retained, preserving their distinct substrate contexts. A3 optimized the final Residual Predictor output, propagating gradients through both global and local branches to ESM2 input embeddings. Three gradients were averaged along two path updates of step size 0.5, followed by distance-penalized candidate ranking. Five first-round single-substitution candidates were evaluated by re-encoding their discrete sequences. Sequential Top1 attacks continued from the first-ranked candidate and recomputed gradients on the mutated sequence. The second round excluded the previously modified position throughout path construction and final ranking. This constraint ensured two distinct substitutions and prevented reversions to the original sequence. Negative predicted changes were retained, without a best-so-far acceptance rule.

### 10.5 Limitations: wording that must accompany the results

The existing UniKP substrate cache contained positional encoding applied along the batch axis, introducing batch-slot dependence. Accordingly, the reported comparisons characterize the legacy feature pipeline rather than a corrected, batch-invariant representation. Independent BRENDA evaluation excluded exact overlaps but did not establish separation from all homologous proteins. Predictor development also involved repeated inspection of the fixed test results, limiting claims of prospective evaluation. Finally, predicted mutation gains require experimental validation and should not be interpreted as measured improvements in catalytic activity.

**当前不提供新攻击 Results 英文段落：** 全部 2697 条尚未完成，不能填入估计的均值、成功率或置信区间。

## 11. 修改论文时应替换、保留或暂缓的内容

| 原稿可能的表述 | 处理 | 本轮证据支持的写法 |
|---|---|---|
| “PCC 和 Pearson 均提升” | 合并指标 | PCC/Pearson 为同一列，另报 Spearman |
| “32 层 MLP 优于 16 层” | 删除误称 | K16/K32 为 TabM 成员数；没有一致测试提升 |
| “CondPool 替换 mean 后更好” | 更正架构 | 独立 CondPool 未改善；Residual 保留旧全局预测 |
| “新模型在外部全面超过 Extra Trees” | 限定结论 | 外部 PCC 更高，但 R² 更低、RMSE 更高 |
| “2476 条测试蛋白” | 更正数据身份 | 旧开发攻击队列含训练/验证来源，不是原测试集 |
| “2697 个不同蛋白” | 更正计数单位 | 2697 条测试记录，2345 个唯一序列 |
| “Top5 为五点突变” | 更正预算定义 | 首轮五个独立单点候选 |
| “两轮一定两个位点”用于旧实验 | 限定版本 | 仅新 distinct 规则保证两个不同位点 |
| “攻击显著提高真实 kcat” | 删除未证实结论 | 报告模型预测变化，真实活性尚未测量 |
| “已证明没有过拟合” | 更正 | 明列 train–validation gap，承认差距增加 |
| “SCMR 提升已验证” | 暂缓 | 只有数学审查与退化数值检查，无训练结果 |
| “最终攻击统计已完成” | 暂缓 | 等待当前全队列完成并复核 |

### 正文与补充材料分配建议

- **正文核心：** Residual Predictor 内部比较及外部结果的限制，两者都保留。
- **Methods：** 固定划分、缓存限制、残差结构、损失、单 seed、人工停止与验证选优、A3 不同位点约束。
- **补充材料：** Mean/Standalone CondPool 筛查、完整训练曲线、唯一配对外部敏感性表、数值检查、参数与哈希。
- **旧攻击：** 若保留则单列历史开发队列，不能与新模型新队列的结果混表声称增益。
- **SCMR：** 后续研究计划或内部附录，不作为本轮已完成结果。

本次没有提供原稿，因此不存在可计算的原稿删改字数差；英文段落是供替换的素材，不建议全部直接追加到正文。主文保留结论改变所需证据，其余移到 Methods/SI，避免重复罗列所有指标。

## 12. 文件索引与复现入口

''')
    files=[('内部固定划分',ROOT/'experiment_mean/split_indices.npz'),('Mean/CondPool 完整报告',TAB/'reports/final_report.md'),
        ('Residue cache 索引',TAB/'cache/esm2_residue/index.json'),('Residue cache 行映射',TAB/'cache/esm2_residue/sample_index.csv'),
        ('Residual Predictor 代码',RES/'scripts/residual_model.py'),('Residual 训练配置',RES/'config.json'),('Residual 最佳 checkpoint',RES/'checkpoints/best.pt'),
        ('Residual 训练曲线',RES/'reports/training_curve.png'),('Residual 完整指标',RES/'reports/metrics.json'),('Residual 内部逐条测试预测',RES/'predictions/test.csv'),
        ('外部三模型完整指标',EXT/'metrics.csv'),('外部逐条预测',EXT/'predictions.csv'),('外部执行脚本',EXT/'evaluate.py'),
        ('当前 A3 适配层',PAMP/'attack_adapter.py'),('当前 A3 runner',PAMP/'run.py'),('当前冻结攻击契约',ACTIVE/'contract.json'),('当前输入队列',ACTIVE/'cohort.csv'),
        ('当前日志',ACTIVE/'run.log'),('当前状态',ACTIVE/'status.json'),('旧2476来源规则',OLD.parent/'pamp_conditional_v2/build_cluster_map.py'),
        ('SCMR 评估',SCMR/'results/method_review.md'),('本成果文档生成脚本',Path(__file__)),('本快照来源哈希',ROOT/'RESULTS_FOR_PAPER_REVISION_20260915_sources.json')]
    for title,p in files:
        assert p.exists(),p
        parts.append(f'- {title}：{link(p)}\n')
    parts.append('''
## 13. 下一步最小补全事项

1. 当前不同位点 A3 完成后，复核全部 2697 条及每条最终恰好两个替换，补上三个终点的最终统计。
2. 若要支持统计推断，补齐按蛋白或合适同源簇聚合的不确定性，并明确记录权重与簇权重。
3. 若要声称训练稳定性，完成预先固定种子的重复；本次不能从单次结果推断稳定优势。
4. 若要声称新的底物表示可靠，修正缓存并通过 batch/order/neighbor 不变性测试，在同特征上重训基线。
5. 如需证明残基信息本身带来增益，需要匹配容量/训练预算的对照；现有不同架构筛查不能独立排除容量与优化差异。

上述事项为明确的证据缺口，不影响已完成点估计的复现，也不代表本轮已执行这些新增实验。
''')
    OUT.write_text('\n\n'.join(parts))
    # Recheck every generated numeric row against its machine-readable source.
    assert r['metrics']['test']['N']==2766
    assert len(ext)==12 and set(ext[ext.unit=='rows'].N)=={4835,3794}
    assert all(np.isfinite(v) for row in changes for v in row.values() if isinstance(v,(float,np.floating)))
    for row in internal:
        assert row['RMSE']>0 and -1<=row['PCC']<=1 and -1<=row['Spearman']<=1
    print(json.dumps({'report':str(OUT),'characters':len(OUT.read_text()),'snapshot_utc':stamp,'attack_status':status,'tables_checked':True},indent=2))


if __name__=='__main__':main()
