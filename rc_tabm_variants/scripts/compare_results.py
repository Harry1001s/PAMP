from common import *


def main():
    baseline=json.loads((OUT/'reports/baselines.json').read_text())
    rows=[]
    for name in ('PAMP predictor','Two-head average','Extra Trees'):
        v=next(r for r in baseline if r['model']==name and r['split']=='val')
        t=next(r for r in baseline if r['model']==name and r['split']=='test')
        rows.append(dict(model=name,seed='reference',val_R2=v['R2'],val_RMSE=v['RMSE'],test_R2=t['R2'],test_RMSE=t['RMSE'],test_MAE=t['MAE'],Pearson=t['PCC'],Spearman=t['Spearman'],parameters=None,best_epoch=None,gap=None))
    if (OUT/'reports/baseline_train.json').exists():
        bt=json.loads((OUT/'reports/baseline_train.json').read_text())
        rows[0]['gap']=rows[0]['val_RMSE']-bt['RMSE']; rows[0]['parameters']=bt['parameters']
    runs=[]
    for p in sorted((OUT/'reports').glob('tabm_*_metrics.json')):
        r=json.loads(p.read_text()); runs.append(r)
        v,t=r['metrics']['val'],r['metrics']['test']
        rows.append(dict(model=r['model'],seed=r['seed'],val_R2=v['R2'],val_RMSE=v['RMSE'],test_R2=t['R2'],test_RMSE=t['RMSE'],test_MAE=t['MAE'],Pearson=t['PCC'],Spearman=t['Spearman'],parameters=r['parameters'],best_epoch=r['best_epoch'],gap=r['generalization_gap']))
    df=pd.DataFrame(rows); df.to_csv(OUT/'reports/comparison.csv',index=False)
    def cell(x):
        if pd.isna(x): return '—'
        if isinstance(x,(float,np.floating)): return f'{x:.4f}'
        return str(x)
    table='| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'
    table+='\n'.join('| '+' | '.join(cell(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))
    common='SCREENING ONLY — legacy substrate cache. Same existing CataPro split, ESM2 layer33 and substrate cache; log2(kcat) units.\n\n'
    text='# CataPro TabM comparison\n\n'+common+table+'\n'
    aggregate_start=len(text)
    if len({r['seed'] for r in runs})>1:
        text+='\n## Three-seed mean ± sample standard deviation\n\n'
        for kind in sorted({r['model'].split('_seed')[0] for r in runs}):
            subset=df[df.model.map(lambda n:n.split('_seed')[0]==kind)]
            if len(subset)<2:
                text+=f'- {kind}: single seed only; no across-seed standard deviation.\n'
                continue
            for metric in ('val_R2','val_RMSE','test_R2','test_RMSE','test_MAE','Pearson','Spearman','gap'):
                text+=f'- {kind} {metric}: {subset[metric].mean():.4f} ± {subset[metric].std(ddof=1):.4f} (n={len(subset)}).\n'
    aggregate=text[aggregate_start:]
    (OUT/'reports/comparison.md').write_text(text)
    mean=next((r for r in runs if r['model']=='tabm_mean'),None)
    cond=next((r for r in runs if r['model']=='tabm_condpool'),None)
    if mean is None: return
    mv,mt=mean['metrics']['val'],mean['metrics']['test']
    answers=[f'Q1 — Mean-TabM vs current differentiable ensemble: Δtest R²={mt["R2"]-rows[0]["test_R2"]:+.4f}; Δtest RMSE={mt["RMSE"]-rows[0]["test_RMSE"]:+.4f}; Δvalidation RMSE={mv["RMSE"]-rows[0]["val_RMSE"]:+.4f}.',
        f'Q2 — Mean-TabM minus Extra Trees: Δtest R²={mt["R2"]-rows[2]["test_R2"]:+.4f}; Δtest RMSE={mt["RMSE"]-rows[2]["test_RMSE"]:+.4f}.']
    if cond is None:
        answers+=['Q3–Q6 — CondPool has not completed. No architecture conclusion yet.']
        decision='Pending CondPool experiment.'
    else:
        cv,ct=cond['metrics']['val'],cond['metrics']['test']
        answers+=['Q3 — CondPool minus Mean-TabM test changes: '+', '.join(f'{k} {ct[k]-mt[k]:+.4f}' for k in ('R2','RMSE','MAE','PCC','Spearman'))+'.',
                  f'Q4 — Training/validation RMSE gap: Mean-TabM {mean["generalization_gap"]:.4f}; CondPool {cond["generalization_gap"]:.4f}; change {cond["generalization_gap"]-mean["generalization_gap"]:+.4f}.']
        paired=[]
        for seed in (42,3407,2026):
            a=next((r for r in runs if r['seed']==seed and r['model'].split('_seed')[0]=='tabm_mean'),None)
            b=next((r for r in runs if r['seed']==seed and r['model'].split('_seed')[0]=='tabm_condpool'),None)
            if a and b: paired.append((a,b))
        stable=len(paired)==3 and all(b['metrics']['val']['RMSE']<a['metrics']['val']['RMSE'] for a,b in paired)
        answers+=[f'Q5 — Complete paired seeds: {len(paired)}; CondPool validation RMSE improved in {sum(b["metrics"]["val"]["RMSE"]<a["metrics"]["val"]["RMSE"] for a,b in paired)}. '+('Stable validation improvement across three seeds; still screening only.' if stable else 'Stable improvement is not established; do not advance to merged-data formal experiments on this evidence alone.')]
        if cv['RMSE']<mv['RMSE'] and ct['RMSE']<mt['RMSE'] and (ct['PCC']>mt['PCC'] or ct['Spearman']>mt['Spearman']) and cond['generalization_gap']<=mean['generalization_gap']+.1:
            decision='Continue CondPool-TabM as a screening candidate; '+('three-seed validation gains are consistent.' if stable else 'confirmation remains limited.')
        else:
            decision='Keep Mean-TabM as the simpler TabM reference; retain the existing predictor for production unless validation and test evidence both justify replacement.'
        answers+=['Q6 — '+decision]
    report='# CataPro TabM Small Experiment\n\n'+common
    report+='## 1. Data and cache audit\n\nSee environment_audit.md and cache_alignment_check.json. Training 22,126; validation 2,766; test 2,766. Existing split reused byte-for-byte.\n\n'
    report+='## 2. Fair-comparison conditions\n\nIdentical dataset, labels, ESM2 checkpoint/layer, legacy substrate cache and fixed split. Scalers fit on train only. Official TabM 0.0.3; K=16; 3 blocks × width 512; per-member MSE; AdamW 5e-4, weight decay 1e-3; batch 64; max 200 epochs; patience 20. Only validation RMSE selects checkpoints. Baseline val/test metrics are recomputed from existing prediction files, without refitting or changing those files.\n\n'
    report+='## 3. Mean-TabM\n\nConcatenate standardized ESM2 mean (1280) and UniKP substrate (1024), then official TabM.\n\n'
    report+='## 4. CondPool-TabM\n\nFrozen residue features; projection 256; substrate-conditioned additive attention hidden 128; masked mean/global, weighted local and substrate vectors concatenated (768) before the same TabM backbone. Padding is excluded from attention and mean. This changes both information and representation compression, so gains alone cannot prove attention causality.\n\n'
    report+='## 5. Validation results\n\n## 6. Test results\n\n'+table+'\n\n'
    report+='## 7. Generalization gap\n\nGap = validation RMSE minus train RMSE, evaluated at the selected checkpoint. See comparison table and training curves.\n\n'
    report+='## 8. Comparison with current differentiable predictor\n\n'+answers[0]+'\n\n## 9. Comparison with Extra Trees\n\n'+answers[1]+'\n\n'
    report+='## 10. Decision\n\n'+'\n\n'.join(answers[2:])+'\n\n'+decision+'\n'
    report+='\n## Correlations versus Extra Trees\n\nPCC and Pearson r refer to the same metric. Higher is better for both Pearson and Spearman.\n\n| Model | Test Pearson/PCC | Δ vs Extra Trees | Test Spearman | Δ vs Extra Trees |\n|---|---:|---:|---:|---:|\n'
    for row in rows:
        report+=f'| {row["model"]} | {row["Pearson"]:.4f} | {row["Pearson"]-rows[2]["Pearson"]:+.4f} | {row["Spearman"]:.4f} | {row["Spearman"]-rows[2]["Spearman"]:+.4f} |\n'
    report+='\nAttention weights are engineering diagnostics, not catalytic-site annotations. No mutation, adversarial or merged-data experiments were performed. Existing substrate batch-slot dependence limits interpretation and future sequence/SMILES deployment; supplied cached embeddings are required for exactly reproducible inference.\n'
    mean32=next((r for r in runs if r['model']=='tabm_mean_k32'),None)
    if mean32:
        v32,t32=mean32['metrics']['val'],mean32['metrics']['test']
        section='\n## User-requested K=32 comparison\n\n'
        section+='The user explicitly requested 32 MLP members after the K=16 screening run. All other settings and splits are unchanged. K=32 was not selected by test performance.\n\n'
        section+=f'K=32 minus K=16: validation RMSE {v32["RMSE"]-mv["RMSE"]:+.4f}; test RMSE {t32["RMSE"]-mt["RMSE"]:+.4f}; test R² {t32["R2"]-mt["R2"]:+.4f}.\n'
        preferred=mean32 if v32['RMSE']<mv['RMSE'] else mean
        section+=f'Validation-selected mean model: {preferred["model"]}. Single seed; stability is not established.\n'
        dump(OUT/'reports/selected_mean_predictor.json',dict(model=preferred['model'],checkpoint=preferred['checkpoint'],criterion='validation RMSE only',validation_rmse=preferred['metrics']['val']['RMSE']))
        report+=section
        with (OUT/'reports/comparison.md').open('a') as f: f.write(section)
    report+=aggregate
    summary='## 中文摘要\n\n沿用原始 CataPro 划分：训练 22,126 / 验证 2,766 / 测试 2,766。底物使用与旧模型完全相同的 legacy 缓存，本实验仅作初筛。\n\n'
    summary+=f'- Mean-TabM K=16：测试 R² {mt["R2"]:.4f}，RMSE {mt["RMSE"]:.4f}，Pearson/PCC {mt["PCC"]:.4f}，Spearman {mt["Spearman"]:.4f}。\n'
    if mean32:
        summary+=f'- Mean-TabM K=32：测试 R² {t32["R2"]:.4f}，RMSE {t32["RMSE"]:.4f}，Pearson/PCC {t32["PCC"]:.4f}，Spearman {t32["Spearman"]:.4f}。成员数增加是否有效，应同时看验证和测试结果。\n'
    if cond:
        summary+=f'- CondPool-TabM K=16：测试 R² {ct["R2"]:.4f}，RMSE {ct["RMSE"]:.4f}，Pearson/PCC {ct["PCC"]:.4f}，Spearman {ct["Spearman"]:.4f}。\n'
    else:
        summary+='- CondPool 尚未完成，暂不作效果判断。\n'
    if (OUT/'cache/esm2_residue/completion.json').exists():
        summary+='- CataPro full embedding 已完成：13,211 条唯一序列，覆盖 27,658 条样本，约 13.23 GiB；逐条长度和 mean 一致性检查已通过。\n'
    summary+='\nPCC 与 Pearson 是同一指标。具体与 Extra Trees 的差值见下方相关性表。\n\n'
    report=report.replace('\n\n','\n\n'+summary,1)
    (OUT/'reports/final_report.md').write_text(report)
    manifest=json.loads((OUT/'reports/run_manifest.json').read_text())
    manifest['completed_runs']=[dict(model=r['model'],seed=r['seed'],checkpoint=r['checkpoint'],
        checkpoint_sha256=sha(r['checkpoint']),best_epoch=r['best_epoch'],
        metrics_file=str(OUT/f'reports/{r["model"]}_metrics.json')) for r in runs]
    manifest['seeds']=sorted({r['seed'] for r in runs})
    dump(OUT/'reports/run_manifest.json',manifest)
    print(table)


if __name__=='__main__': main()
