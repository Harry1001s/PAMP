"""Row-preserving summaries and WT-cluster bootstrap uncertainty."""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

METRICS = ['wt_mean_plddt','mut_mean_plddt','delta_mean_plddt',
           'wt_mutation_site_plddt','mut_mutation_site_plddt','delta_mutation_site_plddt',
           'wt_local_plddt_pm5','mut_local_plddt_pm5','delta_local_plddt_pm5',
           'tm_score','usalign_rmsd_angstrom','backbone_rmsd_angstrom',
           'mutation_site_ca_displacement_angstrom','local_backbone_rmsd_pm5_angstrom']

def sample_indices(groups, repeats, seed):
    codes,_=pd.factorize(groups)
    members=[np.flatnonzero(codes==i) for i in range(codes.max()+1)]
    rng=np.random.default_rng(seed)
    for _ in range(repeats):
        yield np.concatenate([members[i] for i in rng.integers(0,len(members),len(members))])

def stats_table(df,repeats,seed):
    rows=[]
    for j,metric in enumerate(METRICS):
        values=pd.to_numeric(df.get(metric,pd.Series(index=df.index,dtype=float)),errors='coerce')
        valid=np.isfinite(values); x=values[valid].to_numpy(float); groups=df.loc[valid,'orig_seq']
        row={'metric':metric,'n':len(x),'n_wt_clusters':groups.nunique(),
             'mean':np.mean(x) if len(x) else np.nan,
             'median':np.median(x) if len(x) else np.nan,
             'std':np.std(x,ddof=1) if len(x)>1 else np.nan}
        boots=[]
        if len(x)>1 and groups.nunique()>1:
            for idx in sample_indices(groups,repeats,seed+j):
                b=x[idx]; boots.append([b.mean(),np.median(b),np.std(b,ddof=1)])
        for k,stat in enumerate(['mean','median','std']):
            lo,hi=np.nanquantile(np.array(boots)[:,k],[.025,.975]) if boots else (np.nan,np.nan)
            row[f'bootstrap95_{stat}_low']=lo; row[f'bootstrap95_{stat}_high']=hi
        rows.append(row)
    return pd.DataFrame(rows)

def correlations(df,repeats,seed):
    rows=[]
    for j,metric in enumerate(METRICS):
        if metric not in df: continue
        x=pd.to_numeric(df.delta_log2_pred,errors='coerce')
        y=pd.to_numeric(df[metric],errors='coerce'); valid=np.isfinite(x)&np.isfinite(y)
        x=x[valid].to_numpy(); y=y[valid].to_numpy(); groups=df.loc[valid,'orig_seq']
        eligible=len(x)>=3 and np.ptp(x)>0 and np.ptp(y)>0
        estimates=[pearsonr(x,y),spearmanr(x,y)] if eligible else [(np.nan,np.nan)]*2
        boots=[]
        if eligible and groups.nunique()>1:
            for idx in sample_indices(groups,repeats,seed+100+j):
                if np.ptp(x[idx])==0 or np.ptp(y[idx])==0: continue
                boots.append([pearsonr(x[idx],y[idx]).statistic,spearmanr(x[idx],y[idx]).statistic])
        for k,method in enumerate(['pearson','spearman']):
            lo,hi=np.nanquantile(np.array(boots)[:,k],[.025,.975]) if boots else (np.nan,np.nan)
            rows.append({'metric':metric,'method':method,'n':len(x),'n_wt_clusters':groups.nunique(),
                         'coefficient':estimates[k][0],'p_value_naive_row_independence':estimates[k][1],
                         'wt_cluster_bootstrap95_low':lo,'wt_cluster_bootstrap95_high':hi,
                         'valid_bootstrap_repeats':len(boots)})
    result=pd.DataFrame(rows)
    if len(result):
        result['p_fdr_bh_naive']=np.nan
        p=result.p_value_naive_row_independence; idx=p.index[np.isfinite(p)]
        ordered=idx[np.argsort(p.loc[idx].to_numpy())]
        q=p.loc[ordered].to_numpy()*len(ordered)/np.arange(1,len(ordered)+1)
        result.loc[ordered,'p_fdr_bh_naive']=np.minimum(1,np.minimum.accumulate(q[::-1])[::-1])
    return result

def finalize(results,out,repeats,seed):
    from run_pamp_structure_preservation import write_csv,save_json,atomic_text
    if results.structure_status.eq('pending').any(): raise RuntimeError('Cannot finalize pending rows')
    success=results.structure_status.eq('success')
    failures=results.loc[~success]
    write_csv(out/'failures.csv',failures)
    counts=results.structure_status.value_counts().to_dict()
    report={'n_input_rows':len(results),'n_complete_rows':int(success.sum()),
            'n_failed_rows':int((~success).sum()),'failure_rate':float((~success).mean()),
            'status_counts':counts,'bootstrap_unit':'exact WT sequence',
            'bootstrap_repeats':repeats,'seed':seed,
            'negative_activity_rows_retained':int((pd.to_numeric(results.delta_log2_pred)<0).sum()),
            'summary_population':'all input rows with finite values per metric; no confidence or activity filter'}
    write_csv(out/'failure_summary.csv',pd.DataFrame([{'status':k,'n_rows':v,'fraction':v/len(results)} for k,v in counts.items()]))
    for scope,df in [('all_rows',results),('unique_pairs',results.drop_duplicates(['orig_seq','mut_seq']))]:
        print(f'Statistics: {scope}, bootstrap={repeats}',flush=True)
        stat=stats_table(df,repeats,seed)
        path='summary_metrics.csv' if scope=='all_rows' else 'summary_unique_pairs.csv'
        write_csv(out/path,stat)
        if scope=='all_rows': report['metrics']=stat.to_dict('records')
    save_json(out/'summary_metrics.json',report)
    # Activity varies across substrate rows even for the same structure pair.
    # Keep row-level activity for the primary exploratory correlations.
    corr=correlations(results,repeats,seed)
    write_csv(out/'structural_correlations.csv',corr)
    # Explicitly labelled pair-level sensitivity, with mean activity per pair.
    unique=results.copy(); unique['delta_log2_pred']=pd.to_numeric(unique.delta_log2_pred)
    mean_delta=unique.groupby(['orig_seq','mut_seq']).delta_log2_pred.transform('mean')
    unique['delta_log2_pred']=mean_delta
    pair_corr=correlations(unique.drop_duplicates(['orig_seq','mut_seq']),repeats,seed)
    write_csv(out/'structural_correlations_unique_pairs.csv',pair_corr)
    atomic_text(out/'README.md',
        '# PAMP structural preservation evaluation\n\n'
        'Input rows are preserved by cache_row, including negative predicted activity changes. '
        'Structures are ESMFold v1 predictions of the supplied monomer sequences. '
        'pLDDT is C-alpha confidence on a 0–100 scale. Global means weight residues equally; '
        'local pLDDT averages the mutation and up to five residues on each side. '
        'All deltas are mutant minus WT. TM-score uses fixed residue numbering and WT length normalization. '
        'Backbone RMSD uses N, CA, C and O of every residue after one global Kabsch fit, in angstroms.\n\n'
        'Summaries include all rows with a finite value for each metric. Bootstrap confidence intervals '
        'resample exact WT-sequence clusters with replacement and retain all rows in sampled clusters. '
        'Unique-pair summaries are a separate sensitivity analysis. Pearson/Spearman raw p-values and '
        'BH-adjusted p-values assume independent rows; repeated enzymes violate that assumption, so '
        'interpret the WT-cluster bootstrap intervals as the primary uncertainty estimates. '
        'Homologous enzymes may still create dependence beyond exact WT clusters.\n\n'
        'pLDDT measures model confidence, not folding free energy or experimental stability. '
        'High predicted similarity does not establish preserved catalytic activity, ligand binding, '
        'oligomerization, or experimental structure. No confidence threshold is used to remove rows.\n')
    print(f'Finalized {len(results)} rows; failure rate={report["failure_rate"]:.4%}',flush=True)
