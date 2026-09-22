import sys as _sys, pathlib as _pl
for _c in _pl.Path(__file__).resolve().parents:
    if (_c / 'pamp_paths.py').exists():
        _sys.path.insert(0, str(_c)); break
from pamp_paths import DATA_MANIFEST, SPLIT_INDICES
from pathlib import Path
import subprocess,sys,json,datetime,os
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent

def status(**kw):
 kw['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(R/'STATUS.json').write_text(json.dumps(kw,indent=2))
def run():
 (R/'pipeline.pid').write_text(str(os.getpid()))
 for dim in [2471,2304]:
  status(stage='training',features=dim,trees=100,seed=42)
  with (R/'pipeline.log').open('a') as log:
   subprocess.run([sys.executable,'-u',str(R/'run_quick_extratrees.py'),'--features',str(dim),'--out',str(R/f'features_{dim}'),'--confirm-training','--n-jobs','16'],stdout=log,stderr=subprocess.STDOUT,check=True)
 old=pd.read_csv(R.parent/'verified_metrics.csv');old=old[(old.scope=='predictor_test')&(old.split=='test')]
 rows=[]
 for dim in [2304,2471]:
  x=pd.read_csv(R/f'features_{dim}/test_metrics.csv').iloc[0].to_dict();x['model']=f'Extra Trees ({dim})';x['features']=dim;rows.append(x)
 for key,name,dim in [('mean_cls_transformer','CLS head',2304),('mean_interaction_mlp','Interaction MLP',2304),('old_CLS_MLP_plus_esm_maccs','Fixed ensemble',2471)]:
  x={'model':name,'features':dim}
  for metric in ['N','R2','RMSE','MAE','PCC','Spearman']:
   v=old[(old.model==key)&(old.metric==metric)];assert len(v)==1;x[metric]=float(v.iloc[0].value)
  rows.append(x)
 result=pd.DataFrame(rows);result.to_csv(R/'comparison_summary.csv',index=False)
 # Independently reconstruct test metrics and check pair/row alignment for the quick exports.
 from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
 splits=np.load(str(SPLIT_INDICES));manifest=pd.read_csv(str(DATA_MANIFEST))
 for dim in [2304,2471]:
  d=pd.read_csv(R/f'features_{dim}/test_predictions_seed_42.csv');assert np.array_equal(d.row_id,splits['test_idx'])
  assert np.array_equal(d.pair_key,manifest.iloc[d.row_id].pair_key)
  assert np.allclose(d.y_true_log2,manifest.iloc[d.row_id].y_log2,atol=1e-12)
  rec=result[result.features.eq(dim)&result.model.str.startswith('Extra')].iloc[0]
  assert abs(rec.R2-r2_score(d.y_true_log2,d.y_pred_log2))<1e-12
  assert abs(rec.RMSE-np.sqrt(mean_squared_error(d.y_true_log2,d.y_pred_log2)))<1e-12
  assert abs(rec.MAE-mean_absolute_error(d.y_true_log2,d.y_pred_log2))<1e-12
 lines=['# Extra Trees 快速对照结果','','100 棵树，seed=42，CPU 16 路并行，max_features=0.3、min_samples_leaf=1、bootstrap=False。参数预先固定，不进行调参。训练 22,126 条、验证 2,766 条、测试 2,766 条；只在训练集拟合。','','| 模型 | R² | RMSE | MAE |','|---|---:|---:|---:|']
 for x in rows:lines.append('| '+x['model']+' | '+' | '.join(f'{x[k]:.4f}' for k in ['R2','RMSE','MAE'])+' |')
 et=result[result.model.eq('Extra Trees (2471)')].iloc[0];ens=result[result.model.eq('Fixed ensemble')].iloc[0]
 lines+=['',f'完整特征下，Extra Trees 相对固定集成的 R² 差为 {et.R2-ens.R2:+.4f}，RMSE 差为 {et.RMSE-ens.RMSE:+.4f} log₂。','',f'两组树模型的拟合时间分别为 {rows[0]["train_seconds"]:.1f} 秒（2,304 维）和 {rows[1]["train_seconds"]:.1f} 秒（2,471 维），不含输入读取、模型保存及预测。','','这是单种子、未经调参的快速基线，不代表 Extra Trees 最优效果，不估计随机种子或划分不确定性。神经网络比较值来自同一测试集的历史已核验输出；其推理精度依原实验而异。现有 UniKP 缓存缺陷保留，训练/测试仍有重复序列。该比较只评价预测效果，不能作为 PAMP 攻击独立验证或实测酶活证据。']
 (R/'RESULTS.md').write_text('\n'.join(lines)+'\n');(R/'validation.json').write_text(json.dumps({'row_alignment':True,'pair_alignment':True,'labels_match':True,'metrics_recomputed':True,'n_test':2766},indent=2))
 status(stage='complete');print('\n'.join(lines),flush=True)
if __name__=='__main__':
 try:run()
 except Exception as e:status(stage='failed',error=repr(e));raise
