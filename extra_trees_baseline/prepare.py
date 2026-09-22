from pathlib import Path
r=Path(__file__).resolve().parent
s=(r.parent/'pending_experiments/run_matched_extratrees.py').read_text()
s=s.replace('"""NOT EXECUTED.','"""Quick, fixed-parameter comparison.')
s=s.replace('seeds=[42,2024,3407]','seeds=[42]')
s=s.replace("'n_estimators':500", "'n_estimators':100,'max_features':0.3,'min_samples_leaf':1")
s=s.replace("'selection':'validation RMSE at seed42, strict training-only fits; no test/model selection'", "'selection':'fixed quick configuration before fitting; no hyperparameter search; train only; validation descriptive'")
a=s.index('    grid=[];best=None');b=s.index('    fitted=[]',a)
s=s[:a]+"    best={'max_features':0.3,'min_samples_leaf':1}\n    save(dest/'selected_parameters_locked.json',best)\n"+s[b:]
s=s.replace('n_estimators=500','n_estimators=100')
s=s.replace('t=time.perf_counter();model.fit',"save(dest/'STATUS.json',{'stage':'training','trees':100,'seed':seed})\n        print('TRAINING',args.features,seed,flush=True)\n        t=time.perf_counter();model.fit")
s=s.replace("joblib.dump(model,dest/f'model_seed_{seed}.joblib')", "joblib.dump(model,dest/f'model_seed_{seed}.joblib',compress=3)")
s=s.replace("result=dict(seed=seed", "val_pred=model.predict(x[va]);save(dest/'validation_metrics.json',scores(y[va],val_pred))\n        result=dict(seed=seed")
s=s.replace('results.append(result)',"results.append(result)\n        print('TEST_RESULT',args.features,json.dumps(result),flush=True)")
s=s.replace("save(dest/'STATUS.json',{'status':'complete'", "cfg['status']='COMPLETE';save(dest/'run_config.json',cfg)\n    save(dest/'STATUS.json',{'status':'complete'")
s=s.replace('Seed variation is not ten-fold SD. No Extra Trees advantage was assumed.','Quick fixed-parameter single-seed baseline; not a tuned result or ten-fold estimate.')
(r/'run_quick_extratrees.py').write_text(s)
