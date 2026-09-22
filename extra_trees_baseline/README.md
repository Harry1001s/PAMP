# Extra Trees baseline (Table I, Table III)

Following UniKP's regression framework: Extra Trees fit on the concatenated
1,280-dimensional ESM2 protein representation and 1,024-dimensional UniKP substrate
representation (2,304 features). Reported as a turnover-prediction baseline in Table I,
and used in `../pamp_search/` as the independent, non-differentiable re-scorer for
§Transfer to an Independent Predictor (Table III) — it shares no parameters or gradients
with the Residual Predictor that guides PAMP.

- `prepare.py` — builds the 2,304-d feature matrix from the cached ESM2/UniKP embeddings
  for the CataPro-derived train/val/test partitions.
- `run_quick_extratrees.py` — hyperparameter search on the validation set
  (`max_features`, `min_samples_leaf`) and final fit across seeds 42/2024/3407.
- `run_pipeline.py` — orchestrates `prepare.py` + `run_quick_extratrees.py` end to end
  and writes the model referenced elsewhere in this repository as
  `extra_trees_quick_v1/features_2304/model_seed_42.joblib`.

See `ORIGINAL_README.md` for the original experiment-directory notes.
