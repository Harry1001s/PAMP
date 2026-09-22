# Original Predictor: CLS head + interaction MLP + compact MLPs (§III.B, Eq. 1)

The frozen global branch `f(h,s)` of the Residual Predictor, and a turnover-prediction
baseline in its own right (Table I: "Original Predictor (ESM2)").

- `kcat_models.py` — architectures for the two "old" heads averaged in Eq. 1:
  `mean_cls_transformer` (CLS head: projects protein/substrate vectors to a shared
  space, attends a learned CLS token over them, scalar head on the CLS output) and
  `mean_interaction_mlp` (interaction branch: MLP over
  `[p; q; p*q; |p-q|] ∈ R^2048`).
- `interaction_kcat_models.py`, `kcat_cls_mlp_hybrid.py` — supporting architecture
  variants/utilities used while developing the interaction and CLS heads.
- `run_mean_kcat_comparison.py` — trains the CLS head and interaction MLP with AdamW
  (batch size 64, warmup + cosine decay, early stopping on validation RMSE) as described
  in §Implementation Details.
- `kcat_prediction_ensemble.py` — `MeanKcatEnsemble`: the frozen, fixed-weight
  ("Two-head average" in Table I) combination of the CLS head and interaction MLP.
- `compact_kcat_models.py` — the three compact MLPs `f_{M,1..3}` trained on concatenated,
  standardized protein/substrate vectors (the `1/9 * sum_j f_{M,j}` term in Eq. 1).
- `train_compact_kcat.py` — training driver for the compact MLPs.
- `improved_kcat_ensemble.py` — `ImprovedKcatEnsemble`, the complete Original Predictor:
  combines the frozen `MeanKcatEnsemble` (two-head average) with the three frozen
  compact-MLP checkpoints using the fixed weights in Eq. 1. This is the object loaded as
  the frozen global branch inside `../method/residual_predictor/residual_model.py`.
- `generate_full_esm2_embeddings_pkl.py`, `generate_unikp_smiles1024_fixed.py` —
  precompute the frozen ESM2 mean protein embeddings and UniKP SMILES-Transformer
  substrate embeddings that all of the above heads are trained on.
