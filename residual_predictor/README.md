# k_cat Residual Predictor / RC-TabM (§III.B)

The complete, differentiable objective `F(h,H,s) = f(h,s) + C(H,s)` (Eq. 2) used to
guide PAMP: a frozen Original Predictor (`f`, see `../original_predictor/`) plus a
substrate-conditioned residue-level correction learned on top of it.

- `residual_model.py` — `LocalCorrection` is the Residue-Conditioned TabM (RC-TabM)
  branch: substrate-conditioned additive attention over residue representations
  (Bahdanau-style, masking padding), followed by a `TabM` head (`k=16` members,
  parameter-efficient ensembling, vendored under `../rc_tabm_variants/vendor/tabm.py`).
  `ResidualPredictor` combines this with the frozen `ImprovedKcatEnsemble` global branch
  (`../original_predictor/improved_kcat_ensemble.py`); `gamma` is the learned scale in
  Eq. 2, initialized to zero so the model starts equivalent to the Original Predictor.
- `run.py` — training driver: optimizes RC-TabM and `gamma` with AdamW against the
  standardized-target squared error while the global branch stays frozen; selects the
  final checkpoint by validation RMSE.
- `check_model.py` — loads a checkpoint and sanity-checks it against the frozen-branch
  and shape/scale invariants asserted in `residual_model.py`.
- `config.json` — architecture/training hyperparameters for the run that produced the
  checkpoint referenced elsewhere in this repository as
  `experiments/catapro_residual_condpool/checkpoints/best.pt`.

See `ORIGINAL_README.md` for the original experiment-directory notes.
