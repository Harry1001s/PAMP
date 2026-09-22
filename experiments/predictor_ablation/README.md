# Predictor ablation framework (Table I)

A general, config-driven training/evaluation package used to run the predictor-side
ablation in Table I (CLS head, interaction MLP, two-head average, compact MLPs, ESMC vs.
ESM2, and the final Residual Predictor).

- `src/` — the `predictor_ablation` package (`scripts/` and `tests/` add this
  directory's parent to `sys.path` and `import src.<module>`): `models.py` (configurable
  two-modality predictor architectures, incl. the `ConcatPredictor` fusion variants),
  `config.py`, `data.py`, `preprocessing.py`, `training.py`, `metrics.py`,
  `parameter_budget.py`, `registry.py`, `reproducibility.py`.
- `scripts/` — entry points: `count_parameters.py` (parameter-count accounting, e.g. the
  "8,507,926 prediction-head parameters" figure in §Model Complexity Analysis),
  `verify_brenda_source.py`, `prepare_protocol.py`.
- `tests/test_correctness.py` — correctness tests for the training/evaluation pipeline.

Only code is included; `runs/` (trained checkpoints and audit logs from the original
experiment directory) is not part of this repository. See
[`../docs/provenance/predictor_ablation.README.md`](../docs/provenance/predictor_ablation.README.md)
for the original experiment-directory notes.
