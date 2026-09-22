# RC-TabM variants and comparisons (Supplementary Information)

Supplementary comparison referenced in §III.B.1: "The Supplementary Information provides
the detailed RC-TabM architecture, training settings, and comparisons with Mean-TabM and
Standalone CondPool-TabM."

- `scripts/` — training/evaluation code shared across the TabM variants: `train.py`,
  `predictor.py`, `predict.py`, `common.py` (data loading, mirrors
  `../residual_predictor/`'s conventions), `cache_reader.py`, `extract_residue_cache.py`,
  `finalize_cache.py` (build/verify the cached full-residue ESM2 representations used by
  every residue-conditioned variant), `sanity_check.py`, `compare_results.py`,
  `evaluate_baseline_train.py`, and `verify_*.py` deliverable checks.
- `tabm_condpool.json`, `tabm_mean.json` — configs for the Standalone CondPool-TabM and
  Mean-TabM variants (a `tabm_mean_k32` variant with a larger ensemble was also run from
  the same config with `k` overridden).
- `../third_party/tabm/tabm.py`, `../third_party/tabm/rtdl_num_embeddings.py` — vendored third-party TabM
  implementation (Gorishniy et al., *TabM: Advancing Tabular Deep Learning with
  Parameter-Efficient Ensembling*, ICLR 2025, `yandex-research/tabm`), used by both this
  directory and `../residual_predictor/residual_model.py`. Original license header
  preserved at the top of each file.
- `build_comparison.py` / `ITERATION_COMPARISON_REPORT.md` — aggregates metrics across
  RC-TabM iterations and the Mean-TabM / CondPool-TabM variants into a single comparison
  table.

See [`../docs/provenance/rc_tabm_variants.README.md`](../docs/provenance/rc_tabm_variants.README.md) for the original experiment-directory notes.
