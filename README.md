# PAMP: Continuous-to-Discrete Mutation Projection for Enzyme Engineering

This repository collects the core algorithm code behind *PAMP: Continuous-to-Discrete
Mutation Projection for Enzyme Engineering* (`PAMP_latest_v3.tex`). It is curated from a
much larger research workspace (experiment logs, cached embeddings, checkpoints, figure
sources) down to the scripts that implement the methods described in the paper. Large
data artifacts (model checkpoints, embedding caches, per-record result CSVs) are **not**
included; only the code that produced them.

## Layout, mapped to the paper

| Folder | Paper section | Contents |
|---|---|---|
| [`pamp_search/`](pamp_search) | §III.C *Adversarial Mutation Search* (Fig. 1) | The PAMP algorithm itself: multi-point gradient averaging, distance-penalized projection, HotFlip/Random baselines, sequential multi-round design, and the driver scripts that produced Tables II and III. |
| [`residual_predictor/`](residual_predictor) | §III.B, *Residue-Conditioned TabM (RC-TabM)* | The complete Residual Predictor: frozen Original Predictor + substrate-conditioned residue correction (Eq. 2), and its training driver. |
| [`original_predictor/`](original_predictor) | §III.B, Eq. 1 | The frozen global branch used inside the Residual Predictor: the CLS-token self-attention head, the interaction MLP, the three compact MLPs, and the ensemble that averages them. |
| [`predictor_ablation/`](predictor_ablation) | Table I, §Predictor Ablation | General model/config/training framework used to run the predictor-architecture ablation (CLS head, interaction MLP, two-head average, ESMC vs. ESM2, ...). |
| [`rc_tabm_variants/`](rc_tabm_variants) | Supplementary Information | Mean-TabM and Standalone CondPool-TabM comparison variants referenced alongside RC-TabM, plus the vendored TabM implementation they depend on. |
| [`extra_trees_baseline/`](extra_trees_baseline) | Table I / Table III | Extra Trees baseline (UniKP-style regression framework) used both as a turnover-prediction baseline and as the independent, non-differentiable re-scorer in §Transfer to an Independent Predictor. |
| [`structural_validation/`](structural_validation) | §Mutation Design (sequence recovery / TM-score / pLDDT) | ESMFold-based structural sanity check of PAMP designs. |
| [`cost_benefit_analysis/`](cost_benefit_analysis) | §Model Complexity Analysis | Exhaustive 19L-scan vs. PAMP (4 passes/round) cost-benefit comparison, including the standalone single-file PAMP/HotFlip attack implementation it is built on. |

## Provenance

Source workspace, for reference (not part of this repository):
- Most recent/canonical results: `rivermind-data/experiments/pamp_multi_predictor_v1/`
  (paths `A0_pamp`…`B2_random` in that code correspond to Single-gradient, HotFlip,
  PAMP-no-penalty, PAMP, and Random in the paper).
- Revision package with manuscript, figures, and evidence tables:
  `paper_revision/20260910T034935Z/`.
- A handful of shared, lower-level modules (the Original Predictor components,
  structural-validation pipeline, and cost-benefit scripts) lived at the root of
  `rivermind-data/` rather than under `experiments/`; they are reproduced here under
  `original_predictor/`, `structural_validation/`, and `cost_benefit_analysis/`.

Each subfolder's `ORIGINAL_README.md` (where present) is the README that shipped with
that experiment directory in the source workspace, kept verbatim for context.

## Notes on scope

- Code only. Trained weights, cached embeddings, and per-record prediction CSVs are
  intentionally excluded (multi-GB and not suitable for this repository); scripts
  reference their original absolute paths (e.g. `/root/rivermind-data/...`) since they
  were run in-place in the source workspace rather than packaged for standalone reuse.
- `rc_tabm_variants/vendor/tabm.py` and `rc_tabm_variants/vendor/rtdl_num_embeddings.py`
  are vendored third-party code (TabM, ICLR 2025, `yandex-research/tabm`); the original
  license header is preserved at the top of each file.
- Several earlier/superseded iterations of the mutation-search code exist in the source
  workspace (e.g. `run_PAMP_one_shot_ensemble_*.py`, `run_MS_PAMP_model2_raw_vs_subspace_validation.py`)
  predating the refactor into `pamp_multi_predictor_v1`; they were left out as they are
  not what produced the numbers reported in the paper.
