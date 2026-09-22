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
| [`method/search/`](pamp_search) | §III.C *Adversarial Mutation Search* (Fig. 1) | The PAMP algorithm itself: multi-point gradient averaging, distance-penalized projection, HotFlip/Random baselines, sequential multi-round design, and the driver scripts that produced Tables II and III. |
| [`method/residual_predictor/`](residual_predictor) | §III.B, *Residue-Conditioned TabM (RC-TabM)* | The complete Residual Predictor: frozen Original Predictor + substrate-conditioned residue correction (Eq. 2), and its training driver. |
| [`method/original_predictor/`](original_predictor) | §III.B, Eq. 1 | The frozen global branch used inside the Residual Predictor: the CLS-token self-attention head, the interaction MLP, the three compact MLPs, and the ensemble that averages them. |
| [`experiments/predictor_ablation/`](predictor_ablation) | Table I, §Predictor Ablation | General model/config/training framework used to run the predictor-architecture ablation (CLS head, interaction MLP, two-head average, ESMC vs. ESM2, ...). |
| [`experiments/rc_tabm_variants/`](rc_tabm_variants) | Supplementary Information | Mean-TabM and Standalone CondPool-TabM comparison variants referenced alongside RC-TabM. |
| [`third_party/tabm/`](third_party/tabm) | — | Vendored TabM implementation that `method/residual_predictor/` and `experiments/rc_tabm_variants/` depend on. |
| [`experiments/extra_trees_baseline/`](extra_trees_baseline) | Table I / Table III | Extra Trees baseline (UniKP-style regression framework) used both as a turnover-prediction baseline and as the independent, non-differentiable re-scorer in §Transfer to an Independent Predictor. |
| [`experiments/structural_validation/`](structural_validation) | §Mutation Design (sequence recovery / TM-score / pLDDT) | ESMFold-based structural sanity check of PAMP designs. |
| [`experiments/cost_benefit/`](cost_benefit_analysis) | §Model Complexity Analysis | Exhaustive 19L-scan vs. PAMP (4 passes/round) cost-benefit comparison, including the standalone single-file PAMP/HotFlip attack implementation it is built on. |
| [`data/`](data) | §Data and Code Availability | Placeholder + manifest for the data partitions, design-cohort membership, candidate-level predictions, and algorithm settings the paper's Data and Code Availability statement refers to; not yet released (see that folder's README). |
| [`docs/provenance/`](docs/provenance) | — | The original per-experiment READMEs from the source workspace, kept verbatim for context. |

## Method naming

The paper's method names (§Implementation Details, Table II/III) and the internal
identifiers used throughout `method/search/` and `experiments/cost_benefit/` differ. This is
the authoritative mapping:

| Code | Paper |
|---|---|
| `A0_pamp` | Single-gradient |
| `A1_hotflip` | HotFlip |
| `A2_fw_avg` | PAMP (no penalty) |
| `A3_fw_avg_pamp` | PAMP (with penalty) — "PAMP" without a qualifier elsewhere in the paper |
| `B2_random` | Random |
| `B0_esm_lm` | *Not reported in the paper.* An ESM2 masked-language-model scoring baseline was implemented and run alongside the others but is not one of the methods presented in the paper; treat it as exploratory, not as part of the reported results. |

## Provenance

Source workspace, for reference (not part of this repository):
- Most recent/canonical results: `rivermind-data/experiments/pamp_multi_predictor_v1/`.
- Revision package with manuscript, figures, and evidence tables:
  `paper_revision/20260910T034935Z/`.
- A handful of shared, lower-level modules (the Original Predictor components,
  structural-validation pipeline, and cost-benefit scripts) lived at the root of
  `rivermind-data/` rather than under `experiments/`; they are reproduced here under
  `method/original_predictor/`, `experiments/structural_validation/`, and `experiments/cost_benefit/`.

`docs/provenance/*.README.md` are the READMEs that shipped with each corresponding
experiment directory in the source workspace, kept verbatim for context.

## Setup

`requirements.txt` lists the Python dependencies these scripts import (PyTorch, ESM2 via
`fair-esm`, RDKit, scikit-learn, etc.). It is a manifest of what's needed, not a pinned,
tested environment — see the note above about hardcoded paths before expecting any
script to run standalone.

## Notes on scope

- Code only. Trained weights, cached embeddings, and per-record prediction CSVs are
  intentionally excluded (multi-GB and not suitable for this repository); see
  [`data/README.md`](data/README.md) for what those artifacts are and their release status.
- Scripts still reference their original absolute paths (e.g. `/root/rivermind-data/...`)
  since they were run in-place in the source workspace rather than packaged for
  standalone reuse. Making them portable (config- or CLI-driven paths, a proper
  `src/`/`experiments/`/`analysis/` layering across the whole repo) is planned as a
  follow-up after submission, to avoid touching import-sensitive code right now.
- `third_party/tabm/tabm.py` and `third_party/tabm/rtdl_num_embeddings.py` are vendored
  third-party code (TabM, ICLR 2025, `yandex-research/tabm`); the original license header
  is preserved at the top of each file. This repository's [`LICENSE`](LICENSE) (MIT)
  covers the rest of the code, not these vendored files.
- Several earlier/superseded iterations of the mutation-search code exist in the source
  workspace (e.g. `run_PAMP_one_shot_ensemble_*.py`, `run_MS_PAMP_model2_raw_vs_subspace_validation.py`)
  predating the refactor into `pamp_multi_predictor_v1`; they were left out as they are
  not what produced the numbers reported in the paper.

## Paths and configuration

Scripts no longer contain absolute paths from the original workspace. Every
location is resolved in [`pamp_paths.py`](pamp_paths.py) and can be overridden
with an environment variable:

```bash
export PAMP_DATA_ROOT=/path/to/workspace      # cached embeddings, splits, outputs
export PAMP_KCAT_CSV=/path/to/kcat-data_0.4simi-10fold.csv
export PAMP_ESM2_CHECKPOINT=/path/to/esm2_t33_650M_UR50D.pt
export PAMP_USALIGN=/path/to/USalign          # structural validation only
```

Defaults assume the artifacts listed in [`data/README.md`](data/README.md) sit
under `data/`.

## Method naming

The code predates the paper's terminology. The correspondence is:

| Code | Paper |
|---|---|
| `A0_pamp` | Single-gradient |
| `A1_hotflip` | HotFlip |
| `A2_fw_avg` | PAMP (no penalty) |
| `A3_fw_avg_pamp` | PAMP (with penalty) |
| `B2_random` | Random |
| `B0_esm_lm` | exploratory ESM2 masked-LM baseline, not reported in the paper |
