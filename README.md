# PAMP: Continuous-to-Discrete Mutation Projection for Enzyme Engineering

This repository contains the core method code and **13 core experiment output files**
for *PAMP: Continuous-to-Discrete Mutation Projection for Enzyme Engineering*
(`PAMP_latest_v3.tex`). Outputs cover the models actually used, their training records,
predictor evaluation, and complete mutant sequences. Model weights and embedding
caches are external runtime requirements.

## Core outputs

See **[data/README.md](data/README.md)** for the complete index and field definitions.

- Six training histories: CLS head, interaction MLP, three compact MLP seeds,
  and the Residual Predictor correction branch.
- Predictor configuration, two ensemble manifests, evaluation metrics,
  and test predictions.
- [Mutation endpoints](data/outputs/mutation_endpoints.csv.gz): 2,754 input pairs,
  five methods, and three endpoints, including full mutant sequences.
- [PAMP five-round dataset](data/outputs/pamp_five_rounds.csv.gz): all five sequential
  mutant sequences for each input pair.

Mutation gains are saved predictor estimates, not measured mutant activities.

## Layout, mapped to the paper

| Folder | Paper section | Contents |
|---|---|---|
| [method/search/](method/search/) | §III.C, Tables II/III | PAMP gradient averaging, distance-penalized projection, HotFlip/Random baselines, sequential design, and transfer evaluation. |
| [method/residual_predictor/](method/residual_predictor/) | §III.B, Eq. 2 | Frozen Original Predictor plus substrate-conditioned residue correction (RC-TabM) and its training driver. |
| [method/original_predictor/](method/original_predictor/) | §III.B, Eq. 1 | CLS head, interaction MLP, compact MLPs, and frozen global ensemble. |
| [experiments/predictor_ablation/](experiments/predictor_ablation/) | Predictor architecture analysis | Configurable training and audit framework. |
| [experiments/rc_tabm_variants/](experiments/rc_tabm_variants/) | Supplementary comparisons | Mean-TabM and standalone CondPool-TabM variants. |
| [experiments/extra_trees_baseline/](experiments/extra_trees_baseline/) | Tables I/III | Independent Extra Trees turnover predictor and mutation re-scorer. |
| [experiments/structural_validation/](experiments/structural_validation/) | Structural evaluation | ESMFold sequence recovery, TM-score, and pLDDT analysis. |
| [third_party/tabm/](third_party/tabm/) | — | Vendored TabM implementation and numerical embeddings. |
| [exploratory/cost_benefit/](exploratory/cost_benefit/) | Exploratory | Enumeration versus gradient search against the Original Predictor. |
| [data/outputs/](data/outputs/) | Core output release | Training records, predictor results, and mutation datasets. |

## Method naming

| Code | Paper |
|---|---|
| `A0_pamp` | Single-gradient |
| `A1_hotflip` | HotFlip |
| `A2_fw_avg` | PAMP (no penalty) |
| `A3_fw_avg_pamp` | PAMP (with penalty) |
| `B2_random` | Random |
| `B0_esm_lm` | Exploratory ESM2 masked-LM baseline, not included in the primary result release. |

## Setup and paths

`requirements.txt` lists dependencies imported by the experiment scripts; it is not
a pinned environment. Runtime artifact paths are resolved through
[pamp_paths.py](pamp_paths.py):

```bash
export PAMP_DATA_ROOT=/path/to/workspace
export PAMP_REVISION_ROOT=/path/to/revision/package
export PAMP_KCAT_CSV=/path/to/kcat-data_0.4simi-10fold.csv
export PAMP_ESM2_CHECKPOINT=/path/to/esm2_t33_650M_UR50D.pt
export PAMP_USALIGN=/path/to/USalign
```

`data/outputs/` contains saved results for inspection and reuse. Training and inference
require the full runtime workspace and external weights/feature caches. Original
manifests retain historical paths and hashes; see [data/README.md](data/README.md)
for their sources.

Vendored TabM files preserve their upstream license headers. The repository's
[LICENSE](LICENSE) covers the remaining code and does not replace upstream data terms.
