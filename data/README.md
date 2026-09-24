# Core experiment outputs

This directory contains **13 core output files** (plus this README). It includes the
training records of the five frozen heads actually used by the Original Predictor and
the residual correction branch, together with their essential results and mutant datasets.

## Training and predictor results

| File in `outputs/` | Contents |
|---|---|
| [cls_head_training.csv](outputs/cls_head_training.csv) | CLS head, seed 42: epoch-by-epoch training history. |
| [interaction_head_training.csv](outputs/interaction_head_training.csv) | Interaction MLP, seed 42: training history. |
| [compact_seed42_training.csv](outputs/compact_seed42_training.csv) | Protein + substrate compact MLP, seed 42, selected configuration `std_w128_d0.1_wd0.001`. |
| [compact_seed2024_training.csv](outputs/compact_seed2024_training.csv) | Protein + substrate compact MLP, seed 2024. |
| [compact_seed3407_training.csv](outputs/compact_seed3407_training.csv) | Protein + substrate compact MLP, seed 3407. |
| [predictor_training.csv](outputs/predictor_training.csv) | Residual Predictor / RC-TabM training history: baseline epoch 0 and epochs 1–127. |
| [predictor_config.json](outputs/predictor_config.json) | Residual Predictor architecture and training configuration. |
| [original_predictor_manifest.json](outputs/original_predictor_manifest.json) | Exact frozen global predictor: three compact heads and the two-head ensemble, including original checkpoint hashes. |
| [two_head_ensemble_manifest.json](outputs/two_head_ensemble_manifest.json) | CLS/interaction ensemble and checkpoint hashes. |
| [predictor_metrics.json](outputs/predictor_metrics.json) | Train/validation/test metrics, selected epoch 116, runtime, and baseline comparison. |
| [predictor_test_predictions.csv](outputs/predictor_test_predictions.csv) | Per-record predictions for the 2,766-record fixed test partition. |

These are the actual protein + substrate models used in the reported Residual Predictor.
The compact-head records are from `esm_plain` and its selected seed-42 search trial,
not the separate MACCS or low-rank comparison variants.

## Mutation datasets

| File | Contents |
|---|---|
| [mutation_endpoints.csv.gz](outputs/mutation_endpoints.csv.gz) | 41,310 endpoint records: 2,754 input pairs × five methods × three endpoints. |
| [pamp_five_rounds.csv.gz](outputs/pamp_five_rounds.csv.gz) | 13,770 records: 2,754 input pairs × five sequential PAMP rounds. |

Both gzip CSVs contain full reference and mutant sequences, substrate and SMILES,
substitutions, and saved predictor outputs. They can be read directly:

```python
import pandas as pd

endpoints = pd.read_csv("data/outputs/mutation_endpoints.csv.gz")
pamp_round2 = endpoints[
    (endpoints.method == "A3_fw_avg_pamp") & (endpoints.endpoint == "round2_top1")
]
five_rounds = pd.read_csv("data/outputs/pamp_five_rounds.csv.gz")
```

Methods are `A0_pamp` (Single-gradient), `A1_hotflip` (HotFlip), `A2_fw_avg`
(PAMP without penalty), `A3_fw_avg_pamp` (PAMP), and `B2_random` (Random).
The three endpoints are `round1_top1`, `round1_best_of_top5`, and `round2_top1`.
The second round continues from the first-ranked candidate at a different position;
best-of-Top5 is a separate single-mutant endpoint.

`row_id` is the zero-based source-table row index. `edits` is a JSON list of
substitutions such as `D351W`, using **one-based residue positions**.
`reference_sequence` is the input sequence and is not necessarily a biological wild type.
`wt_log2` and `mutant_log2` are saved log2(kcat) predictions; `delta_log2` is their
difference. In the five-round table, `increment_log2` compares adjacent rounds.
These are predicted changes, not measured mutant activities. All saved endpoints and
negative changes are retained.

## Sources and checks

Training histories, configurations, manifests, metrics, and test predictions were copied
without altering their contents from the original workspace:

- `rivermind-data/experiment_mean/{mean_cls_transformer,mean_interaction_mlp}/seed_42/`.
- `rivermind-data/experiment_compact_improvements/search/std_w128_d0.1_wd0.001/`
  and `esm_plain/{seed_2024,seed_3407}/`.
- `rivermind-data/experiments/catapro_residual_condpool/`.
- `rivermind-data/model_improvement_analysis/random_ensemble_manifest.json`.
- `paper_revision/20260910T034935Z/pamp_protein_substrate_v1/model_manifest.json`.

Mutation data originate from
`rivermind-data/experiments/pamp_multi_predictor_v1/results_no_esm_lm_2754/` and
`catapro_test_2754_residual_a3_round5_distinct/`. Full sequences were reconstructed
from the saved substitutions and joined to the original substrate metadata. Each
reference residue, distinct position, substitution count, saved final sequence, and
first/second-round prediction was checked for consistency. Predictions were not rerun.
The 2,754 design pairs represent 2,391 unique input sequences.

Historical paths in copied JSON files are provenance references, not portable paths.
Weights and embedding caches are external runtime requirements. These outputs preserve
the original feature pipeline, including its legacy substrate-encoding limitation;
they do not represent retraining with a corrected encoding.
