# Core outputs

13 files containing model training records and mutation results.

| File | Contents |
|---|---|
| [cls_head_training.csv](outputs/cls_head_training.csv) | CLS head, seed 42 |
| [interaction_head_training.csv](outputs/interaction_head_training.csv) | Interaction MLP, seed 42 |
| [compact_seed42_training.csv](outputs/compact_seed42_training.csv) | Compact MLP, seed 42 |
| [compact_seed2024_training.csv](outputs/compact_seed2024_training.csv) | Compact MLP, seed 2024 |
| [compact_seed3407_training.csv](outputs/compact_seed3407_training.csv) | Compact MLP, seed 3407 |
| [predictor_training.csv](outputs/predictor_training.csv) | RC-TabM, epochs 0–127 |
| [predictor_config.json](outputs/predictor_config.json) | Predictor architecture and training settings |
| [original_predictor_manifest.json](outputs/original_predictor_manifest.json) | Global ensemble components and checkpoint hashes |
| [two_head_ensemble_manifest.json](outputs/two_head_ensemble_manifest.json) | CLS/interaction ensemble configuration |
| [predictor_metrics.json](outputs/predictor_metrics.json) | Train/validation/test metrics; selected epoch 116 |
| [predictor_test_predictions.csv](outputs/predictor_test_predictions.csv) | 2,766 test predictions |
| [mutation_endpoints.csv.gz](outputs/mutation_endpoints.csv.gz) | 41,310 mutation endpoint records |
| [pamp_five_rounds.csv.gz](outputs/pamp_five_rounds.csv.gz) | 13,770 sequential PAMP records |

## Read the datasets

```python
import pandas as pd

endpoints = pd.read_csv("data/outputs/mutation_endpoints.csv.gz")
five_rounds = pd.read_csv("data/outputs/pamp_five_rounds.csv.gz")
pamp_round2 = endpoints[
    (endpoints.method == "A3_fw_avg_pamp") & (endpoints.endpoint == "round2_top1")
]
```

## Fields

- `row_id`: zero-based source-table row index.
- `reference_sequence`, `mutant_sequence`: complete input and designed sequences.
- `substrate`, `smiles`, `canonical_smiles`: substrate identity and representation.
- `edits`: JSON list of substitutions with one-based residue positions, such as `D351W`.
- `wt_log2`, `mutant_log2`: predicted log2(kcat) before and after mutation.
- `delta_log2`: mutant prediction minus reference prediction.
- `increment_log2`: change from the preceding round.

`round1_top1` is the first-ranked single mutant; `round1_best_of_top5` is the highest-scoring evaluated single mutant; `round2_top1` applies a second substitution at a distinct position to `round1_top1`. The five-round dataset continues this sequence at five distinct positions.

| Method | Name |
|---|---|
| `A0_pamp` | Single-gradient |
| `A1_hotflip` | HotFlip |
| `A2_fw_avg` | PAMP without penalty |
| `A3_fw_avg_pamp` | PAMP |
| `B2_random` | Random |
