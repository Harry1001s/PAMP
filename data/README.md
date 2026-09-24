# Experiment data

The release contains **18 core output files**. Training histories retain the recorded epochs; prediction and mutation tables retain individual enzyme–substrate records. All turnover targets and predictions use log₂(kcat / (1 s⁻¹)).

## File index

| File | Records or content | Manuscript correspondence |
|---|---|---|
| [cls_head_training.csv](outputs/cls_head_training.csv) | CLS head, seed 42 | Original Predictor training |
| [interaction_head_training.csv](outputs/interaction_head_training.csv) | Interaction MLP, seed 42 | Original Predictor training |
| [compact_seed42_training.csv](outputs/compact_seed42_training.csv) | Compact MLP, seed 42 | Original Predictor training |
| [compact_seed2024_training.csv](outputs/compact_seed2024_training.csv) | Compact MLP, seed 2024 | Original Predictor training |
| [compact_seed3407_training.csv](outputs/compact_seed3407_training.csv) | Compact MLP, seed 3407 | Original Predictor training |
| [predictor_training.csv](outputs/predictor_training.csv) | RC-TabM, epochs 0–127 | Residual Predictor training |
| [predictor_config.json](outputs/predictor_config.json) | RC-TabM architecture and optimization | Implementation details |
| [original_predictor_manifest.json](outputs/original_predictor_manifest.json) | Ensemble weights and component checkpoint hashes | Original Predictor |
| [two_head_ensemble_manifest.json](outputs/two_head_ensemble_manifest.json) | CLS/interaction ensemble and checkpoint hashes | Predictor ablation |
| [predictor_metrics.json](outputs/predictor_metrics.json) | Partition metrics; selected epoch 116 | Residual Predictor evaluation |
| [predictor_test_predictions.csv](outputs/predictor_test_predictions.csv) | 2,766 records; six predictors | Table I, internal test |
| [mutation_endpoints.csv.gz](outputs/mutation_endpoints.csv.gz) | 41,310 records: 2,754 pairs × 5 methods × 3 endpoints | Table II |
| [pamp_five_rounds.csv.gz](outputs/pamp_five_rounds.csv.gz) | 13,770 records: 2,754 pairs × 5 rounds | Sequential design |
| [split_manifest.csv.gz](outputs/split_manifest.csv.gz) | 27,658 source records, targets, fixed partitions, and design-cohort membership | Dataset preparation |
| [mutation_candidates.csv.gz](outputs/mutation_candidates.csv.gz) | 82,620 records: 5 first-round candidates and 1 second-round candidate per pair and method | Candidate-budget evaluation |
| [brenda_sequence_unseen.csv.gz](outputs/brenda_sequence_unseen.csv.gz) | 3,794 records with sequence-unseen prediction results | Table I, external evaluation |
| [extra_trees_transfer.csv.gz](outputs/extra_trees_transfer.csv.gz) | 27,540 records: 2,754 pairs × 5 methods × 2 first-round endpoints | Table III and transfer analysis |
| [paper_results.json](outputs/paper_results.json) | Recomputed tables, archived intervals, settings, timing, supplementary predictor metrics, and SHA-256 provenance | Cross-reference and numerical summary |

## Identifiers and prediction fields

`row_id` is the zero-based row index in the CataPro-derived source table. It joins the partition, internal prediction, candidate, endpoint, sequential, and transfer tables. `pair_key` identifies a normalized sequence–substrate pair. The `fold` field preserves the source-table annotation; `split` specifies the fixed training/validation/test partition used here. `design_cohort` identifies test records with sequence length greater than 80.

External BRENDA records use the independent identifier `brenda_row`. Their `pair_key` is computed using the same pair-identity convention. `eligible_sequence_novel` marks the sequence-unseen evaluation cohort.

| Field | Definition |
|---|---|
| `y_true` / `y_log2` | Recorded turnover target |
| `y_cls`, `y_interaction`, `y_two_head` | CLS, interaction, and equal-weight two-head predictions |
| `y_global` | Original Predictor (ESM2) prediction |
| `y_pred` | Complete Residual Predictor prediction |
| `y_extra_trees` | Extra Trees prediction |
| `delta_local` | Scaled correction added to the global prediction |
| `reference_sequence`, `mutant_sequence` | Complete input and designed sequences |
| `substrate`, `smiles`, `canonical_smiles` | Substrate identity and molecular representations |
| `edits` | JSON list of one-based substitutions, e.g. `["D351W"]` |
| `candidate_rank` | Rank assigned by the proposal procedure |
| `net_substitutions` | Number of positions differing from the original reference |
| `wt_log2`, `mutant_log2` | Reference and mutant predictions used in the design experiment |
| `delta_log2` | Mutant prediction minus original-reference prediction |
| `increment_log2` | Change relative to the preceding sequential round |
| `res_delta`, `et_delta` | Change under the Residual Predictor and Extra Trees |
| `wt_stored`, `wt_res`, `wt_et` | Stored reference score and reference scores used in transfer evaluation |
| `stored_res_delta` | Residual change copied from the frozen endpoint selection |

Prediction evaluation uses the saved global predictions plus the correction evaluated with BF16 autocasting. Mutation search uses FP32 scoring and an FP16 residue anchor, as recorded in `paper_results.json`.

## Methods and endpoints

| Method identifier | Manuscript name | Gradient states | Distance penalty |
|---|---|---:|---|
| `A0_pamp` | Single-gradient | 1 | Yes |
| `A1_hotflip` | HotFlip | 1 | No |
| `A2_fw_avg` | PAMP (no penalty) | 3 | No |
| `A3_fw_avg_pamp` | PAMP | 3 | Yes |
| `B2_random` | Random | 0 | No |

- `round1_top1`: the first-ranked single substitution.
- `round1_best_of_top5`: the single substitution with the highest Residual Predictor score among five proposals, with at most two proposals per position.
- `round2_top1`: a second substitution at a distinct position, selected after updating to the first-ranked mutant.

Five-round trajectories extend Top1 sequentially at distinct positions. Transfer evaluation retains the Residual Predictor's selected candidate at each endpoint. A positive outcome is a predicted change greater than zero. Fractions are stored on a 0–1 scale; multiply by 100 to obtain percentages or percentage-point differences.

## Read and validate

```python
import pandas as pd

partitions = pd.read_csv("data/outputs/split_manifest.csv.gz")
candidates = pd.read_csv("data/outputs/mutation_candidates.csv.gz")
pamp_top5 = candidates.query("method == 'A3_fw_avg_pamp' and regime == 'round1_top5'")
pamp_top5 = pamp_top5.merge(
    partitions[["row_id", "reference_sequence", "substrate", "canonical_smiles"]],
    on="row_id", validate="many_to_one",
)
```

```bash
python analysis/reproduce_tables.py --check --bootstrap
```

`paper_results.json` stores source hashes under `source_sha256`, released-file hashes under `files`, derived numerical tables under `computed`, and original confidence intervals under `archived_statistics`. Historical workspace paths in provenance fields identify the source artifacts. The table-reconstruction command reads only the files in this directory.
