# Manuscript–artifact correspondence

Audited manuscript: **PAMP_latest_v3.tex**, *PAMP: Continuous-to-Discrete Mutation Projection for Enzyme Engineering*. Its SHA-256 is recorded in [`paper_results.json`](../data/outputs/paper_results.json). File paths below are relative to `data/outputs/` unless otherwise indicated.

## Methods and evidence

| Manuscript component | Released evidence |
|---|---|
| Original Predictor and ensemble equation | `method/original_predictor/ensemble.py`; two ensemble manifests; five head-training histories |
| RC-TabM correction and training | `method/residual_predictor/`; `predictor_config.json`; `predictor_training.csv`; `predictor_metrics.json` |
| Gradient averaging and penalized projection | `method/search/attack_adapter.py`; `paper_results.json → search_settings` |
| Fixed data partitions and design cohort | `split_manifest.csv.gz`, including reference sequences, substrates, targets, and membership |
| Table I: ESM2-based internal prediction comparisons | Six aligned prediction columns in `predictor_test_predictions.csv` |
| Table I: BRENDA sequence-unseen evaluation | `brenda_sequence_unseen.csv.gz` |
| Table II: search methods and candidate budgets | `mutation_candidates.csv.gz`; `mutation_endpoints.csv.gz` |
| Paired search comparisons | Endpoint records; `paper_results.json → computed.search_paired` and `archived_statistics.search_paired` |
| Substitution composition | Candidate edits; `paper_results.json → computed.substitutions_top1` |
| Table III and transfer comparisons | `extra_trees_transfer.csv.gz`; Extra Trees configuration and checkpoint hash in `paper_results.json` |
| Parameter counts and timing | `paper_results.json → parameter_counts`, `runtime`, and `training_environment` |
| Five-round sequential design | `pamp_five_rounds.csv.gz`; `paper_results.json → computed.five_rounds` |
| Mean-TabM and Standalone CondPool comparisons | Archived reports under `paper_results.json → supplementary_predictor_metrics` |

[`analysis/reproduce_tables.py`](../analysis/reproduce_tables.py) reconstructs the numerical tables from individual records and validates their cross-file correspondence. It also reproduces the transfer confidence intervals using 4,000 exact-sequence cluster-bootstrap draws with seed 0.

## Items requiring manuscript or source reconciliation

| Item | Audit finding |
|---|---|
| Table I, Original Predictor (ESMC) | The manuscript reports RMSE 3.1848 and Pearson r 0.7742. The corresponding completed training and prediction artifacts were not located. This row is not included among the released numerical comparisons. |
| Structural validation paragraph | Structural results are outside this release. The available structure workspace does not establish the paragraph's reported values for the 2,754-pair cohort. |
| Search-comparison confidence intervals | The original interval estimates are preserved. Their bootstrap seed and replicate count were absent from the available summary; the release verifies the paired point estimates and W/T/L counts. |
| Substitution frequencies | For PAMP Top1 on 2,754 pairs, the six listed exchange types account for 17.83%; substitutions introducing W account for 25.20%, and substitutions replacing C account for 14.78%. The paragraph's 16.7% and 8.6% values are lower bounds, rather than these cohort-specific frequencies. |
| Encoder-pass count | The manuscript counts three gradient states and one selected-mutant evaluation. `AttackRow.state` first encodes the sequence and then recomputes each chunk for its input gradient. Thus four logical states are not four literal encoder forward calls in the released implementation. The operation-count wording needs to specify this distinction. |
| Runtime scope | The 5,346.4739 s record covers rounds 3–5 over 2,754 pairs. It corresponds to 1,782.1580 s per round and 0.6471 s per record per round. |
| Data and Code Availability | The manuscript still describes public release as pending. The repository now publishes code and 18 derived data/record files; model checkpoints and encoder feature caches remain workspace inputs. |

The manuscript source is unchanged by this repository update.
