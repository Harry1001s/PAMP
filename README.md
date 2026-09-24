# PAMP

Continuous-to-Discrete Mutation Projection for Enzyme Engineering.

PAMP searches enzyme substitutions using averaged embedding gradients and a distance-penalized projection. The predictor combines a frozen global ensemble with a substrate-conditioned RC-TabM correction branch.

## Code

| Directory | Contents |
|---|---|
| [method/search](method/search/) | PAMP, Single-gradient, HotFlip, Random, and sequential mutation search |
| [method/original_predictor](method/original_predictor/) | CLS head, interaction MLP, compact MLPs, ensemble, and training |
| [method/residual_predictor](method/residual_predictor/) | RC-TabM model, residue batching, and training |
| [third_party/tabm](third_party/tabm/) | TabM implementation |
| [data/outputs](data/outputs/) | 13 core training and result files |

## Setup

```bash
pip install -r requirements.txt
export PAMP_DATA_ROOT=/path/to/workspace
export PAMP_REVISION_ROOT=/path/to/revision/package
export PAMP_KCAT_CSV=/path/to/kcat-data_0.4simi-10fold.csv
export PAMP_ESM2_CHECKPOINT=/path/to/esm2_t33_650M_UR50D.pt
```

The workspace supplies protein/substrate features, split indices, residue caches, and predictor checkpoints. Paths are configured in [pamp_paths.py](pamp_paths.py).

## Run

```bash
# Train the CLS and interaction heads.
python method/original_predictor/run_mean_kcat_comparison.py --out experiment_mean

# Train RC-TabM on the frozen global predictor.
python method/residual_predictor/train.py

# Run PAMP for two rounds.
python method/search/run.py --dataset catapro --methods a3 --out outputs/pamp

# Continue PAMP to five rounds.
python method/search/run_a3_round5.py --base outputs/pamp --out outputs/pamp_round5
```

Compact heads use `fit_run` in [train_compact_kcat.py](method/original_predictor/train_compact_kcat.py). Use `--methods all` to run the five-method comparison.

## Results

[File index and data fields](data/README.md)

- Six model-training histories, predictor configuration, ensemble manifests, evaluation metrics, and test predictions.
- [Mutation endpoints](data/outputs/mutation_endpoints.csv.gz): 2,754 pairs × five methods × three endpoints.
- [PAMP trajectories](data/outputs/pamp_five_rounds.csv.gz): 2,754 pairs × five rounds.

## License

[MIT](LICENSE). TabM files retain their upstream license headers.
