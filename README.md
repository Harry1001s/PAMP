# PAMP

**Continuous-to-Discrete Mutation Projection for Enzyme Engineering**

Proximal Adversarial Mutation Projection (PAMP) selects amino-acid substitutions by averaging input gradients at three embedding states and projecting the resulting direction onto legal substitutions with a squared-distance penalty. A substrate-conditioned Residual Predictor supplies the turnover objective.

## Method

The Original Predictor combines an attention-based CLS head, an interaction MLP, and three compact MLPs. Its output is

$$
f(h,s)=\frac{f_C(h,s)+f_I(h,s)}{3}+\frac{1}{9}\sum_{j=1}^{3}f_{M,j}(h,s).
$$

The Residual Predictor adds the mean of 16 Residue-Conditioned TabM (RC-TabM) members, scaled by a learned gate and the training-target standard deviation. The protein encoder is ESM2-650M; substrate representations are 1,024-dimensional UniKP features. Predictions are expressed as log₂(kcat / (1 s⁻¹)).

The first search round returns five ranked substitutions. Top1 follows the first-ranked candidate; best-of-Top5 selects the candidate with the highest Residual Predictor score. Sequential rounds update the sequence and select a previously unmodified position. Extra Trees re-scores the selected mutants for transfer evaluation.

## Repository

| Location | Contents |
|---|---|
| [method/search](method/search/) | PAMP, HotFlip, component ablations, Random, and sequential search |
| [method/original_predictor](method/original_predictor/) | Global prediction heads, training, and ensemble assembly |
| [method/residual_predictor](method/residual_predictor/) | RC-TabM architecture, residue batching, and training |
| [analysis/reproduce_tables.py](analysis/reproduce_tables.py) | Table reconstruction and data validation |
| [data/outputs](data/outputs/) | 18 core training, prediction, and mutation files |
| [data/README.md](data/README.md) | File index, units, and field definitions |
| [docs/MANUSCRIPT_MAP.md](docs/MANUSCRIPT_MAP.md) | Manuscript-to-artifact correspondence |
| [third_party/tabm](third_party/tabm/) | TabM implementation |

## Data and evaluation

The fixed partition contains 22,126 training, 2,766 validation, and 2,766 test enzyme–substrate pairs. Mutation design uses the 2,754 test pairs with sequences longer than 80 residues, representing 2,391 unique enzymes. External prediction evaluation contains 3,794 BRENDA records with exact sequences absent from the CataPro-derived source.

## Reproduce the reported tables

The released tables can be analyzed on CPU using NumPy and pandas:

```bash
python -m pip install numpy pandas
python analysis/reproduce_tables.py --check --bootstrap
python analysis/reproduce_tables.py --json
```

These commands reconstruct the available predictor comparisons, mutation-search table, transfer table, and five-round summaries. Validation checks file hashes, partitions, substitution sequences, candidate selection, paired estimates, and the archived Extra Trees bootstrap intervals.

## Train predictors and run mutation search

Install the model dependencies and configure the experiment workspace:

```bash
pip install -r requirements.txt
export PAMP_DATA_ROOT=/path/to/workspace
export PAMP_REVISION_ROOT=/path/to/revision/package
export PAMP_KCAT_CSV=/path/to/kcat-data_0.4simi-10fold.csv
export PAMP_ESM2_CHECKPOINT=/path/to/esm2_t33_650M_UR50D.pt
```

[pamp_paths.py](pamp_paths.py) defines workspace paths for cached protein/substrate features, residue representations, partition indices, and trained checkpoints. Checkpoint identities and architecture settings are recorded in the released manifests and `paper_results.json`.

```bash
# Train the global CLS and interaction heads.
python method/original_predictor/run_mean_kcat_comparison.py --out "$PAMP_DATA_ROOT/experiment_mean"

# Train the residue-conditioned correction.
python method/residual_predictor/train.py

# Evaluate all five mutation methods on the manuscript cohort.
python method/search/run.py --dataset catapro --methods all \
  --sources residual --min-length 81 --site-policy distinct --out outputs/comparison

# Continue the PAMP trajectory through rounds 3–5.
python method/search/run_a3_round5.py --base outputs/comparison --out outputs/pamp_round5
```

Compact heads use `fit_run` in [train_compact_kcat.py](method/original_predictor/train_compact_kcat.py), with seeds 42, 2024, and 3407. Use `--methods a3` for PAMP alone or add `--extra-trees` to re-score candidates with the configured Extra Trees checkpoint.

The recorded training environment used Python 3.9.25, PyTorch 2.7.1+cu118, CUDA 11.8, and an NVIDIA GeForce RTX 3090. Detailed configurations, source hashes, and the timing record for rounds 3–5 are included in [paper_results.json](data/outputs/paper_results.json).

## License

[MIT](LICENSE). The vendored TabM files retain their upstream license headers.
