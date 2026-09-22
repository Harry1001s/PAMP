# Data

This repository is code-only (see the top-level README). The paper's *Data and Code
Availability* section states:

> The CataPro dataset and associated code are available through the original study and
> its public repository. The revision package includes data partitions, design-cohort
> membership, candidate-level predictions, algorithm settings, and analysis scripts.
> Public release of the present model artifacts and derived records remains pending,
> subject to the applicable data-sharing permissions.

Concretely, four data products are referenced by the code in this repository but not
(yet) included in it:

| Artifact | Produced/consumed by | Status |
|---|---|---|
| **Data partitions** — the 80:10:10 train/val/test split (22,126 / 2,766 / 2,766 records) over the CataPro-derived 27,658-record enzyme–substrate dataset | `method/original_predictor/`, `method/residual_predictor/`, `experiments/extra_trees_baseline/` | Not released here |
| **Design-cohort membership** — the 2,754-pair filtered test cohort (2,391 unique enzymes, 103–2,432 residues) used for mutation design | `method/search/` | Not released here |
| **Candidate-level predictions** — per-record, per-method proposed substitutions and predicted `Δlog2(kcat)` (Tables II/III) | `method/search/build_paper_results.py` and related aggregation scripts | Not released here |
| **Algorithm settings** — frozen hyperparameters/config locked in for the reported runs (e.g. `method/residual_predictor/config.json`, `extra_trees_baseline`'s selected parameters) | training/search scripts throughout | Partially included where small (e.g. `method/residual_predictor/config.json`); larger per-run manifests not included |

The upstream CataPro dataset itself (drawing on BRENDA and SABIO-RK) is available
through the original CataPro study and its public repository, not through this one.

If/when the model artifacts and derived records referenced above are cleared for public
release, they belong under this directory, organized to match the table above.
