# PAMP mutation search (§III.C)

Implements the adversarial mutation search described in the paper: multi-point gradient
averaging at the reference embedding and two half-step intermediate states, followed by
distance-penalized projection onto the legal single-substitution set.

- `attack_adapter.py` — core algorithm. `Engine` wraps the frozen ESM2 encoder and the
  Residual Predictor; `AttackRow` implements, per enzyme–substrate pair:
  - `state()` — encode a sequence (optionally with an embedding-space delta) and its
    input gradient w.r.t. the objective `Φ` (Eq. 3).
  - `path()` — the two-step multi-point gradient averaging walk (Eq. 4): at each step,
    pick the best legal vertex by unpenalized directional score, take a half-step
    towards it, and recompute the gradient.
  - `scores()` — the distance-penalized projection score `S(i,a)` (Eq. 5/6), or its
    unpenalized HotFlip-style variant.
  - `proposals()` — evaluates all six methods in `METHODS` for one round: PAMP with and
    without the penalty (`A3_fw_avg_pamp`, `A2_fw_avg`), single-gradient variants
    (`A0_pamp`, `A1_hotflip`/HotFlip), the ESM2 masked-LM baseline (`B0_esm_lm`), and
    Random (`B2_random`).
- `run.py` — main experiment driver: loads a cohort (CataPro test partition or BRENDA
  sequence-unseen set), runs Top1/Top5 proposals, applies the winning substitution, and
  re-encodes to score it with both the Residual Predictor and Extra Trees (the transfer
  evaluator in §Transfer to an Independent Predictor).
- `run_a3_round5.py` — five-round sequential design (§Five-Round Sequential Mutation
  Design): repeatedly applies PAMP while excluding already-mutated positions.
- `run_long58.py` — same procedure restricted to a cohort of long sequences, used for
  wall-clock timing.
- `speed_a1_a3.py` — per-round timing measurement (§Model Complexity Analysis, the
  5,346.47 s / 1,782.16 s / 0.65 s figures).
- `build_paper_results.py` — aggregates raw per-record outputs into the paper's summary
  statistics and paired cluster-bootstrap comparisons (Tables II and III).
- `transfer_extratrees_2754.py` / `summarize_transfer_extratrees_2754.py` — re-score
  every first-round design with the independent Extra Trees baseline and summarize the
  retained-gain figures in Table III.
- `export_results_no_esm_lm.py`, `plot_boxplots_no_esm_lm.py` — figure/table export for
  the method comparison excluding the ESM2 masked-LM baseline.
- `launch_catapro_a3.py`, `launch_remaining.py` — job launchers for running the above
  over the full cohort in batches/background processes.

Depends on `residual_predictor/residual_model.py` (`load_predictor`) for the scoring
objective, and on the Extra Trees model trained in `extra_trees_baseline/`.
