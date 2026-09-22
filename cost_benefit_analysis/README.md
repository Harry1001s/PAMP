# Exhaustive-scan vs. PAMP cost-benefit analysis (§Model Complexity Analysis)

Supports: "an exhaustive scan requires 19L forward passes, which is 7,239 passes at the
median cohort length of 381 residues... PAMP performs four encoder passes regardless of
L, three gradient evaluations plus one encoding of the selected mutant."

- `run_compact_topk_adversarial_attack.py` — standalone, single-file implementation of
  the PAMP / HotFlip / Random mutation search (an earlier, self-contained counterpart to
  `../pamp_search/attack_adapter.py`) used here as the object under comparison against
  exhaustive enumeration.
- `run_pamp_hotflip_enum_costbenefit.py` — runs both the exhaustive 19L single-substitution
  scan and PAMP/HotFlip on the same cohort and records wall-clock cost alongside the
  score of the best candidate found by each.
- `summarize_pamp_hotflip_enum_costbenefit.py` — aggregates and plots the cost-vs-benefit
  comparison (encoder passes / wall-clock time vs. predicted gain).
