# Exhaustive-scan vs. gradient-search cost-benefit study (exploratory)

**No number in the manuscript comes from this directory.** It is kept for
reference because it is the only place an exhaustive `19L` enumeration is
implemented.

## What it actually ran

These scripts score candidates with the **Original Predictor**
(`ImprovedKcatEnsemble`, internal R^2 = 0.6187), not the Residual Predictor that
the paper uses as its design objective. `run_pamp_hotflip_enum_costbenefit.py`
imports `run_compact_topk_adversarial_attack` and therefore inherits the same
model. The study was run before the Residual Predictor became the objective, and
its numbers are not comparable to anything reported in the manuscript.

For the manuscript's cost figures, see `method/search/speed_a1_a3.py`, which
drives `attack_adapter.py` and loads the Residual Predictor.

## Files

- `run_compact_topk_adversarial_attack.py` — standalone, single-file PAMP /
  HotFlip / Random search against the Original Predictor. An earlier,
  self-contained counterpart to `method/search/attack_adapter.py`.
- `run_pamp_hotflip_enum_costbenefit.py` — runs the exhaustive `19L` scan and the
  gradient searches on the same cohort, recording wall-clock cost alongside the
  best candidate each one finds. The enumeration table acts as an oracle, so the
  rank, regret and percentile of each search's pick are exact.
- `summarize_pamp_hotflip_enum_costbenefit.py` — aggregates that comparison.

## To make this usable for the paper

Re-run `run_pamp_hotflip_enum_costbenefit.py` with the Residual Predictor in
place of `ImprovedKcatEnsemble`, so the enumeration oracle and the search share
the objective the paper optimizes. That would yield the approximation-quality
evidence the manuscript currently lacks: where PAMP's Top1 falls in the exact
ranking, how often the five-candidate shortlist contains the true optimum, and
the matched-hardware wall-clock ratio against direct screening.
