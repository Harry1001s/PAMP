# Structural validation of PAMP designs (§Mutation Design)

Supports the paper's structural sanity check: "Sequence recovery relative to the
original reference averaged 0.9966 after one round and 0.9933 after two... Structures
predicted with ESMFold gave a mean TM-score of 0.9872 after one round and 0.9819 after
two, and mean pLDDT changed by only -0.1887 and -0.2872 points."

- `run_pamp_structure_pipeline.py` — top-level orchestrator: waits for the ESMFold
  checkpoint download, then runs a length-stratified pilot followed by the full
  evaluation.
- `run_pamp_structure_preservation.py` — resumable ESMFold-v1 structural evaluation:
  folds wild-type and mutant sequences, computes TM-score and pLDDT deltas, and journals
  progress so an interrupted run can resume without repeating folds.
- `pamp_structure_statistics.py` — aggregates per-pair fold results (sequence recovery,
  TM-score, pLDDT) into the summary statistics reported in the paper.
- `test_pamp_structure_preservation.py` — unit tests for the two modules above.
