"""Central path resolution for this repository.

Every location the scripts need is resolved here instead of being hardcoded.
Override any of them with an environment variable, or edit the defaults below.

    export PAMP_DATA_ROOT=/path/to/workspace
    export PAMP_KCAT_CSV=/path/to/kcat-data_0.4simi-10fold.csv

The defaults assume the artifacts described in data/README.md have been placed
under ``data/`` at the repository root.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _path(env: str, default) -> Path:
    return Path(os.environ.get(env, str(default))).expanduser()


def bootstrap() -> Path:
    """Put the repository root on ``sys.path`` so scripts can import this module.

    Call from a script that lives in a subdirectory::

        import pamp_paths  # after bootstrap, or use the snippet in the README
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return REPO_ROOT


# Workspace that held the cached embeddings, splits and experiment outputs.
DATA_ROOT = _path("PAMP_DATA_ROOT", REPO_ROOT / "data")

# Manuscript revision package (Extra Trees checkpoint, earlier result tables).
REVISION_ROOT = _path("PAMP_REVISION_ROOT", DATA_ROOT / "paper_revision")

# Source kinetics table (CataPro-derived, 0.4 sequence-identity 10-fold file).
KCAT_CSV = _path("PAMP_KCAT_CSV", DATA_ROOT / "kcat-data_0.4simi-10fold.csv")

# Local ESM2-650M checkpoint consumed by esm.pretrained.load_model_and_alphabet_local.
ESM2_CHECKPOINT = _path(
    "PAMP_ESM2_CHECKPOINT",
    Path.home() / ".cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt",
)

# Derived locations used by more than one script.
EXPERIMENT_MEAN = DATA_ROOT / "experiment_mean"
SPLIT_INDICES = EXPERIMENT_MEAN / "split_indices.npz"
DATA_MANIFEST = EXPERIMENT_MEAN / "data_manifest.csv"
PROTEIN_SUBSTRATE_V1 = REVISION_ROOT / "pamp_protein_substrate_v1"
EXTRA_TREES_MODEL = (
    REVISION_ROOT / "extra_trees_quick_v1/features_2304/model_seed_42.joblib"
)


# --- in-repo source directories -------------------------------------------
# Several scripts import sibling modules by bare name (``from common import *``,
# ``from tabm import TabM``). These constants let them put the right in-repo
# directory on sys.path instead of a path from the original workspace.
SEARCH_DIR = REPO_ROOT / "method/search"
RESIDUAL_DIR = REPO_ROOT / "method/residual_predictor"
ORIGINAL_PREDICTOR_DIR = REPO_ROOT / "method/original_predictor"
TABM_VENDOR = REPO_ROOT / "third_party/tabm"


def add_module_paths(*dirs) -> None:
    """Put in-repo source directories on ``sys.path`` for by-name imports."""
    for d in dirs:
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)
