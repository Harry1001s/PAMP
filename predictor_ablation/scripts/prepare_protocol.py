#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from predictor_ablation.data import load_manifest
from predictor_ablation.reproducibility import atomic_json, environment_snapshot, sha256_json


def main():
    p = argparse.ArgumentParser(description="Resolve and lock a predictor ablation protocol.")
    p.add_argument("--config", default="configs/protocol.yaml"); p.add_argument("--experiment", required=True); p.add_argument("--resume", action="store_true")
    a = p.parse_args(); run = HERE / "runs" / a.experiment; run.mkdir(parents=True, exist_ok=True)
    lock = run / "protocol.lock.json"
    if lock.exists():
        if a.resume: print(lock.read_text()); return
        raise SystemExit(f"Protocol already locked: {lock}; use --resume")
    protocol = yaml.safe_load(Path(a.config).read_text()); manifest = load_manifest(HERE / "configs/data_manifest.json")
    audit = run / 'audit/leakage_audit.json'
    if not audit.exists() or json.loads(audit.read_text()).get('status') != 'passed':
        raise SystemExit('Protocol cannot lock until provenance/leakage audit passes')
    if manifest.get('target_unit_verified') is not True:
        raise SystemExit('Protocol cannot lock: source kcat unit not yet verified')
    if not (run / 'smoke/smoke_results.json').exists():
        raise SystemExit('Protocol cannot lock before successful smoke and pilot')
    resolved = {"protocol_id": "protocol_" + sha256_json({"protocol": protocol, "data_hash": manifest["data_hash"], "split_hash": manifest["split_hash"]})[:12],
                "locked_at_unix": time.time(), "protocol": protocol, "data_hash": manifest["data_hash"], "split_hash": manifest["split_hash"],
                "test_previously_observed": True, "environment": environment_snapshot()}
    atomic_json(lock, resolved); (HERE / "environment.lock.txt").write_text(json.dumps(resolved["environment"], indent=2) + "\n")
    print(json.dumps(resolved, indent=2, ensure_ascii=False))

if __name__ == "__main__": main()
