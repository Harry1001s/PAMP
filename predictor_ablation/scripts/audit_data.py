#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from src.data import audit_manifest, derive_protein_pooling, load_manifest
from src.reproducibility import atomic_json


def args():
    p = argparse.ArgumentParser(description="Audit cache alignment/leakage and derive fixed protein pooling without encoders.")
    p.add_argument("--manifest", default="configs/data_manifest.json")
    p.add_argument("--experiment", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-derived-features", action="store_true")
    return p.parse_args()


def main():
    a = args(); manifest_path = Path(a.manifest).resolve(); manifest = load_manifest(manifest_path)
    run = HERE / "runs" / a.experiment; audit_dir = run / "audit"; audit_dir.mkdir(parents=True, exist_ok=True)
    result = audit_manifest(manifest)
    if result["hard_errors"]:
        atomic_json(audit_dir / "leakage_audit.json", result); raise SystemExit("Audit failed: " + ", ".join(result["hard_errors"]))
    if not a.skip_derived_features:
        existing = manifest.get("derived_protein_features", {})
        paths_exist = existing and all(Path(v["path"]).exists() for v in existing.values())
        if not (a.resume and paths_exist):
            derived = derive_protein_pooling(manifest, run / "derived_features/protein.npy", ("mean", "max"))
            manifest["derived_protein_features"] = derived; atomic_json(manifest_path, manifest)
    atomic_json(audit_dir / "leakage_audit.json", result)
    (audit_dir / "leakage_audit.md").write_text(
        f"# 数据与泄漏审计\n\n状态：{result['status']}\n\n硬错误：{result['hard_errors'] or '无'}\n\n"
        "当前表没有原始 measurement ID 或 sequence cluster，因此无法证明同源独立；历史 test 已被观察。\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__": main()
