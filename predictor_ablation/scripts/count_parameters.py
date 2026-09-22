#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
from predictor_ablation.config import a0, capacity_grid


def main():
    p=argparse.ArgumentParser(description="Instantiate and count all required concat capacity models.")
    p.add_argument("--experiment",required=True); p.add_argument("--all-required",action="store_true"); p.add_argument("--resume",action="store_true")
    a=p.parse_args(); rows=[a0(HERE)]+capacity_grid(HERE); out=HERE/"runs"/a.experiment/"exports"; out.mkdir(parents=True,exist_ok=True)
    fields=["config_id","family","stage","budget_id","target_params","trainable_params","depth","hidden_dims","parameter_error_fraction","budget_match_status"]
    with (out/"parameter_count.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields,extrasaction="ignore");w.writeheader()
        for row in rows:
            r=dict(row);r["hidden_dims"]=json.dumps(r["hidden_dims"]);w.writerow(r)
    for row in rows: print(row["config_id"],row["budget_id"],row["depth"],row["hidden_dims"],row["trainable_params"],row["budget_match_status"])

if __name__=="__main__":main()
