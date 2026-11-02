#!/usr/bin/env python3
"""Run OncoSpot reference-based parameter calibration."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cancer", required=True)
    p.add_argument("--counts", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--gene-mapping", required=True)
    p.add_argument("--gmt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--beta", type=float, default=0.4)
    p.add_argument("--min-cells", type=int, default=20)
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    cal = root / "calibration"
    out = Path(args.output).resolve()

    shared = [
        "--counts", args.counts,
        "--annotations", args.annotations,
        "--gene-mapping", args.gene_mapping,
        "--output", str(out),
        "--cancer", args.cancer,
    ]

    commands = [
        [
            sys.executable,
            str(cal / "02_estimate_baselines.py"),
            *shared,
            "--min-cells", str(args.min_cells),
        ],
        [
            sys.executable,
            str(cal / "03_estimate_malignant.py"),
            *shared,
            "--origin-baseline",
            str(out / "malignant/cell_of_origin/origin_baseline.tsv"),
            "--gmt", args.gmt,
            "--beta", str(args.beta),
        ],
    ]

    for i, command in enumerate(commands, 1):
        print(f"\n=== Calibration stage {i}/2 ===", flush=True)
        subprocess.run(command, check=True)

    required = [
        "malignant/cell_of_origin/origin_baseline.tsv",
        "malignant/expression/malignant_expression_parameters.tsv",
        "malignant/expression/malignant_library_sizes.tsv",
        "nonmalignant/expression/tme_expression_baselines.tsv",
        "nonmalignant/expression/library_sizes_by_cell_type.tsv",
        "nonmalignant/expression/tme_dispersion.yaml",
        "tumor_states/state_reference.tsv",
    ]

    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing calibration outputs: {missing}")

    print("\nONCOSPOT PARAMETER CALIBRATION COMPLETE")
    print("Output:", out)
    print("Status: provisional; simulation validation still required")


if __name__ == "__main__":
    main()
