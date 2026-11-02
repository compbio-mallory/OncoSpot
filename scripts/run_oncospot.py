#!/usr/bin/env python3
"""Run OncoSpot by selecting a supported cancer type."""

import argparse
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

EXAMPLES = {
    "HGSOC": "hgsoc_validation_15k_hallmark_v1.yaml",
    "CRC": "crc_validation_15k.yaml",
    "HCC": "hcc_validation_15k_v1.yaml",
}


def main():
    parser = argparse.ArgumentParser(
        description="Run OncoSpot with automatic DEMON mode selection."
    )
    parser.add_argument(
        "--cancer", required=True, type=str.upper, choices=EXAMPLES
    )
    parser.add_argument("--cells", type=int, default=15000)
    parser.add_argument("--tumor-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--cna-burden",
        choices=["low", "default", "high"],
        default="default",
    )
    parser.add_argument(
        "--output", help="New output directory inside results/runs/"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check configuration without running the simulation",
    )
    args = parser.parse_args()

    template = ROOT / "configs" / "examples" / EXAMPLES[args.cancer]
    cfg = yaml.safe_load(template.read_text())

    cfg["cancer_type"] = args.cancer
    cfg.setdefault("simulation", {})["n_cells"] = args.cells
    if args.seed is not None:
        cfg["simulation"]["seed"] = args.seed

    cfg.setdefault("tissue", {})["tumor_fraction"] = args.tumor_fraction
    cfg.setdefault("customize", {})["cna_burden"] = args.cna_burden

    cfg.setdefault("observation", {})["default_technology"] = "visium"
    visium = cfg["observation"].setdefault("visium", {})
    visium["enabled"] = True
    visium.setdefault("geometry", {})[
        "target_median_cells_per_spot"
    ] = 2
    visium.setdefault("depth_reference", {})["parameter_dir"] = (
        "parameters/observation/visium/gse211956_sp4"
    )

    sys.path.insert(0, str(ROOT / "src"))
    from cancer_st_sim.config import resolve_config

    (ROOT / "results").mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="oncospot_config_", dir=ROOT / "results"
    ) as temporary_dir:
        config_path = Path(temporary_dir) / "input.yaml"
        config_path.write_text(
            yaml.safe_dump(cfg, sort_keys=False)
        )

        resolved, warnings = resolve_config(ROOT, config_path)

        print("Cancer:", resolved["cancer_type"])
        print("Cells:", resolved["simulation"]["n_cells"])
        print("Tumor fraction:", resolved["tissue"]["tumor_fraction"])
        print("DEMON mode:", resolved["demon"]["mode"])
        print("DEMON preset:", resolved["demon"]["parameter_file"])

        for warning in warnings:
            print("WARNING:", warning)

        if args.dry_run:
            return

        name = (
            f"{args.cancer.lower()}_"
            f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
        )
        output = (
            Path(args.output)
            if args.output
            else Path("results/runs") / name
        )
        if not output.is_absolute():
            output = ROOT / output

        output = output.resolve()
        run_root = (ROOT / "results" / "runs").resolve()

        if not output.is_relative_to(run_root) or output == run_root:
            parser.error("--output must be inside results/runs/")

        if output.exists():
            parser.error(f"Output directory already exists: {output}")

        env = os.environ.copy()
        env.setdefault("ONCOSPOT_PYTHON", sys.executable)

        subprocess.run(
            [
                "bash",
                str(ROOT / "scripts/pipeline/run_cancer_simulation.sh"),
                str(config_path),
                str(output),
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
