#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)

from cancer_st_sim.spatial import (  # noqa: E402
    write_tissue_compartments,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assign malignant and non-malignant "
            "tissue compartments."
        )
    )

    parser.add_argument(
        "--run-dir",
        required=True,
    )

    args = parser.parse_args()

    run_dir = Path(
        args.run_dir
    ).resolve()

    resolved_path = (
        run_dir
        / "resolved_config.yaml"
    )

    cells_path = (
        run_dir
        / "cells"
        / "cells.tsv"
    )

    if not resolved_path.exists():
        raise SystemExit(
            f"Missing resolved config: {resolved_path}"
        )

    if not cells_path.exists():
        raise SystemExit(
            f"Missing cell scaffold: {cells_path}"
        )

    with resolved_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        resolved = yaml.safe_load(
            handle
        )

    tumor_fraction = float(
        resolved[
            "tissue"
        ][
            "tumor_fraction"
        ]
    )

    seed = int(
        resolved[
            "simulation"
        ][
            "seed"
        ]
    )

    parameter_file = (
        resolved
        .get(
            "tme",
            {},
        )
        .get(
            "compartment_parameter_file"
        )
    )

    if not parameter_file:
        raise SystemExit(
            "Resolved configuration does not define "
            "tme.compartment_parameter_file."
        )

    preset_path = (
        PROJECT_ROOT
        / parameter_file
    )

    result = write_tissue_compartments(
        cells_path=cells_path,
        output_dir=(
            run_dir
            / "cells"
        ),
        tumor_fraction=tumor_fraction,
        seed=seed,
        preset_path=preset_path,
    )

    summary = result.summary

    print()
    print("=" * 80)
    print("TISSUE COMPARTMENT ASSIGNMENT COMPLETE")
    print("=" * 80)

    print(
        "Total cells:",
        f'{summary["n_total_cells"]:,}',
    )

    print(
        "Malignant:",
        f'{summary["actual_malignant_cells"]:,}',
    )

    print(
        "Non-malignant:",
        f'{summary["actual_nonmalignant_cells"]:,}',
    )

    print(
        "Tumor fraction:",
        f'{summary["actual_tumor_fraction"]:.4f}',
    )

    print(
        "Demes with non-malignant cells:",
        summary[
            "n_demes_with_nonmalignant_cells"
        ],
    )

    print(
        "Mixed demes:",
        summary[
            "n_demes_mixed"
        ],
    )

    print(
        "Fully malignant demes:",
        summary[
            "n_demes_fully_malignant"
        ],
    )

    print(
        "Exact counts conserved:",
        summary[
            "exact_cell_count_conserved"
        ],
    )

    print()


if __name__ == "__main__":
    main()
