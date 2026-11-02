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
    write_nonmalignant_cell_types,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assign broad normal/TME cell types "
            "to non-malignant cells."
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
        / "cells_with_compartment.tsv"
    )

    if not resolved_path.exists():
        raise SystemExit(
            f"Missing resolved config: {resolved_path}"
        )

    if not cells_path.exists():
        raise SystemExit(
            f"Missing compartment table: {cells_path}"
        )

    with resolved_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        resolved = yaml.safe_load(
            handle
        )

    parameter_file = (
        resolved
        .get(
            "tme",
            {},
        )
        .get(
            "cell_type_parameter_file"
        )
    )

    if not parameter_file:
        raise SystemExit(
            "Resolved config does not define "
            "tme.cell_type_parameter_file."
        )

    seed = int(
        resolved[
            "simulation"
        ][
            "seed"
        ]
    )

    result = write_nonmalignant_cell_types(
        cells_path=cells_path,
        output_dir=(
            run_dir
            / "cells"
        ),
        preset_path=(
            PROJECT_ROOT
            / parameter_file
        ),
        seed=seed,
    )

    summary = result.summary

    print()
    print("=" * 80)
    print("NON-MALIGNANT CELL-TYPE ASSIGNMENT COMPLETE")
    print("=" * 80)

    print(
        "Malignant:",
        f'{summary["n_malignant_cells"]:,}',
    )

    print(
        "Non-malignant:",
        f'{summary["n_nonmalignant_cells"]:,}',
    )

    print()
    print("Non-malignant cell types:")

    for name, count in (
        summary[
            "cell_type_counts"
        ].items()
    ):
        print(
            f"  {name:<24} "
            f"{count:>6,}"
        )

    print()
    print(
        "Exact counts conserved:",
        summary[
            "exact_counts_conserved"
        ],
    )

    print()


if __name__ == "__main__":
    main()
