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
    write_cell_scaffold,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Expand the canonical DEMON scaffold into "
            "individual spatial cells."
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

    scaffold_dir = (
        run_dir
        / "scaffold"
    )

    output_dir = (
        run_dir
        / "cells"
    )

    if not resolved_path.exists():
        raise SystemExit(
            f"Missing resolved config: "
            f"{resolved_path}"
        )

    with resolved_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        resolved = yaml.safe_load(
            handle
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
        .get("tme", {})
        .get("compartment_parameter_file")
    )

    spatial_preset_path = None

    if parameter_file:
        spatial_preset_path = (
            PROJECT_ROOT
            / parameter_file
        )

    result = write_cell_scaffold(
        scaffold_dir=scaffold_dir,
        output_dir=output_dir,
        seed=seed,
        spatial_preset_path=
            spatial_preset_path,
    )

    print()
    print("=" * 80)
    print("CELL SCAFFOLD COMPLETE")
    print("=" * 80)

    print(
        "Cells:",
        f'{result.summary["n_cells"]:,}',
    )

    print(
        "Demes:",
        result.summary["n_demes"],
    )

    print(
        "Extant subclones:",
        result.summary[
            "n_extant_subclones"
        ],
    )

    print(
        "Population conserved:",
        result.summary[
            "population_conserved"
        ],
    )

    print(
        "Deme × clone conserved:",
        result.summary[
            "deme_subclone_composition_conserved"
        ],
    )

    coordinate_valid = result.summary.get(
        "all_cells_within_target_tile",
        result.summary.get(
            "all_cells_within_source_deme",
            False,
        ),
    )

    print(
        "Coordinate model:",
        result.summary.get(
            "coordinate_model",
            "unknown",
        ),
    )

    print(
        "Coordinates valid:",
        coordinate_valid,
    )

    if "n_spatial_tiles" in result.summary:
        print(
            "Filled spatial tiles:",
            result.summary[
                "n_spatial_tiles"
            ],
        )

        print(
            "Previously empty tiles filled:",
            result.summary[
                "n_filled_previous_empty_tiles"
            ],
        )

    print()
    print(
        "Output:",
        output_dir,
    )
    print()


if __name__ == "__main__":
    main()
