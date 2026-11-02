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

from cancer_st_sim.tumor import (  # noqa: E402
    write_demon_scaffold,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw DEMON output into the canonical "
            "Cancer ST Simulator spatial/evolutionary scaffold."
        )
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        help=(
            "Simulator run directory containing "
            "resolved_config.yaml and demon/raw/."
        ),
    )

    args = parser.parse_args()

    run_dir = Path(
        args.run_dir
    ).resolve()

    resolved_path = (
        run_dir
        / "resolved_config.yaml"
    )

    raw_dir = (
        run_dir
        / "demon"
        / "raw"
    )

    output_dir = (
        run_dir
        / "scaffold"
    )

    if not resolved_path.exists():
        raise SystemExit(
            f"Missing resolved config: {resolved_path}"
        )

    with resolved_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        resolved = yaml.safe_load(
            handle
        )

    requested_seed = int(
        resolved[
            "simulation"
        ][
            "seed"
        ]
    )

    result = write_demon_scaffold(
        raw_dir=raw_dir,
        output_dir=output_dir,
        requested_seed=requested_seed,
    )

    print()
    print("=" * 80)
    print("DEMON SCAFFOLD COMPLETE")
    print("=" * 80)

    print(
        "Tumor scaffold cells:",
        f'{result.summary["tumor_cells"]:,}',
    )

    print(
        "Occupied tumor demes:",
        result.summary[
            "occupied_tumor_demes"
        ],
    )

    print(
        "Extant driver subclones:",
        result.summary[
            "extant_driver_subclones"
        ],
    )

    print(
        "Driver nodes in tree:",
        result.summary[
            "driver_nodes_in_tree"
        ],
    )

    print(
        "Requested seed:",
        result.summary[
            "requested_seed"
        ],
    )

    print(
        "Effective DEMON seed:",
        result.summary[
            "effective_demon_seed"
        ],
    )

    print(
        "DEMON trials:",
        result.summary[
            "demon_trials"
        ],
    )

    print()
    print(
        "Population conservation:"
    )

    print(
        "  global:",
        result.summary[
            "global_population_conserved"
        ],
    )

    print(
        "  subclone:",
        result.summary[
            "subclone_population_conserved"
        ],
    )

    print(
        "  deme:",
        result.summary[
            "deme_population_conserved"
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
