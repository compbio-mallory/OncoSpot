#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(
        PROJECT_ROOT
        / "src"
    ),
)

from cancer_st_sim.expression import (  # noqa: E402
    write_empirical_tumor_states,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample empirical HGSOC tumor-state vectors "
            "for malignant simulated cells."
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

    with resolved_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(
            handle
        )

    seed = int(
        config[
            "simulation"
        ][
            "seed"
        ]
    )

    state_config = config[
        "tumor_states"
    ]

    reference_path = (
        PROJECT_ROOT
        / state_config[
            "parameter_file"
        ]
    )

    result = write_empirical_tumor_states(
        cells_path=(
            run_dir
            / "cells"
            / "cells_with_cell_types.tsv"
        ),

        reference_path=(
            reference_path
        ),

        output_dir=(
            run_dir
            / "expression"
            / "tumor_states"
        ),

        seed=seed,

        rng_stage=4001,
        state_columns=state_config["states"],
    )

    summary = result.summary

    print()
    print("=" * 80)
    print("TUMOR-STATE ASSIGNMENT COMPLETE")
    print("=" * 80)

    print(
        "Reference cells:",
        f'{summary["n_reference_cells"]:,}',
    )

    print(
        "Simulated malignant cells:",
        f'{summary["n_simulated_malignant_cells"]:,}',
    )

    print(
        "Unique reference cells sampled:",
        f'{summary["n_unique_reference_cells_used"]:,}',
    )

    print()
    print("State means / SDs:")

    for state in summary[
        "states"
    ]:

        values = summary[
            "simulated_state_summary"
        ][
            state
        ]

        print(
            f"  {state:<18} "
            f'mean={values["mean"]: .4f} '
            f'sd={values["sd"]:.4f}'
        )

    print()

    validation = summary[
        "validation"
    ]

    print(
        "Max |mean difference|:",
        f'{validation["max_abs_mean_difference"]:.4f}',
    )

    print(
        "Max |SD difference|:",
        f'{validation["max_abs_sd_difference"]:.4f}',
    )

    print(
        "Max |correlation difference|:",
        f'{validation["max_abs_correlation_difference"]:.4f}',
    )

    print()


if __name__ == "__main__":
    main()
