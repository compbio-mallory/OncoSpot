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

from cancer_st_sim.expression import (  # noqa: E402
    generate_malignant_counts,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )

    parser.add_argument(
        "--preflight-cells",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    run_dir = args.run_dir.resolve()

    with (
        run_dir
        / "resolved_config.yaml"
    ).open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    expression_config = (
        config[
            "expression"
        ][
            "malignant"
        ]
    )

    result = generate_malignant_counts(
        cells_path=(
            run_dir
            / "cells"
            / "cells_with_cell_types.tsv"
        ),

        states_path=(
            run_dir
            / "expression"
            / "tumor_states"
            / "malignant_cell_states.tsv"
        ),

        clone_gene_cna_path=(
            run_dir
            / "cna"
            / "gene_level"
            / "clone_gene_cna.tsv"
        ),

        parameter_path=(
            PROJECT_ROOT
            / expression_config[
                "parameter_file"
            ]
        ),

        library_size_path=(
            PROJECT_ROOT
            / expression_config[
                "library_size_file"
            ]
        ),

        output_dir=(
            run_dir
            / "expression"
            / "malignant"
        ),

        seed=int(
            config[
                "simulation"
            ][
                "seed"
            ]
        ),

        state_names=config["tumor_states"]["states"],
        chunk_size=args.chunk_size,

        preflight_only=(
            args.preflight_only
        ),

        preflight_cells=(
            args.preflight_cells
        ),
    )

    s = result.summary

    print()
    print("=" * 80)

    if args.preflight_only:
        print(
            "MALIGNANT EXPRESSION PREFLIGHT COMPLETE"
        )
    else:
        print(
            "MALIGNANT EXPRESSION GENERATION COMPLETE"
        )

    print("=" * 80)

    print(
        "Cells:",
        f'{s["n_cells"]:,}',
    )

    print(
        "Genes:",
        f'{s["n_genes"]:,}',
    )

    print(
        "Library median:",
        f'{s["library_size_median"]:.1f}',
    )

    print(
        "Simulated count median:",
        f'{s["simulated_count_median"]:.1f}',
    )

    print(
        "Expected counts/cell:",
        f'{s["mean_expected_counts_per_cell"]:.1f}',
    )

    print(
        "Max mu:",
        f'{s["max_mu"]:.3f}',
    )

    print(
        "CN0 hard-zero passed:",
        s[
            "cn0_rule_passed"
        ],
    )

    print()


if __name__ == "__main__":
    main()
