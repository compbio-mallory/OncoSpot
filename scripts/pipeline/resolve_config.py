#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC = (
    PROJECT_ROOT
    / "src"
)

sys.path.insert(
    0,
    str(SRC),
)

from cancer_st_sim.config import (  # noqa: E402
    ConfigError,
    resolve_config,
)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Resolve a user cancer-ST configuration "
            "against global and cancer-specific presets."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to user YAML configuration.",
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory in which to save resolved config.",
    )

    args = parser.parse_args()

    config_path = Path(
        args.config
    ).resolve()

    run_dir = Path(
        args.run_dir
    ).resolve()

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        resolved, warnings = resolve_config(
            project_root=PROJECT_ROOT,
            user_config_path=config_path,
        )

    except ConfigError as exc:

        print()
        print("CONFIGURATION ERROR")
        print("-------------------")
        print(str(exc))
        print()

        raise SystemExit(1)

    shutil.copy2(
        config_path,
        run_dir
        / "input_config.yaml",
    )

    resolved_path = (
        run_dir
        / "resolved_config.yaml"
    )

    with resolved_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        yaml.safe_dump(
            resolved,
            handle,
            sort_keys=False,
            default_flow_style=False,
        )

    print()
    print("=" * 80)
    print("CANCER ST SIMULATOR CONFIGURATION")
    print("=" * 80)

    print(
        "Cancer:",
        resolved["cancer_type"],
    )

    print(
        "Cells:",
        f'{resolved["simulation"]["n_cells"]:,}',
    )

    print(
        "Malignant:",
        f'{resolved["tissue"]["n_malignant"]:,}',
    )

    print(
        "Non-malignant:",
        f'{resolved["tissue"]["n_nonmalignant"]:,}',
    )

    print(
        "Tumor fraction:",
        resolved[
            "tissue"
        ][
            "tumor_fraction"
        ],
    )

    print(
        "Technology:",
        resolved[
            "simulation"
        ][
            "technology"
        ],
    )

    print(
        "Seed:",
        resolved[
            "simulation"
        ][
            "seed"
        ],
    )

    print(
        "DEMON mode:",
        resolved
        .get(
            "demon",
            {},
        )
        .get(
            "mode"
        ),
    )

    print()

    if warnings:

        print("WARNINGS")

        for warning in warnings:
            print(
                f"  - {warning}"
            )

        print()

    print(
        "Resolved configuration:"
    )

    print(
        resolved_path
    )

    print()


if __name__ == "__main__":
    main()
