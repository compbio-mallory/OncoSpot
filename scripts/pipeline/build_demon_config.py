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
    build_demon_config,
    load_yaml,
)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build a DEMON config.dat from a resolved "
            "cancer-ST simulator configuration."
        )
    )

    parser.add_argument(
        "--resolved-config",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    resolved_path = Path(
        args.resolved_config
    ).resolve()

    output_path = Path(
        args.output
    ).resolve()

    resolved = load_yaml(
        resolved_path
    )

    demon = resolved.get(
        "demon",
        {},
    )

    parameter_file = demon.get(
        "parameter_file"
    )

    if not parameter_file:
        raise SystemExit(
            "Resolved configuration does not define "
            "demon.parameter_file."
        )

    preset_path = (
        PROJECT_ROOT
        / parameter_file
    )

    preset = load_yaml(
        preset_path
    )

    text = build_demon_config(
        resolved_config=resolved,
        demon_preset=preset,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        text,
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("DEMON CONFIGURATION BUILT")
    print("=" * 80)

    print(
        "Cancer:",
        resolved["cancer_type"],
    )

    print(
        "Architecture:",
        demon.get("mode"),
    )

    print(
        "Preset:",
        demon.get("preset"),
    )

    print(
        "Total DEMON cells:",
        resolved["simulation"]["n_cells"],
    )

    print(
        "Seed:",
        resolved["simulation"]["seed"],
    )

    print(
        "Output:",
        output_path,
    )

    print()


if __name__ == "__main__":
    main()
