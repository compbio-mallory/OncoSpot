#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DEMON_CANDIDATES = ([Path(os.environ["ONCOSPOT_DEMON_BIN"])] if os.environ.get("ONCOSPOT_DEMON_BIN") else []) + [
    PROJECT_ROOT / "external/demon_linux/bin/demon",
    PROJECT_ROOT / "external/demon_reference/bin/demon",
]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected YAML mapping in {path}"
        )

    return data


def find_demon_binary() -> Path:
    for candidate in DEFAULT_DEMON_CANDIDATES:
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "No active DEMON executable found. Checked:\n"
        + "\n".join(
            f"  - {p}"
            for p in DEFAULT_DEMON_CANDIDATES
        )
    )


def verify_linux_binary(binary: Path) -> str:
    result = subprocess.run(
        ["file", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )

    info = result.stdout.strip()

    if "ELF" not in info:
        raise RuntimeError(
            "Selected DEMON executable is not an ELF Linux binary:\n"
            f"{info}"
        )

    return info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DEMON for one Cancer ST Simulator run."
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        help=(
            "Run directory containing resolved_config.yaml "
            "and demon/config.dat."
        ),
    )

    parser.add_argument(
        "--demon-binary",
        default=None,
        help=(
            "Optional explicit DEMON executable. "
            "Otherwise an active project executable is selected."
        ),
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()

    resolved_path = (
        run_dir / "resolved_config.yaml"
    )

    demon_dir = (
        run_dir / "demon"
    )

    config_path = (
        demon_dir / "config.dat"
    )

    raw_dir = (
        demon_dir / "raw"
    )

    if not resolved_path.exists():
        raise SystemExit(
            f"Missing resolved config: {resolved_path}"
        )

    if not config_path.exists():
        raise SystemExit(
            f"Missing DEMON config: {config_path}"
        )

    resolved = load_yaml(
        resolved_path
    )

    if args.demon_binary:
        demon_binary = Path(
            args.demon_binary
        ).resolve()
    else:
        demon_binary = find_demon_binary()

    if not demon_binary.exists():
        raise SystemExit(
            f"DEMON executable does not exist: {demon_binary}"
        )

    binary_info = verify_linux_binary(
        demon_binary
    )

    # Fresh raw DEMON output for this run.
    if raw_dir.exists():
        shutil.rmtree(raw_dir)

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        config_path,
        raw_dir / "config.dat",
    )

    runtime_binary = (
        raw_dir / "demon"
    )

    shutil.copy2(
        demon_binary,
        runtime_binary,
    )

    runtime_binary.chmod(
        runtime_binary.stat().st_mode | 0o111
    )

    stdout_path = (
        demon_dir / "stdout.log"
    )

    stderr_path = (
        demon_dir / "stderr.log"
    )

    manifest_path = (
        demon_dir / "run_manifest.json"
    )

    start = datetime.now().astimezone()

    manifest = {
        "stage": "demon",
        "status": "running",
        "cancer_type": resolved["cancer_type"],
        "architecture": (
            resolved
            .get("demon", {})
            .get("mode")
        ),
        "preset": (
            resolved
            .get("demon", {})
            .get("preset")
        ),
        "n_cells_requested": (
            resolved["simulation"]["n_cells"]
        ),
        "tumor_fraction_requested": (
            resolved["tissue"]["tumor_fraction"]
        ),
        "seed": (
            resolved["simulation"]["seed"]
        ),
        "source_binary": str(
            demon_binary
        ),
        "binary_info": binary_info,
        "config": str(config_path),
        "raw_directory": str(raw_dir),
        "start_time": start.isoformat(),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("DEMON RUN")
    print("=" * 80)

    print(
        "Cancer:",
        resolved["cancer_type"],
    )

    print(
        "Architecture:",
        manifest["architecture"],
    )

    print(
        "Requested cells:",
        f'{manifest["n_cells_requested"]:,}',
    )

    print(
        "Seed:",
        manifest["seed"],
    )

    print(
        "Executable:",
        demon_binary,
    )

    print(
        "Binary:",
        binary_info,
    )

    print(
        "Working directory:",
        raw_dir,
    )

    print()

    command = [
        "./demon",
        "./",
        "config.dat",
    ]

    print(
        "Command:",
        " ".join(command),
    )

    print()

    try:
        with (
            stdout_path.open(
                "w",
                encoding="utf-8",
            ) as stdout_handle,
            stderr_path.open(
                "w",
                encoding="utf-8",
            ) as stderr_handle,
        ):
            completed = subprocess.run(
                command,
                cwd=raw_dir,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )

        if completed.returncode != 0:
            manifest["status"] = "failed"
            manifest["return_code"] = (
                completed.returncode
            )

            manifest["end_time"] = (
                datetime
                .now()
                .astimezone()
                .isoformat()
            )

            manifest_path.write_text(
                json.dumps(
                    manifest,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            raise SystemExit(
                "DEMON failed with return code "
                f"{completed.returncode}.\n"
                f"See:\n"
                f"  {stdout_path}\n"
                f"  {stderr_path}"
            )

        required_outputs = [
            "clones.dat",
            "demes.dat",
            "output_driver_genotype_properties.dat",
        ]

        missing = []

        for name in required_outputs:
            path = raw_dir / name

            if (
                not path.exists()
                or path.stat().st_size == 0
            ):
                missing.append(name)

        if missing:
            manifest["status"] = (
                "failed_missing_outputs"
            )

            manifest["missing_outputs"] = (
                missing
            )

            manifest["end_time"] = (
                datetime
                .now()
                .astimezone()
                .isoformat()
            )

            manifest_path.write_text(
                json.dumps(
                    manifest,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            raise SystemExit(
                "DEMON finished but required outputs "
                "are missing:\n  "
                + "\n  ".join(missing)
            )

        produced_files = []

        for path in sorted(
            raw_dir.iterdir()
        ):
            if path.is_file():
                produced_files.append(
                    {
                        "name": path.name,
                        "size_bytes": path.stat().st_size,
                    }
                )

        manifest["status"] = "complete"
        manifest["return_code"] = 0
        manifest["end_time"] = (
            datetime
            .now()
            .astimezone()
            .isoformat()
        )
        manifest["outputs"] = (
            produced_files
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    except Exception:
        if (
            manifest.get("status")
            == "running"
        ):
            manifest["status"] = (
                "failed_exception"
            )

            manifest["end_time"] = (
                datetime
                .now()
                .astimezone()
                .isoformat()
            )

            manifest_path.write_text(
                json.dumps(
                    manifest,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        raise

    print("=" * 80)
    print("DEMON COMPLETE")
    print("=" * 80)

    print()

    for item in produced_files:
        print(
            f'{item["name"]:<45}'
            f'{item["size_bytes"]:>12,} bytes'
        )

    print()
    print(
        "Manifest:",
        manifest_path,
    )

    print(
        "stdout:",
        stdout_path,
    )

    print(
        "stderr:",
        stderr_path,
    )

    print()


if __name__ == "__main__":
    main()
