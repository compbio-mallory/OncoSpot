#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CNASIM = Path(os.environ.get("ONCOSPOT_CNASIM") or shutil.which("cnasim") or (PROJECT_ROOT / "envs/st_sim/bin/cnasim"))


def load_yaml(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected YAML mapping: {path}"
        )

    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run CNAsim on the malignant DEMON-derived lineage."
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

    bridge_dir = (
        run_dir
        / "cna"
        / "bridge"
    )

    tree_path = (
        bridge_dir
        / "malignant_tree.nwk"
    )

    mapping_path = (
        bridge_dir
        / "node_mapping.tsv"
    )

    out_dir = (
        run_dir
        / "cna"
        / "cnasim_raw"
    )

    log_path = (
        run_dir
        / "cna"
        / "cnasim.log"
    )

    manifest_path = (
        run_dir
        / "cna"
        / "cnasim_manifest.json"
    )

    if not resolved_path.exists():
        raise SystemExit(
            f"Missing resolved config: {resolved_path}"
        )

    if not tree_path.exists():
        raise SystemExit(
            f"Missing malignant tree: {tree_path}"
        )

    if not mapping_path.exists():
        raise SystemExit(
            f"Missing node mapping: {mapping_path}"
        )

    if (
        not CNASIM.exists()
        or not CNASIM.is_file()
    ):
        raise SystemExit(
            f"CNAsim executable not found: {CNASIM}"
        )

    resolved = load_yaml(
        resolved_path
    )

    parameter_file = (
        resolved
        .get(
            "cnasim",
            {},
        )
        .get(
            "parameter_file"
        )
    )

    if not parameter_file:
        raise SystemExit(
            "Resolved configuration does not define "
            "cnasim.parameter_file."
        )

    preset_path = (
        PROJECT_ROOT
        / parameter_file
    )

    preset = load_yaml(
        preset_path
    )

    # User-facing CNA burden.
    burden = (
        resolved
        .get(
            "customize",
            {},
        )
        .get(
            "cna_burden",
            "default",
        )
    )

    burden_levels = preset[
        "burden_levels"
    ]

    if burden not in burden_levels:
        raise SystemExit(
            f"Unsupported CNA burden {burden!r}. "
            f"Allowed: {sorted(burden_levels)}"
        )

    multiplier = float(
        burden_levels[
            burden
        ][
            "event_rate_multiplier"
        ]
    )

    focal = preset[
        "focal_events"
    ]

    broad = preset[
        "broad_events"
    ]

    simulation = preset[
        "simulation"
    ]

    observation = preset[
        "observation_noise"
    ]

    focal_rate = (
        float(
            focal[
                "placement_param"
            ]
        )
        * multiplier
    )

    founder_rate = (
        float(
            broad[
                "founder_rate"
            ]
        )
        * multiplier
    )

    post_founder_rate = (
        float(
            broad[
                "immediate_post_founder_rate"
            ]
        )
        * multiplier
    )

    internal_rate = (
        float(
            broad[
                "internal_branch_rate"
            ]
        )
        * multiplier
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        str(CNASIM),

        "-m",
        str(
            simulation[
                "mode"
            ]
        ),

        "-o",
        str(out_dir),

        "-t",
        str(
            preset[
                "tree"
            ][
                "tree_type"
            ]
        ),

        "-T",
        str(tree_path),

        "-n",
        str(
            simulation[
                "num_cells_placeholder"
            ]
        ),

        "-n1",
        str(
            simulation[
                "normal_fraction"
            ]
        ),

        "-n2",
        str(
            simulation[
                "pseudonormal_fraction"
            ]
        ),

        "-c",
        str(
            simulation[
                "num_additional_clones"
            ]
        ),

        "-p",
        str(
            focal[
                "placement_type"
            ]
        ),

        "-p1",
        str(
            focal_rate
        ),

        "-j",
        str(
            int(
                focal[
                    "founder_event_multiplier"
                ]
            )
        ),

        "-k",
        str(
            focal[
                "region_length_bp"
            ]
        ),

        "-B",
        str(
            focal[
                "output_bin_length_bp"
            ]
        ),

        "-l",
        str(
            focal[
                "mean_event_length_bp"
            ]
        ),

        "-l1",
        str(
            focal[
                "minimum_event_length_bp"
            ]
        ),

        "-a",
        str(
            focal[
                "copy_parameter"
            ]
        ),

        "-b",
        str(
            focal[
                "amplification_probability"
            ]
        ),
    ]

    if bool(
        broad[
            "enabled"
        ]
    ):
        command.extend(
            [
                "-v",

                "-q",
                str(
                    broad[
                        "chromosome_arm_probability"
                    ]
                ),

                "-i",
                str(
                    founder_rate
                ),

                "-i1",
                str(
                    post_founder_rate
                ),

                "-i2",
                str(
                    internal_rate
                ),

                "-u",
                str(
                    broad[
                        "duplication_probability"
                    ]
                ),
            ]
        )

    genome_preset = (
        preset[
            "genome"
        ][
            "preset"
        ]
    )

    if genome_preset == "hg38_autosomes":
        command.append(
            "-U"
        )
    else:
        raise SystemExit(
            "Only genome.preset=hg38_autosomes "
            "is implemented in v1."
        )

    command.extend(
        [
            "-E1",
            str(
                observation[
                    "error_rate_1"
                ]
            ),

            "-E2",
            str(
                observation[
                    "error_rate_2"
                ]
            ),
        ]
    )

    start = (
        datetime
        .now()
        .astimezone()
    )

    manifest = {
        "stage":
            "cnasim",

        "status":
            "running",

        "cancer_type":
            resolved[
                "cancer_type"
            ],

        "preset":
            preset[
                "preset_name"
            ],

        "preset_version":
            preset[
                "preset_version"
            ],

        "burden_level":
            burden,

        "event_rate_multiplier":
            multiplier,

        "effective_rates": {
            "focal_events_per_edge":
                focal_rate,

            "broad_founder_rate":
                founder_rate,

            "broad_post_founder_rate":
                post_founder_rate,

            "broad_internal_rate":
                internal_rate,
        },

        "tree":
            str(
                tree_path
            ),

        "node_mapping":
            str(
                mapping_path
            ),

        "output_directory":
            str(
                out_dir
            ),

        "command":
            command,

        "start_time":
            start.isoformat(),
    }

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    print("CNASIM RUN")
    print("=" * 80)

    print(
        "Cancer:",
        resolved[
            "cancer_type"
        ],
    )

    print(
        "Preset:",
        preset[
            "preset_name"
        ],
    )

    print(
        "CNA burden:",
        burden,
    )

    print(
        "Rate multiplier:",
        multiplier,
    )

    print(
        "Tree:",
        tree_path,
    )

    print(
        "Output:",
        out_dir,
    )

    print()
    print("Command:")

    print(
        " ".join(
            command
        )
    )

    print()

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_handle:
        completed = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )

    end = (
        datetime
        .now()
        .astimezone()
    )

    manifest[
        "return_code"
    ] = (
        completed.returncode
    )

    manifest[
        "end_time"
    ] = (
        end.isoformat()
    )

    if completed.returncode == 0:
        manifest[
            "status"
        ] = "complete"
    else:
        manifest[
            "status"
        ] = "failed"

    produced = []

    if out_dir.exists():
        for path in sorted(
            out_dir.iterdir()
        ):
            if path.is_file():
                produced.append(
                    {
                        "name":
                            path.name,

                        "size_bytes":
                            path.stat().st_size,
                    }
                )

    manifest[
        "outputs"
    ] = produced

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if (
        completed.returncode
        != 0
    ):
        raise SystemExit(
            "CNAsim failed with return code "
            f"{completed.returncode}. "
            f"See {log_path}"
        )

    print("=" * 80)
    print("CNASIM COMPLETE")
    print("=" * 80)

    print()

    for item in produced:
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
        "Log:",
        log_path,
    )

    print()


if __name__ == "__main__":
    main()
