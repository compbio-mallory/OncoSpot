#!/usr/bin/env python3
"""Register a calibrated custom cancer in an OncoSpot checkout."""

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

MODES = {
    "invasive_glandular": "HGSOC",
    "gland_fission": "CRC",
    "boundary_growth": "HCC",
}

FILES = {
    "malignant/cell_of_origin/origin_baseline.tsv":
        "malignant/cell_of_origin/origin_baseline.tsv",
    "malignant/expression/malignant_expression_parameters.tsv":
        "malignant/expression/malignant_expression_parameters.tsv",
    "malignant/expression/malignant_library_sizes.tsv":
        "malignant/expression/malignant_library_sizes.tsv",
    "malignant/expression/expression_metadata.yaml":
        "malignant/expression/expression_metadata.yaml",
    "nonmalignant/expression/tme_expression_baselines.tsv":
        "nonmalignant/expression/tme_expression_baselines.tsv",
    "nonmalignant/expression/library_sizes_by_cell_type.tsv":
        "nonmalignant/expression/library_sizes_by_cell_type.tsv",
    "nonmalignant/expression/tme_dispersion.yaml":
        "nonmalignant/expression/tme_dispersion.yaml",
    "tumor_states/state_reference.tsv":
        "tumor_states/state_reference.tsv",
    "tumor_states/state_metadata.yaml":
        "tumor_states/state_metadata.yaml",
}

STATES = [
    "PROLIFERATION", "WNT", "MYC",
    "HYPOXIA", "EMT", "METABOLIC",
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cancer", required=True)
    p.add_argument("--parameters", required=True)
    p.add_argument("--demon-mode", choices=MODES, required=True)
    p.add_argument(
        "--base-preset",
        choices=["HGSOC", "CRC", "HCC"],
        default=None,
        help="Spatial template; must match the selected DEMON mode",
    )
    args = p.parse_args()

    cancer = args.cancer.upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,31}", cancer):
        p.error("Cancer ID must be 2-32 letters, digits or underscores")

    if cancer in ["HGSOC", "CRC", "HCC"]:
        p.error("Cannot replace an existing validated cancer")

    base = args.base_preset or MODES[args.demon_mode]
    if base != MODES[args.demon_mode]:
        p.error(
            f"{args.demon_mode} currently requires the "
            f"{MODES[args.demon_mode]} spatial preset"
        )

    source = Path(args.parameters).resolve()
    if not source.is_dir():
        p.error(f"Parameter bundle not found: {source}")

    for relative in FILES:
        if not (source / relative).is_file():
            p.error(f"Missing calibrated file: {relative}")

    malignant = pd.read_csv(
        source / "malignant/expression/malignant_expression_parameters.tsv",
        sep="\t", nrows=2,
    )
    necessary = {
        "gene_id", "gene_symbol", "alpha", "beta", "dispersion",
        "state_effect_lower", "state_effect_upper",
        *[f"gamma_{x}" for x in STATES],
    }
    if not necessary.issubset(malignant.columns):
        p.error(
            "Malignant parameter schema missing: "
            + str(sorted(necessary - set(malignant.columns)))
        )

    cancer_yaml = ROOT / "configs/cancers" / f"{cancer}.yaml"
    example_yaml = ROOT / "configs/examples" / (
        f"{cancer.lower()}_custom.yaml"
    )
    destination = ROOT / "parameters" / cancer

    if any(x.exists() for x in [cancer_yaml, example_yaml, destination]):
        p.error("Registration already exists; refusing to overwrite it")

    template = ROOT / "configs/cancers" / f"{base}.yaml"
    if not template.is_file():
        p.error(f"Missing base preset: {template}")

    cfg = yaml.safe_load(template.read_text())
    cfg["cancer_type"] = cancer

    metadata = cfg.setdefault("metadata", {})
    metadata["full_name"] = f"{cancer} custom cancer"
    metadata["status"] = "custom_provisional_unvalidated"
    metadata["description"] = (
        "Reference-calibrated expression with provisional "
        f"{base} spatial and CNA configuration."
    )

    cfg["demon"]["mode"] = args.demon_mode
    cfg["demon"]["preset"] = f"{cancer}_custom_{args.demon_mode}"

    prefix = f"parameters/{cancer}"

    expr = cfg["expression"]
    expr["malignant"]["parameter_file"] = (
        f"{prefix}/malignant/expression/"
        "malignant_expression_parameters.tsv"
    )
    expr["malignant"]["metadata_file"] = (
        f"{prefix}/malignant/expression/expression_metadata.yaml"
    )
    expr["malignant"]["library_size_file"] = (
        f"{prefix}/malignant/expression/malignant_library_sizes.tsv"
    )

    expr["normal"]["baseline_file"] = (
        f"{prefix}/nonmalignant/expression/tme_expression_baselines.tsv"
    )
    expr["normal"]["library_size_file"] = (
        f"{prefix}/nonmalignant/expression/library_sizes_by_cell_type.tsv"
    )
    expr["normal"]["dispersion_file"] = (
        f"{prefix}/nonmalignant/expression/tme_dispersion.yaml"
    )
    expr["cell_of_origin"]["label"] = f"{cancer}_origin"
    expr["cell_of_origin"]["parameter_file"] = (
        f"{prefix}/malignant/cell_of_origin/origin_baseline.tsv"
    )

    # The original pipeline may read this field for provenance.
    expr["cell_of_origin"].pop("metadata_file", None)

    cfg.setdefault("malignant", {}).setdefault(
        "cell_of_origin", {}
    )["name"] = f"{cancer}_origin"
    cfg["malignant"]["cell_of_origin"][
        "source_parameter_bundle"
    ] = cancer

    cfg["tumor_states"]["parameter_file"] = (
        f"{prefix}/tumor_states/state_reference.tsv"
    )
    cfg["tumor_states"]["metadata_file"] = (
        f"{prefix}/tumor_states/state_metadata.yaml"
    )
    cfg["tumor_states"]["states"] = list(STATES)
    cfg["malignant"]["tumor_states"]["states"] = list(STATES)

    # Reuse the validated base cancer's DEMON, CNA and
    # compartment presets. These remain explicitly provisional
    # for the newly registered cancer.
    destination.mkdir(parents=True)
    try:
        for old, new in FILES.items():
            target = destination / new
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / old, target)

        cancer_yaml.write_text(
            yaml.safe_dump(cfg, sort_keys=False)
        )

        base_example = {
            "HGSOC": "hgsoc_validation_15k_hallmark_v1.yaml",
            "CRC": "crc_validation_15k.yaml",
            "HCC": "hcc_validation_15k_v1.yaml",
        }[base]
        example_src = ROOT / "configs/examples" / base_example
        if not example_src.is_file():
            raise FileNotFoundError(example_src)

        user = yaml.safe_load(example_src.read_text())
        user["cancer_type"] = cancer

        # Do not inherit any explicit reference-expression overrides
        # from the base example.
        for key in ["expression", "tumor_states", "malignant", "tme"]:
            user.pop(key, None)

        example_yaml.write_text(
            yaml.safe_dump(user, sort_keys=False)
        )

    except Exception:
        cancer_yaml.unlink(missing_ok=True)
        example_yaml.unlink(missing_ok=True)
        shutil.rmtree(destination)
        raise

    print("REGISTERED:", cancer)
    print("Cancer config:", cancer_yaml)
    print("Example config:", example_yaml)
    print("Parameter directory:", destination)
    print("DEMON mode:", args.demon_mode)
    print("Spatial/CNA template:", base)
    print("Status: PROVISIONAL — validate with a smoke simulation")


if __name__ == "__main__":
    main()
