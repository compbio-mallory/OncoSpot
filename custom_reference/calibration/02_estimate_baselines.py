#!/usr/bin/env python3
"""Estimate donor-balanced OncoSpot expression baselines from raw UMI counts."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Estimate OncoSpot cell-of-origin and TME parameters."
    )
    parser.add_argument("--counts", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--gene-mapping", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cancer", required=True)
    parser.add_argument("--min-cells", type=int, default=20)
    parser.add_argument("--max-dispersion", type=float, default=10.0)
    args = parser.parse_args()

    if args.min_cells < 1 or args.max_dispersion <= 0:
        parser.error("Invalid min-cells or max-dispersion")

    out = Path(args.output)
    origin_out = out / "malignant/cell_of_origin"
    tme_out = out / "nonmalignant/expression"
    origin_out.mkdir(parents=True, exist_ok=True)
    tme_out.mkdir(parents=True, exist_ok=True)

    ann = pd.read_csv(args.annotations, sep="\t", dtype=str)
    required = {
        "reference_cell_id", "patient",
        "oncospot_cell_type", "is_cell_origin",
    }
    missing = required - set(ann.columns)
    if missing:
        raise ValueError(f"Annotation columns missing: {missing}")

    if not ann["reference_cell_id"].is_unique:
        raise ValueError("Duplicate reference cell IDs")

    for col in ["reference_cell_id", "patient", "oncospot_cell_type"]:
        if ann[col].isna().any() or ann[col].eq("").any():
            raise ValueError(f"Missing values in {col}")

    ann["is_cell_origin"] = (
        ann["is_cell_origin"].str.lower().map({
            "true": True, "false": False
        })
    )
    if ann["is_cell_origin"].isna().any():
        raise ValueError("Invalid is_cell_origin values")

    if (
        ann.loc[ann["is_cell_origin"], "oncospot_cell_type"]
        == "malignant"
    ).any():
        raise ValueError("Cell-of-origin reference includes malignant cells")

    mapping = pd.read_csv(args.gene_mapping, sep="\t", dtype=str)
    mapping = mapping.loc[mapping["status"] == "unique"].copy()
    mapping = mapping.dropna(subset=["gene_symbol", "gene_id"])

    if mapping["gene_symbol"].duplicated().any():
        raise ValueError("Ambiguous gene symbols in mapping")
    if mapping["gene_id"].duplicated().any():
        raise ValueError("Duplicate gene IDs in mapping")

    gene_lookup = mapping.set_index("gene_symbol")["gene_id"].to_dict()

    with gzip.open(args.counts, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")

    if header[0] != "Index":
        raise ValueError("Expected first matrix column named Index")
    if len(header) != len(set(header)):
        raise ValueError("Duplicate matrix cell IDs")

    cell_index = {name: i for i, name in enumerate(header[1:])}

    if not ann["reference_cell_id"].isin(cell_index).all():
        raise ValueError("Annotation contains cells absent from count matrix")

    ann["matrix_col"] = ann["reference_cell_id"].map(cell_index)

    origin = ann.loc[ann["is_cell_origin"]].copy()
    if origin.empty:
        raise ValueError("No cell-of-origin cells")

    origin_groups = list(origin.groupby("patient", sort=True))
    origin_donors = [name for name, _ in origin_groups]
    origin_idx = [
        group["matrix_col"].to_numpy(dtype=np.int64)
        for _, group in origin_groups
    ]
    origin_all_idx = origin["matrix_col"].to_numpy(dtype=np.int64)

    normal = ann.loc[
        ~ann["oncospot_cell_type"].isin(
            ["malignant", "normal_epithelial"]
        ) | (
            (ann["oncospot_cell_type"] == "normal_epithelial")
            & ~ann["is_cell_origin"]
        ) | (
            ann["oncospot_cell_type"] == "normal_epithelial"
        )
    ].copy()

    # All annotated nonmalignant cells, including cell-of-origin cells.
    normal = ann.loc[
        ann["oncospot_cell_type"] != "malignant"
    ].copy()

    group_counts = (
        normal.groupby(["oncospot_cell_type", "patient"])
        .size().rename("n_cells").reset_index()
    )
    eligible = group_counts.loc[
        group_counts["n_cells"] >= args.min_cells,
        ["oncospot_cell_type", "patient"]
    ]

    normal = normal.merge(
        eligible,
        on=["oncospot_cell_type", "patient"],
        how="inner",
        validate="many_to_one",
    )

    if normal.empty:
        raise ValueError("No eligible nonmalignant cells")

    types = sorted(normal["oncospot_cell_type"].unique())

    groups = list(
        normal.groupby(["oncospot_cell_type", "patient"], sort=True)
    )
    group_keys = [key for key, _ in groups]
    group_idx = [
        group["matrix_col"].to_numpy(dtype=np.int64)
        for _, group in groups
    ]

    type_idx = {
        t: normal.loc[
            normal["oncospot_cell_type"] == t, "matrix_col"
        ].to_numpy(dtype=np.int64)
        for t in types
    }

    for t in types:
        if sum(key[0] == t for key in group_keys) < 2:
            raise ValueError(
                f"{t} has fewer than two eligible donors"
            )

    print("Cancer:", args.cancer)
    print("Cell-of-origin cells:", len(origin))
    print("Cell-of-origin donors:", len(origin_donors))
    print("Nonmalignant types:", types)
    print("Retained nonmalignant cells:", len(normal), flush=True)

    # First pass: gene pseudobulks and empirical library sizes.
    rows = []
    origin_library = np.zeros(len(header) - 1, dtype=np.int64)
    tme_library = np.zeros(len(header) - 1, dtype=np.int64)
    seen_symbols = set()

    with gzip.open(args.counts, "rt") as handle:
        next(handle)
        for line_no, line in enumerate(handle, 1):
            symbol, sep, rest = line.partition("\t")
            gene_id = gene_lookup.get(symbol)
            if gene_id is None:
                continue
            if symbol in seen_symbols:
                raise ValueError(f"Repeated mapped gene: {symbol}")
            seen_symbols.add(symbol)

            y = np.fromstring(rest, sep="\t", dtype=np.int64)
            if len(y) != len(header) - 1:
                raise ValueError(f"Count width mismatch: {symbol}")
            if (y < 0).any():
                raise ValueError(f"Negative counts: {symbol}")

            o = [int(y[idx].sum()) for idx in origin_idx]
            g = [int(y[idx].sum()) for idx in group_idx]

            origin_library[origin_all_idx] += y[origin_all_idx]

            for idx in group_idx:
                tme_library[idx] += y[idx]

            detected_cells = int(np.count_nonzero(y[origin_all_idx]))
            detected_donors = sum(v > 0 for v in o)

            rows.append((
                gene_id, symbol, o, g,
                detected_cells, detected_donors
            ))

            if line_no % 5000 == 0:
                print(f"Pass 1: {line_no:,} rows", flush=True)

    if not rows:
        raise ValueError("No genes matched the supplied gene mapping")

    gene_ids = [r[0] for r in rows]
    gene_symbols = [r[1] for r in rows]

    origin_counts = np.asarray(
        [r[2] for r in rows], dtype=np.float64
    )
    tme_counts = np.asarray(
        [r[3] for r in rows], dtype=np.float64
    )

    # Cell-of-origin donor-balanced pseudobulk.
    origin_totals = origin_counts.sum(axis=0)
    if (origin_totals <= 0).any():
        raise ValueError("An origin donor has no mapped UMIs")

    origin_fraction = (
        origin_counts / origin_totals[None, :]
    ).mean(axis=1)

    pooled = origin_counts.sum(axis=1)
    pooled_fraction = pooled / pooled.sum()

    origin_table = pd.DataFrame({
        "gene_id": gene_ids,
        "gene_symbol": gene_symbols,
        "origin_label": "user_defined_origin",
        "baseline_fraction_donor_balanced": origin_fraction,
        "alpha_origin": np.where(
            origin_fraction > 0,
            np.log(np.maximum(origin_fraction, 1e-300)),
            np.nan,
        ),
        "baseline_fraction_pooled_cells": pooled_fraction,
        "cell_detection_fraction": [
            r[4] / len(origin) for r in rows
        ],
        "donor_detection_fraction": [
            r[5] / len(origin_donors) for r in rows
        ],
        "n_origin_cells": len(origin),
        "n_origin_patients": len(origin_donors),
        "alpha_source": "donor_balanced_pseudobulk",
    })

    origin_table.to_csv(
        origin_out / "origin_baseline.tsv",
        sep="\t", index=False
    )

    # Nonmalignant donor-balanced expression baselines.
    baseline_vectors = {}
    baseline_rows = []

    for t in types:
        cols = [
            j for j, key in enumerate(group_keys)
            if key[0] == t
        ]
        sub = tme_counts[:, cols]
        totals = sub.sum(axis=0)
        if (totals <= 0).any():
            raise ValueError(f"Zero mapped UMIs for {t}")

        p = (sub / totals[None, :]).mean(axis=1)
        p /= p.sum()
        baseline_vectors[t] = p

        for j, (gid, symbol) in enumerate(
            zip(gene_ids, gene_symbols)
        ):
            value = float(p[j])
            baseline_rows.append({
                "broad_cell_type": t,
                "gene_id": gid,
                "gene_symbol": symbol,
                "baseline_fraction": value,
                "n_donors": len(cols),
                "baseline_positive": bool(value > 0),
                "alpha": np.log(value) if value > 0 else np.nan,
            })

    pd.DataFrame(baseline_rows).to_csv(
        tme_out / "tme_expression_baselines.tsv",
        sep="\t", index=False
    )

    libraries = normal[[
        "reference_cell_id", "patient",
        "oncospot_cell_type", "matrix_col"
    ]].copy()

    libraries = libraries.rename(columns={
        "patient": "sample",
        "oncospot_cell_type": "broad_cell_type"
    })

    libraries["library_size"] = tme_library[
        libraries["matrix_col"].to_numpy(dtype=np.int64)
    ]

    if (libraries["library_size"] <= 0).any():
        raise ValueError("Non-positive library sizes")

    libraries.drop(
        columns="matrix_col"
    ).to_csv(
        tme_out / "library_sizes_by_cell_type.tsv",
        sep="\t", index=False
    )

    # Second pass: pooled NB2 method-of-moments dispersion.
    numerator = defaultdict(float)
    denominator = defaultdict(float)
    genes_used = defaultdict(int)

    gene_index = {
        symbol: j for j, symbol in enumerate(gene_symbols)
    }

    print("Starting dispersion estimation", flush=True)

    with gzip.open(args.counts, "rt") as handle:
        next(handle)

        for line_no, line in enumerate(handle, 1):
            symbol, sep, rest = line.partition("\t")
            j = gene_index.get(symbol)
            if j is None:
                continue

            y = np.fromstring(rest, sep="\t", dtype=np.int64)

            for t in types:
                idx = type_idx[t]
                L = tme_library[idx].astype(float)
                keep = L > 0
                if not keep.any():
                    continue

                observed = y[idx][keep].astype(float)
                mu = L[keep] * baseline_vectors[t][j]

                mean_mu = float(mu.mean())
                if not 0.05 <= mean_mu <= 50.0:
                    continue

                numerator[t] += float(
                    np.sum((observed - mu) ** 2 - mu)
                )
                denominator[t] += float(np.sum(mu ** 2))
                genes_used[t] += 1

            if line_no % 5000 == 0:
                print(f"Pass 2: {line_no:,} rows", flush=True)

    dispersion_parameters = {}

    for t in types:
        if denominator[t] <= 0:
            raise ValueError(f"Dispersion unavailable for {t}")

        raw_phi = numerator[t] / denominator[t]
        phi = float(
            np.clip(raw_phi, 0, args.max_dispersion)
        )

        dispersion_parameters[t] = {
            "n_cells": int(len(type_idx[t])),
            "genes_used": int(genes_used[t]),
            "raw_method_of_moments_phi": float(raw_phi),
            "dispersion": phi,
            "clipped": bool(
                raw_phi < 0 or raw_phi > args.max_dispersion
            ),
        }

    dispersion = {
        "dataset": Path(args.counts).name,
        "cancer_type": args.cancer,
        "model": "NB2",
        "variance_definition": "Var(Y)=mu+phi*mu^2",
        "dispersion_level": "one_pooled_dispersion_per_broad_cell_type",
        "estimation": "pooled_residual_method_of_moments",
        "status": "provisional_requires_validation",
        "parameters": dispersion_parameters,
    }

    with (tme_out / "tme_dispersion.yaml").open("w") as handle:
        yaml.safe_dump(dispersion, handle, sort_keys=False)

    summary = {
        "cancer": args.cancer,
        "n_genes": len(rows),
        "origin_cells": len(origin),
        "origin_donors": len(origin_donors),
        "nonmalignant_types": types,
        "dispersion": {
            t: dispersion_parameters[t]["dispersion"]
            for t in types
        },
    }

    (out / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("\nCALIBRATION COMPLETE")
    print(json.dumps(summary, indent=2))
    print("Output:", out.resolve())


if __name__ == "__main__":
    main()
