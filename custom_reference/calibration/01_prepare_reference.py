#!/usr/bin/env python3
"""Validate and standardize reference data for OncoSpot calibration."""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

STATES = [
    "PROLIFERATION", "WNT", "MYC",
    "HYPOXIA", "EMT", "METABOLIC",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cell-type-column", required=True)
    p.add_argument("--patient-column", required=True)
    p.add_argument("--malignant-label", required=True)
    p.add_argument("--origin-label", required=True)
    p.add_argument("--gene-id-column", default="gene_id")
    p.add_argument("--gene-symbol-column", default="gene_symbol")
    p.add_argument("--counts-layer", default=None)
    args = p.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    a = ad.read_h5ad(args.reference)

    for name in [args.cell_type_column, args.patient_column]:
        if name not in a.obs.columns:
            raise ValueError(f"Missing obs column: {name}")

    def get_gene_column(name):
        if name in a.var.columns:
            return a.var[name].astype(str).to_numpy()
        if a.var.index.name == name:
            return a.var_names.astype(str).to_numpy()
        raise ValueError(
            f"Missing gene column '{name}'. "
            "Provide it in adata.var or as the var index."
        )

    gene_id = get_gene_column(args.gene_id_column)
    gene_symbol = get_gene_column(args.gene_symbol_column)

    if len(set(gene_id)) != len(gene_id):
        raise ValueError("Duplicate gene IDs")
    x = a.layers[args.counts_layer] if args.counts_layer else a.X

    if sparse.issparse(x):
        x = x.tocsr()
        values = x.data
    else:
        x = np.asarray(x)
        values = x.ravel()

    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite counts")

    if np.any(values < 0):
        raise ValueError("Negative counts")

    if not np.allclose(values, np.round(values)):
        raise ValueError(
            "Expression matrix must contain raw integer counts"
        )

    labels = a.obs[args.cell_type_column].astype(str)
    patients = a.obs[args.patient_column].astype(str)

    if labels.isin(["nan", "None", ""]).any():
        raise ValueError("Missing cell-type labels")

    if patients.isin(["nan", "None", ""]).any():
        raise ValueError("Missing patient identifiers")

    malignant = labels == args.malignant_label
    origin = labels == args.origin_label
    nonmalignant = ~malignant

    if malignant.sum() == 0:
        raise ValueError("No malignant reference cells")

    if origin.sum() == 0:
        raise ValueError("No cell-of-origin reference cells")

    if origin.sum() == a.n_obs:
        raise ValueError("All reference cells have the origin label")

    if args.origin_label == args.malignant_label:
        raise ValueError(
            "Cell-of-origin and malignant labels must differ"
        )

    a.var["gene_id"] = gene_id
    a.var["gene_symbol"] = gene_symbol

    a.obs["oncospot_cell_type"] = labels.to_numpy()
    a.obs["oncospot_patient"] = patients.to_numpy()

    if sparse.issparse(x):
        library = np.asarray(x.sum(axis=1)).ravel()
    else:
        library = np.asarray(x.sum(axis=1)).ravel()

    if np.any(library <= 0):
        raise ValueError("Zero-library cells must be filtered")

    a.layers["counts"] = x.astype(np.int64)

    subsets = {
        "malignant": malignant.to_numpy(),
        "nonmalignant": nonmalignant.to_numpy(),
        "cell_origin": origin.to_numpy(),
    }

    for name, mask in subsets.items():
        subset = a[mask].copy()
        subset.write_h5ad(output / f"{name}.h5ad")

    metadata = pd.DataFrame({
        "reference_cell_id": a.obs_names.astype(str),
        "patient": patients.to_numpy(),
        "broad_cell_type": labels.to_numpy(),
        "library_size": library.astype(np.int64),
        "is_malignant": malignant.to_numpy(),
        "is_cell_origin": origin.to_numpy(),
    })

    metadata.to_csv(
        output / "reference_cells.tsv",
        sep="\t",
        index=False,
    )

    summary = {
        "n_cells": int(a.n_obs),
        "n_genes": int(a.n_vars),
        "n_malignant": int(malignant.sum()),
        "n_nonmalignant": int(nonmalignant.sum()),
        "n_cell_origin": int(origin.sum()),
        "n_patients": int(patients.nunique()),
        "cell_type_counts": {
            str(k): int(v)
            for k, v in labels.value_counts().items()
        },
        "hallmark_axes": STATES,
    }

    (output / "reference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(json.dumps(summary, indent=2))
    print(f"\nPrepared reference: {output.resolve()}")


if __name__ == "__main__":
    main()
