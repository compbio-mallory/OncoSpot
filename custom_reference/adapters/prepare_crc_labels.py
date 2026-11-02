#!/usr/bin/env python3
"""Map GSE132465 annotations to standardized OncoSpot labels."""

import argparse
from pathlib import Path

import pandas as pd


FIBROBLAST = {
    "Myofibroblasts",
    "Stromal 1",
    "Stromal 2",
    "Stromal 3",
}

ENDOTHELIAL = {
    "Tip-like ECs",
    "Stalk-like ECs",
    "Lymphatic ECs",
    "Proliferative ECs",
}


def classify(row):
    tissue = row["Class"]
    cell_type = row["Cell_type"]
    subtype = row["Cell_subtype"]

    if tissue == "Tumor" and cell_type == "Epithelial cells":
        return "malignant"

    if tissue == "Normal" and cell_type == "Epithelial cells":
        return "normal_epithelial"

    if tissue != "Tumor":
        return None

    if cell_type == "T cells":
        return "T_NK"

    if cell_type == "B cells":
        return "B_plasma"

    if cell_type == "Myeloids":
        return "macrophage_myeloid"

    if cell_type == "Stromal cells":
        if subtype in FIBROBLAST:
            return "fibroblast"
        if subtype in ENDOTHELIAL:
            return "endothelial"

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.annotation, sep="\t", dtype=str)

    required = {
        "Index", "Patient", "Class",
        "Sample", "Cell_type", "Cell_subtype",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing annotation columns: {sorted(missing)}")

    if not df["Index"].is_unique:
        raise ValueError("Duplicate reference cell identifiers")

    df["oncospot_cell_type"] = df.apply(classify, axis=1)

    df["is_cell_origin"] = (
        (df["Class"] == "Normal")
        & (df["Cell_type"] == "Epithelial cells")
        & (df["Cell_subtype"] == "Stem-like/TA")
    )

    df = df.loc[df["oncospot_cell_type"].notna()].copy()

    df = df.rename(columns={
        "Index": "reference_cell_id",
        "Patient": "patient",
        "Sample": "sample",
    })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output, sep="\t", index=False)

    print("\nSTANDARDIZED CELL TYPES")
    print(df["oncospot_cell_type"].value_counts().to_string())

    print("\nCELL-OF-ORIGIN CELLS")
    print(int(df["is_cell_origin"].sum()))

    print("\nPATIENTS")
    print(df["patient"].nunique())

    print("\nOUTPUT")
    print(output.resolve())


if __name__ == "__main__":
    main()
