#!/usr/bin/env python3
"""Match malignant expression genes to complete clone-level CNA data."""

from pathlib import Path
import argparse
import json
import os
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run = Path(args.run_dir).resolve()
    cfg_path = run / "resolved_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    source = Path(cfg["expression"]["malignant"]["parameter_file"])
    if not source.is_absolute():
        source = ROOT / source

    cna_path = run / "cna/gene_level/clone_gene_cna.tsv"
    if not cna_path.is_file():
        raise FileNotFoundError(cna_path)

    params = pd.read_csv(
        source, sep="\t", dtype={"gene_id": str}
    )
    if params["gene_id"].isna().any():
        raise RuntimeError("Missing expression gene IDs")
    if params["gene_id"].duplicated().any():
        raise RuntimeError("Duplicate expression gene IDs")

    # One row per clone and gene is required.
    cna = pd.read_csv(
        cna_path, sep="\t",
        usecols=["subclone_id", "gene_id", "total_cn"],
        dtype={"subclone_id": str, "gene_id": str}
    )

    if cna.empty:
        raise RuntimeError("CNA table is empty")
    if cna.duplicated(["subclone_id", "gene_id"]).any():
        raise RuntimeError("Duplicate clone-gene CNA pairs")

    n_clones = cna["subclone_id"].nunique()

    valid = cna.loc[
        cna["total_cn"].notna()
    ].groupby("gene_id")["subclone_id"].nunique()

    complete_genes = set(
        valid.index[valid == n_clones]
    )

    included = params.loc[
        params["gene_id"].isin(complete_genes)
    ].copy()

    excluded = params.loc[
        ~params["gene_id"].isin(complete_genes)
    ].copy()

    if included.empty:
        raise RuntimeError(
            "No genes have complete CNA coverage"
        )

    audit = run / "expression/malignant/parameter_audit"
    filtered_path = audit / "malignant_parameters_cna_compatible.tsv"
    excluded_path = audit / "excluded_genes.tsv"
    summary_path = audit / "gene_compatibility_summary.json"
    backup_path = run / "resolved_config.before_gene_reconciliation.yaml"

    for path in [filtered_path, excluded_path, summary_path, backup_path]:
        if path.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing audit: {path}"
            )

    audit.mkdir(parents=True, exist_ok=True)

    # Preserve the original parameter columns and gene values.
    included.to_csv(filtered_path, sep="\t", index=False)

    columns = [
        x for x in ["gene_id", "gene_symbol"]
        if x in excluded.columns
    ]
    excluded[columns].to_csv(
        excluded_path, sep="\t", index=False
    )

    summary = {
        "cancer_type": cfg["cancer_type"],
        "source_parameter_file": str(source),
        "cna_file": str(cna_path),
        "n_cna_clones": int(n_clones),
        "original_genes": int(len(params)),
        "included_genes": int(len(included)),
        "excluded_genes": int(len(excluded)),
        "policy": "intersection_with_complete_clone_CNA",
        "imputed_copy_numbers": False,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    # Keep original resolved configuration for provenance.
    backup_path.write_text(cfg_path.read_text())

    if len(excluded):
        cfg["expression"]["malignant"]["parameter_file"] = (
            str(filtered_path.relative_to(ROOT))
        )
        tmp = cfg_path.with_suffix(".yaml.tmp")
        tmp.write_text(yaml.safe_dump(cfg, sort_keys=False))
        os.replace(tmp, cfg_path)

    print("\nCNA/EXPRESSION GENE RECONCILIATION")
    print("Cancer:", cfg["cancer_type"])
    print("CNA clones:", n_clones)
    print("Original genes:", len(params))
    print("Retained genes:", len(included))
    print("Excluded genes:", len(excluded))
    print("Audit:", audit)
    print("PASS")

if __name__ == "__main__":
    main()
