#!/usr/bin/env python3
"""Generalized OncoSpot Hallmark-state and malignant expression calibration.

Input: gene-symbol x cell raw UMI TSV.gz, standardized annotations,
       unique symbol-to-Ensembl mapping, Hallmark GMT and origin baseline.
Output: empirical tumor states, malignant library sizes, gene-level
        alpha/beta/gamma/dispersion and provenance metadata.

Follows the CRC/HCC provisional calibration model. beta is assumed,
not empirically estimated from paired DNA/RNA measurements.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

STATES = ["PROLIFERATION", "WNT", "MYC",
          "HYPOXIA", "EMT", "METABOLIC"]

STATE_SETS = {
    "PROLIFERATION": ["HALLMARK_E2F_TARGETS",
                      "HALLMARK_G2M_CHECKPOINT"],
    "WNT": ["HALLMARK_WNT_BETA_CATENIN_SIGNALING"],
    "MYC": ["HALLMARK_MYC_TARGETS_V1"],
    "HYPOXIA": ["HALLMARK_HYPOXIA"],
    "EMT": ["HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"],
    "METABOLIC": ["HALLMARK_GLYCOLYSIS"],
}


def read_signatures(path):
    sets = {}
    with open(path) as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                sets[fields[0]] = set(fields[2:])
    missing = sorted(set(sum(STATE_SETS.values(), [])) - sets.keys())
    if missing:
        raise ValueError(f"Missing Hallmark gene sets: {missing}")
    return {
        name: set().union(*(sets[s] for s in sources))
        for name, sources in STATE_SETS.items()
    }


def matrix_rows(path, width):
    with gzip.open(path, "rt") as f:
        next(f)
        for symbol, sep, rest in (
            line.partition("\t") for line in f
        ):
            if not sep:
                raise ValueError(f"Malformed matrix row: {symbol}")
            y = np.fromstring(rest, sep="\t", dtype=np.int64)
            if len(y) != width or (y < 0).any():
                raise ValueError(f"Invalid matrix counts: {symbol}")
            yield symbol, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--counts", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--gene-mapping", required=True)
    p.add_argument("--origin-baseline", required=True)
    p.add_argument("--gmt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cancer", required=True)
    p.add_argument("--beta", type=float, default=0.4)
    p.add_argument("--ridge-lambda-per-cell", type=float, default=0.05)
    p.add_argument("--gamma-cap", type=float, default=0.30)
    args = p.parse_args()

    if args.beta <= 0 or args.ridge_lambda_per_cell <= 0:
        p.error("beta and ridge lambda must be positive")
    if args.gamma_cap <= 0:
        p.error("gamma cap must be positive")

    out = Path(args.output)
    state_dir = out / "tumor_states"
    expr_dir = out / "malignant/expression"
    qc_dir = out / "qc"
    for d in (state_dir, expr_dir, qc_dir):
        d.mkdir(parents=True, exist_ok=True)

    annotations = pd.read_csv(args.annotations, sep="\t", dtype=str)
    malignant = annotations.loc[
        annotations["oncospot_cell_type"] == "malignant"
    ].copy()

    if malignant.empty or not malignant["reference_cell_id"].is_unique:
        raise ValueError("Missing or duplicate malignant cells")

    if malignant["patient"].isna().any():
        raise ValueError("Missing patient identifiers")

    with gzip.open(args.counts, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
    if header[0] != "Index" or len(set(header)) != len(header):
        raise ValueError("Invalid matrix header")

    width = len(header) - 1
    indices = {cell: i for i, cell in enumerate(header[1:])}
    if not malignant["reference_cell_id"].isin(indices).all():
        raise ValueError("Some malignant cells are absent from matrix")

    cols = malignant["reference_cell_id"].map(indices).to_numpy(
        dtype=np.int64
    )
    n = len(cols)

    mapping = pd.read_csv(args.gene_mapping, sep="\t", dtype=str)
    mapping = mapping.loc[
        mapping["status"].eq("unique"),
        ["gene_symbol", "gene_id"]
    ].dropna().copy()
    if mapping["gene_symbol"].duplicated().any():
        raise ValueError("Duplicate mapped gene symbols")
    if mapping["gene_id"].duplicated().any():
        raise ValueError("Duplicate mapped Ensembl IDs")
    allowed = set(mapping["gene_symbol"])

    origin = pd.read_csv(args.origin_baseline, sep="\t")
    origin = origin.loc[
        origin["baseline_fraction_donor_balanced"] > 0
    ].copy()
    if origin.empty or not origin["gene_symbol"].is_unique:
        raise ValueError("Invalid positive-origin gene set")
    if not set(origin["gene_symbol"]).issubset(allowed):
        raise ValueError("Origin genes missing from gene mapping")
    lookup = origin.set_index("gene_symbol").to_dict("index")

    signatures = read_signatures(args.gmt)
    needed = set().union(*signatures.values())

    # Pass 1: empirical malignant library sizes over ALL mapped genes.
    L = np.zeros(n, dtype=np.int64)
    mapped_seen = set()
    for symbol, y in matrix_rows(args.counts, width):
        if symbol not in allowed:
            continue
        if symbol in mapped_seen:
            raise ValueError(f"Duplicate matrix gene: {symbol}")
        mapped_seen.add(symbol)
        L += y[cols]
    if np.any(L <= 0):
        raise ValueError("Nonpositive malignant library sizes")

    lib = pd.DataFrame({
        "cell_id": malignant["reference_cell_id"].to_numpy(),
        "patient": malignant["patient"].to_numpy(),
        "sample": malignant["sample"].to_numpy()
                  if "sample" in malignant else
                  malignant["patient"].to_numpy(),
        "cell_subtype": malignant["Cell_subtype"].to_numpy()
                  if "Cell_subtype" in malignant else
                  np.repeat("unspecified", n),
        "library_size": L,
    })
    lib.to_csv(expr_dir / "malignant_library_sizes.tsv",
               sep="\t", index=False)

    # Pass 2: score the six Hallmark axes using malignant reference.
    scores = {}
    for symbol, y in matrix_rows(args.counts, width):
        if symbol not in needed or symbol not in allowed:
            continue
        x = np.log1p(1e4 * y[cols].astype(float) / L)
        sd = x.std()
        if sd > 1e-8:
            scores[symbol] = (x - x.mean()) / sd

    Zcols = []
    coverage = {}
    for state in STATES:
        usable = sorted(signatures[state] & scores.keys())
        if len(usable) < 10:
            raise ValueError(
                f"{state}: only {len(usable)} usable signature genes"
            )
        raw = np.mean(np.stack([scores[g] for g in usable]), axis=0)
        sd = raw.std()
        if sd <= 1e-8:
            raise ValueError(f"{state}: near-zero score variance")
        Zcols.append((raw - raw.mean()) / sd)
        coverage[state] = {
            "signature_genes": len(signatures[state]),
            "usable_genes": len(usable),
            "coverage_fraction": len(usable) / len(signatures[state]),
            "source_sets": STATE_SETS[state],
        }

    Z = np.column_stack(Zcols)
    state_frame = pd.DataFrame(
        Z, columns=STATES
    )
    state_frame.insert(
        0, "reference_cell_id",
        malignant["reference_cell_id"].to_numpy()
    )
    state_frame.to_csv(
        state_dir / "state_reference.tsv",
        sep="\t", index=False
    )
    state_frame[STATES].corr().to_csv(
        qc_dir / "state_correlations.tsv", sep="\t"
    )

    with (state_dir / "state_metadata.yaml").open("w") as f:
        yaml.safe_dump({
            "cancer_type": args.cancer,
            "source_gene_sets": str(args.gmt),
            "scoring_method":
                "mean_of_within_gene_standardized_log1p_CPM",
            "state_scale": "global_mean_0_population_sd_1",
            "reference_cells": int(n),
            "states": STATES,
            "signature_coverage": coverage,
            "model": {
                "type": "empirical_cell_state_sampling",
                "sampling": "row_with_replacement",
                "clone_conditioning": False,
                "lineage_inheritance": False
            },
            "status": "provisional_pending_simulation_validation"
        }, f, sort_keys=False)

    # Patient-centered six-axis ridge design.
    patients, codes = np.unique(
        malignant["patient"].to_numpy(),
        return_inverse=True
    )
    sizes = np.bincount(codes).astype(float)

    def center(y):
        means = np.bincount(
            codes, weights=y, minlength=len(patients)
        ) / sizes
        return y - means[codes]

    X = np.column_stack([center(Z[:, j]) for j in range(6)])
    lam = args.ridge_lambda_per_cell * n
    A = X.T @ X + lam * np.eye(6)
    if not np.isfinite(np.linalg.cond(A)):
        raise ValueError("Invalid ridge design")
    A_inv = np.linalg.inv(A)

    # Pass 3: fit one malignant expression row per positive-origin gene.
    output_rows = []
    qc_rows = []
    seen = set()

    for symbol, values in matrix_rows(args.counts, width):
        if symbol not in lookup:
            continue
        if symbol in seen:
            raise ValueError(f"Duplicate gene: {symbol}")
        seen.add(symbol)

        y = values[cols].astype(float)
        mean_count = float(y.mean())
        detection = float(np.mean(y > 0))
        eligible = detection >= 0.01 and mean_count >= 0.05

        gamma = np.zeros(6, dtype=float)
        if eligible:
            yc = center(np.log1p(1e4 * y / L))
            gamma = np.clip(
                A_inv @ (X.T @ yc),
                -args.gamma_cap, args.gamma_cap
            )

        effect = Z @ gamma
        lower, upper = np.quantile(effect, [0.005, 0.995])

        phi_raw = np.nan
        if eligible:
            relative = np.exp(np.clip(effect, lower, upper))
            exposure = L * relative
            abundance = y.sum() / exposure.sum()
            mu = abundance * exposure
            denominator = np.sum(mu * mu)
            if denominator > 0:
                phi_raw = float(
                    np.sum((y - mu) ** 2 - mu) / denominator
                )

        fallback = not np.isfinite(phi_raw)
        phi = 0.5 if fallback else float(np.clip(phi_raw, 0, 10))

        info = lookup[symbol]
        baseline = float(info["baseline_fraction_donor_balanced"])

        row = {
            "gene_id": info["gene_id"],
            "gene_symbol": symbol,
            "baseline_fraction": baseline,
            "alpha": float(np.log(baseline)),
            "beta": args.beta,
            "dispersion": phi,
            "beta_source": "provisional_fixed_user_supplied",
            "beta_replaced": False,
            "dispersion_source": (
                "pooled_fallback" if fallback
                else "tumor_residual_moments"
            ),
            "gamma_source": (
                "patient_centered_ridge_logcpm"
                if eligible else "zero_fallback"
            ),
            "gamma_calibrated": bool(eligible),
            "state_effect_lower": float(lower),
            "state_effect_upper": float(upper),
        }
        for j, state in enumerate(STATES):
            row[f"gamma_{state}"] = float(gamma[j])
        output_rows.append(row)

        qc_rows.append({
            "gene_id": info["gene_id"],
            "gene_symbol": symbol,
            "mean_tumor_count": mean_count,
            "tumor_detection_fraction": detection,
            "ridge_fit_eligible": bool(eligible),
            "raw_dispersion": phi_raw,
            "final_dispersion": phi,
            "dispersion_fallback": fallback,
            "gamma_at_cap": bool(
                np.any(np.abs(gamma) >= args.gamma_cap - 1e-10)
            )
        })

    if len(seen) != len(origin):
        raise ValueError(
            f"{len(origin) - len(seen)} positive origin genes "
            "not found in count matrix"
        )

    order = origin["gene_id"].to_numpy()
    params = pd.DataFrame(output_rows).set_index(
        "gene_id"
    ).loc[order].reset_index()
    qc = pd.DataFrame(qc_rows).set_index(
        "gene_id"
    ).loc[order].reset_index()

    numeric = ["alpha", "beta", "dispersion",
               "state_effect_lower", "state_effect_upper"] + [
                   f"gamma_{s}" for s in STATES
               ]
    if not np.isfinite(params[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite malignant parameters")

    params.to_csv(
        expr_dir / "malignant_expression_parameters.tsv",
        sep="\t", index=False, float_format="%.12g"
    )
    qc.to_csv(qc_dir / "malignant_parameter_diagnostics.tsv",
              sep="\t", index=False)

    with (expr_dir / "expression_metadata.yaml").open("w") as f:
        yaml.safe_dump({
            "cancer_type": args.cancer,
            "status": "provisional_requires_simulation_validation",
            "n_reference_cells": int(n),
            "n_reference_patients": int(len(patients)),
            "n_runtime_genes": int(len(params)),
            "states": STATES,
            "coefficient_estimation": {
                "model": "patient_centered_ridge_log1p_cpm",
                "not_exact_nb2_glm": True,
                "ridge_lambda_per_cell": args.ridge_lambda_per_cell,
                "gamma_absolute_cap": args.gamma_cap,
                "minimum_detection": 0.01,
                "minimum_mean_count": 0.05,
                "signature_gene_circularity":
                    "Not corrected; sensitivity analysis recommended"
            },
            "cna_dosage": {
                "beta": args.beta,
                "source": "illustrative_positive_fixed_coefficient",
                "calibrated_from_dna": False
            },
            "dispersion": {
                "model": "NB2",
                "estimator": "tumor_scale_residual_method_of_moments",
                "fallback": 0.5,
                "maximum": 10.0
            },
            "state_effect_bounds": {
                "lower_quantile": 0.005,
                "upper_quantile": 0.995,
                "applies_to": "combined_gamma_dot_state"
            }
        }, f, sort_keys=False)

    summary = {
        "cancer": args.cancer,
        "malignant_cells": n,
        "patients": len(patients),
        "positive_origin_genes": len(params),
        "ridge_eligible_genes": int(qc["ridge_fit_eligible"].sum()),
        "dispersion_fallback_genes": int(
            qc["dispersion_fallback"].sum()
        ),
        "signature_coverage": {
            s: coverage[s]["coverage_fraction"] for s in STATES
        }
    }
    (out / "malignant_calibration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    print("MALIGNANT CALIBRATION: PASS")


if __name__ == "__main__":
    main()
