#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.io import mmread


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a binned empirical Visium spot-depth reference "
            "from one published Visium sample."
        )
    )

    parser.add_argument(
        "--matrix",
        required=True,
        help="10x matrix.mtx.gz file (genes x spots).",
    )

    parser.add_argument(
        "--sample-id",
        required=True,
        help="Reference sample identifier.",
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Reference dataset identifier.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output parameter-bundle directory.",
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=50,
        help="Number of equal-width UMI histogram bins.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    matrix_path = Path(args.matrix).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)

    print("Loading:")
    print(matrix_path)

    # 10x matrix orientation: genes x spots
    matrix = mmread(matrix_path).tocsr()

    spot_umis = np.asarray(
        matrix.sum(axis=0)
    ).ravel().astype(np.int64)

    if len(spot_umis) == 0:
        raise RuntimeError("Reference matrix contains zero spots.")

    if np.any(spot_umis < 0):
        raise RuntimeError("Negative UMI counts detected.")

    n_spots = int(len(spot_umis))

    # --------------------------------------------------------------
    # Build equal-width empirical histogram.
    #
    # Runtime sampling will:
    #   1. sample a bin according to empirical probability;
    #   2. sample an integer UMI depth within that bin.
    # --------------------------------------------------------------

    minimum = int(spot_umis.min())
    maximum = int(spot_umis.max())

    edges = np.linspace(
        minimum,
        maximum + 1,
        args.n_bins + 1,
    )

    counts, edges = np.histogram(
        spot_umis,
        bins=edges,
    )

    records = []

    for j, count in enumerate(counts):
        if count == 0:
            continue

        left = edges[j]
        right = edges[j + 1]

        lower_umi = int(np.ceil(left))

        if j == len(counts) - 1:
            upper_umi = maximum
        else:
            upper_umi = int(np.ceil(right) - 1)

        if upper_umi < lower_umi:
            upper_umi = lower_umi

        records.append(
            {
                "bin_index": len(records),
                "lower_umi": lower_umi,
                "upper_umi": upper_umi,
                "count": int(count),
                "probability": float(count / n_spots),
            }
        )

    histogram = pd.DataFrame(records)

    # Numerical normalization guard.
    histogram["probability"] = (
        histogram["probability"]
        / histogram["probability"].sum()
    )

    histogram_path = (
        output_dir / "depth_histogram.tsv"
    )

    histogram.to_csv(
        histogram_path,
        sep="\t",
        index=False,
    )

    # Keep the original values for provenance/validation.
    raw_path = (
        output_dir / "reference_spot_umis.tsv.gz"
    )

    pd.DataFrame(
        {
            "reference_spot_index": np.arange(
                n_spots,
                dtype=int,
            ),
            "total_umis": spot_umis,
        }
    ).to_csv(
        raw_path,
        sep="\t",
        index=False,
        compression="gzip",
    )

    quantile_probs = [
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
    ]

    quantiles = np.quantile(
        spot_umis,
        quantile_probs,
    )

    metadata = {
        "stage": "visium_depth_reference",
        "technology": "visium_like",
        "reference_dataset": args.dataset,
        "reference_sample": args.sample_id,
        "reference_matrix": str(matrix_path),
        "n_reference_spots": n_spots,
        "histogram_requested_bins": int(args.n_bins),
        "histogram_nonempty_bins": int(len(histogram)),
        "mean_umis": float(np.mean(spot_umis)),
        "median_umis": float(np.median(spot_umis)),
        "std_umis": float(np.std(spot_umis)),
        "min_umis": minimum,
        "max_umis": maximum,
        "q05_umis": float(quantiles[0]),
        "q25_umis": float(quantiles[1]),
        "q50_umis": float(quantiles[2]),
        "q75_umis": float(quantiles[3]),
        "q95_umis": float(quantiles[4]),
        "sampling_model": (
            "sample_histogram_bin_then_uniform_integer_within_bin"
        ),
        "default_depth_multiplier": 1.0,
        "outputs": {
            "histogram": "depth_histogram.tsv",
            "reference_values": "reference_spot_umis.tsv.gz",
        },
    }

    metadata_path = (
        output_dir / "metadata.yaml"
    )

    metadata_path.write_text(
        yaml.safe_dump(
            metadata,
            sort_keys=False,
        )
    )

    # --------------------------------------------------------------
    # Reference plots
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    ax.hist(
        spot_umis,
        bins=args.n_bins,
        edgecolor="black",
    )

    ax.axvline(
        np.median(spot_umis),
        linestyle="--",
        linewidth=1.5,
        label=(
            f"median = "
            f"{np.median(spot_umis):,.0f}"
        ),
    )

    ax.set_xlabel(
        "Total UMIs per spot"
    )

    ax.set_ylabel(
        "Number of spots"
    )

    ax.set_title(
        f"{args.dataset} {args.sample_id}: "
        "Visium spot-depth reference"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / "depth_histogram.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    ax.hist(
        np.log10(spot_umis + 1),
        bins=args.n_bins,
        edgecolor="black",
    )

    ax.set_xlabel(
        "log10(total UMIs + 1)"
    )

    ax.set_ylabel(
        "Number of spots"
    )

    ax.set_title(
        f"{args.dataset} {args.sample_id}: "
        "log-depth distribution"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "depth_histogram_log10.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("Reference bundle written:")
    print(output_dir)

    print()
    print("Summary")
    print("-------")
    print("Spots:  ", n_spots)
    print(
        "Mean:   ",
        f"{np.mean(spot_umis):,.2f}",
    )
    print(
        "Median: ",
        f"{np.median(spot_umis):,.2f}",
    )
    print(
        "Q25:    ",
        f"{quantiles[1]:,.2f}",
    )
    print(
        "Q75:    ",
        f"{quantiles[3]:,.2f}",
    )
    print(
        "Q95:    ",
        f"{quantiles[4]:,.2f}",
    )

    print()
    print("PASS")


if __name__ == "__main__":
    main()
