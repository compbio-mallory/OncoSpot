#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(
        PROJECT_ROOT
        / "src"
    ),
)

from cancer_st_sim.annotation import (  # noqa: E402
    parse_gencode_genes,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--summary",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    genes = parse_gencode_genes(
        path=args.input,
        annotation_release="50",
        genome_build="GRCh38.p14",
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.summary.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    genes.to_csv(
        args.output,
        sep="\t",
        index=False,
    )

    summary = {
        "source":
            str(
                args.input.resolve()
            ),

        "annotation_release":
            "50",

        "genome_build":
            "GRCh38.p14",

        "coordinate_system":
            "one_based_closed",

        "autosomes_only":
            True,

        "gene_type_filter":
            None,

        "n_genes":
            int(
                len(
                    genes
                )
            ),

        "n_unique_gene_ids":
            int(
                genes[
                    "gene_id"
                ].nunique()
            ),

        "n_unique_gene_symbols":
            int(
                genes[
                    "gene_symbol"
                ].nunique()
            ),

        "chromosomes":
            genes[
                "chromosome"
            ]
            .drop_duplicates()
            .tolist(),

        "gene_type_counts": {
            str(k): int(v)
            for k, v in (
                genes[
                    "gene_type"
                ]
                .value_counts()
                .to_dict()
                .items()
            )
        },
    }

    args.summary.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print()
    print("=" * 80)
    print("GENCODE ANNOTATION PREPARED")
    print("=" * 80)

    print(
        "Genes:",
        f"{len(genes):,}",
    )

    print(
        "Unique stable IDs:",
        f'{genes["gene_id"].nunique():,}',
    )

    print(
        "Chromosomes:",
        ", ".join(
            genes[
                "chromosome"
            ]
            .drop_duplicates()
            .tolist()
        ),
    )

    print(
        "Gene types:",
        genes[
            "gene_type"
        ].nunique(),
    )

    print()


if __name__ == "__main__":
    main()
