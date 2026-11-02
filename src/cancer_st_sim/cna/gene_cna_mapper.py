from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cancer_st_sim.annotation import (
    normalize_chromosome,
)


class GeneCNAMapperError(
    RuntimeError
):
    pass


@dataclass
class GeneCNAMapperResult:
    gene_bin_mapping: pd.DataFrame
    clone_gene_cna: pd.DataFrame
    gene_cna_summary: pd.DataFrame
    summary: dict[str, Any]


def build_gene_bin_mapping(
    genes: pd.DataFrame,
    clone_bin_cna: pd.DataFrame,
) -> pd.DataFrame:

    required_genes = {
        "gene_id",
        "gene_id_versioned",
        "gene_symbol",
        "gene_type",
        "chromosome",
        "start",
        "end",
    }

    missing = (
        required_genes
        - set(
            genes.columns
        )
    )

    if missing:
        raise GeneCNAMapperError(
            "Genes missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    required_bins = {
        "subclone_id",
        "chrom",
        "start",
        "end",
        "total_cn",
    }

    missing = (
        required_bins
        - set(
            clone_bin_cna.columns
        )
    )

    if missing:
        raise GeneCNAMapperError(
            "CNA table missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    # The bin coordinates are identical for all clones,
    # so use one clone to construct the coordinate map.
    first_clone = str(
        clone_bin_cna[
            "subclone_id"
        ].iloc[0]
    )

    bins = (
        clone_bin_cna.loc[
            clone_bin_cna[
                "subclone_id"
            ].astype(str)
            == first_clone,
            [
                "chrom",
                "start",
                "end",
            ],
        ]
        .drop_duplicates()
        .copy()
    )

    bins[
        "chromosome"
    ] = (
        bins[
            "chrom"
        ]
        .astype(str)
        .map(
            normalize_chromosome
        )
    )

    # CNAsim:
    #   zero-based half-open [start, end)
    #
    # Canonical:
    #   one-based closed [start + 1, end]
    bins[
        "bin_start"
    ] = (
        pd.to_numeric(
            bins[
                "start"
            ],
            errors="raise",
        )
        .astype(int)
        + 1
    )

    bins[
        "bin_end"
    ] = (
        pd.to_numeric(
            bins[
                "end"
            ],
            errors="raise",
        )
        .astype(int)
    )

    bins[
        "source_bin_start"
    ] = (
        pd.to_numeric(
            bins[
                "start"
            ],
            errors="raise",
        )
        .astype(int)
    )

    bins[
        "source_bin_end"
    ] = (
        pd.to_numeric(
            bins[
                "end"
            ],
            errors="raise",
        )
        .astype(int)
    )

    records = []

    for chromosome, gene_group in (
        genes.groupby(
            "chromosome",
            sort=False,
        )
    ):

        chromosome = str(
            chromosome
        )

        chrom_bins = (
            bins.loc[
                bins[
                    "chromosome"
                ]
                == chromosome
            ]
            .sort_values(
                [
                    "bin_start",
                    "bin_end",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        if chrom_bins.empty:
            continue

        for gene in gene_group.itertuples(
            index=False
        ):

            gene_start = int(
                gene.start
            )

            gene_end = int(
                gene.end
            )

            candidates = (
                chrom_bins.loc[
                    (
                        chrom_bins[
                            "bin_end"
                        ]
                        >= gene_start
                    )
                    & (
                        chrom_bins[
                            "bin_start"
                        ]
                        <= gene_end
                    )
                ]
                .copy()
            )

            if candidates.empty:
                continue

            candidates[
                "overlap_bp"
            ] = (
                candidates.apply(
                    lambda row:
                        min(
                            gene_end,
                            int(
                                row[
                                    "bin_end"
                                ]
                            ),
                        )
                        - max(
                            gene_start,
                            int(
                                row[
                                    "bin_start"
                                ]
                            ),
                        )
                        + 1,
                    axis=1,
                )
            )

            candidates = (
                candidates
                .sort_values(
                    [
                        "overlap_bp",
                        "bin_start",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                )
            )

            best = candidates.iloc[
                0
            ]

            gene_length = (
                gene_end
                - gene_start
                + 1
            )

            records.append(
                {
                    "gene_id":
                        gene.gene_id,

                    "gene_id_versioned":
                        gene.gene_id_versioned,

                    "gene_symbol":
                        gene.gene_symbol,

                    "gene_type":
                        gene.gene_type,

                    "chromosome":
                        chromosome,

                    "gene_start":
                        gene_start,

                    "gene_end":
                        gene_end,

                    "gene_length_bp":
                        gene_length,

                    "source_bin_start":
                        int(
                            best[
                                "source_bin_start"
                            ]
                        ),

                    "source_bin_end":
                        int(
                            best[
                                "source_bin_end"
                            ]
                        ),

                    "bin_start":
                        int(
                            best[
                                "bin_start"
                            ]
                        ),

                    "bin_end":
                        int(
                            best[
                                "bin_end"
                            ]
                        ),

                    "overlap_bp":
                        int(
                            best[
                                "overlap_bp"
                            ]
                        ),

                    "overlap_fraction":
                        float(
                            best[
                                "overlap_bp"
                            ]
                            / gene_length
                        ),

                    "n_overlapping_bins":
                        int(
                            len(
                                candidates
                            )
                        ),
                }
            )

    mapping = pd.DataFrame(
        records
    )

    if mapping.empty:
        raise GeneCNAMapperError(
            "No genes mapped to CNAsim bins."
        )

    if mapping[
        "gene_id"
    ].duplicated().any():
        raise GeneCNAMapperError(
            "Gene-bin mapping produced "
            "duplicate gene IDs."
        )

    return mapping


def build_clone_gene_cna(
    mapping: pd.DataFrame,
    clone_bin_cna: pd.DataFrame,
) -> pd.DataFrame:

    bins = clone_bin_cna.copy()

    bins[
        "chromosome"
    ] = (
        bins[
            "chrom"
        ]
        .astype(str)
        .map(
            normalize_chromosome
        )
    )

    bins[
        "source_bin_start"
    ] = pd.to_numeric(
        bins[
            "start"
        ],
        errors="raise",
    ).astype(int)

    bins[
        "source_bin_end"
    ] = pd.to_numeric(
        bins[
            "end"
        ],
        errors="raise",
    ).astype(int)

    result = mapping.merge(
        bins[
            [
                "subclone_id",
                "cnasim_node_id",
                "cnasim_tree_role",
                "biological_role",
                "malignant_cells",
                "chromosome",
                "source_bin_start",
                "source_bin_end",
                "allele_a_cn",
                "allele_b_cn",
                "total_cn",
            ]
        ],
        on=[
            "chromosome",
            "source_bin_start",
            "source_bin_end",
        ],
        how="inner",
        validate="many_to_many",
    )

    expected = (
        mapping[
            "gene_id"
        ].nunique()
        * bins[
            "subclone_id"
        ].nunique()
    )

    if len(result) != expected:
        raise GeneCNAMapperError(
            "Unexpected clone-gene CNA row count. "
            f"Expected {expected:,}; "
            f"found {len(result):,}."
        )

    columns = [
        "subclone_id",
        "cnasim_node_id",
        "cnasim_tree_role",
        "biological_role",
        "malignant_cells",
        "gene_id",
        "gene_id_versioned",
        "gene_symbol",
        "gene_type",
        "chromosome",
        "gene_start",
        "gene_end",
        "allele_a_cn",
        "allele_b_cn",
        "total_cn",
        "source_bin_start",
        "source_bin_end",
        "overlap_bp",
        "overlap_fraction",
        "n_overlapping_bins",
    ]

    return result[
        columns
    ]


def build_gene_cna_summary(
    clone_gene_cna: pd.DataFrame,
) -> pd.DataFrame:

    observed = (
        clone_gene_cna.loc[
            clone_gene_cna[
                "malignant_cells"
            ] > 0
        ]
        .copy()
    )

    summary = (
        observed
        .groupby(
            [
                "gene_id",
                "gene_symbol",
                "gene_type",
                "chromosome",
                "gene_start",
                "gene_end",
            ],
            as_index=False,
        )
        .agg(
            min_total_cn=(
                "total_cn",
                "min",
            ),

            max_total_cn=(
                "total_cn",
                "max",
            ),

            n_unique_cn_states=(
                "total_cn",
                "nunique",
            ),

            n_clones=(
                "subclone_id",
                "nunique",
            ),
        )
    )

    summary[
        "has_cna_variation"
    ] = (
        summary[
            "n_unique_cn_states"
        ] > 1
    )

    return summary


def write_gene_level_cna(
    genes_path: Path,
    clone_bin_cna_path: Path,
    output_dir: Path,
) -> GeneCNAMapperResult:

    genes = pd.read_csv(
        genes_path,
        sep="\t",
    )

    clone_bin_cna = pd.read_csv(
        clone_bin_cna_path,
        sep="\t",
    )

    mapping = build_gene_bin_mapping(
        genes=genes,
        clone_bin_cna=clone_bin_cna,
    )

    clone_gene_cna = (
        build_clone_gene_cna(
            mapping=mapping,
            clone_bin_cna=(
                clone_bin_cna
            ),
        )
    )

    gene_cna_summary = (
        build_gene_cna_summary(
            clone_gene_cna
        )
    )

    n_clones = int(
        clone_gene_cna[
            "subclone_id"
        ].nunique()
    )

    n_genes = int(
        mapping[
            "gene_id"
        ].nunique()
    )

    observed = (
        clone_gene_cna.loc[
            clone_gene_cna[
                "malignant_cells"
            ] > 0
        ]
    )

    weighted_total = float(
        (
            observed[
                "malignant_cells"
            ]
            * observed[
                "total_cn"
            ]
        ).sum()
    )

    weighted_denominator = float(
        observed[
            "malignant_cells"
        ].sum()
    )

    summary = {
        "source":
            "CNAsim_plus_GENCODE_v50",

        "genome_build":
            "GRCh38.p14",

        "coordinate_contract": {
            "cnasim_source":
                "zero_based_half_open",

            "canonical":
                "one_based_closed",

            "gencode":
                "one_based_closed",
        },

        "gene_assignment_rule":
            "largest_gene_body_overlap",

        "tie_break_rule":
            "lowest_genomic_bin_start",

        "n_input_genes":
            int(
                len(
                    genes
                )
            ),

        "n_mapped_genes":
            n_genes,

        "n_unmapped_genes":
            int(
                len(
                    genes
                )
                - n_genes
            ),

        "n_lineage_nodes":
            n_clones,

        "clone_gene_cna_rows":
            int(
                len(
                    clone_gene_cna
                )
            ),

        "genes_spanning_multiple_bins":
            int(
                (
                    mapping[
                        "n_overlapping_bins"
                    ] > 1
                ).sum()
            ),

        "genes_with_cna_variation_across_observed_clones":
            int(
                gene_cna_summary[
                    "has_cna_variation"
                ].sum()
            ),

        "mean_total_cn_across_malignant_cell_gene_pairs":
            float(
                weighted_total
                / weighted_denominator
            ),
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mapping.to_csv(
        output_dir
        / "gene_bin_mapping.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
    )

    clone_gene_cna.to_csv(
        output_dir
        / "clone_gene_cna.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
    )

    gene_cna_summary.to_csv(
        output_dir
        / "gene_cna_summary.tsv",
        sep="\t",
        index=False,
    )

    with (
        output_dir
        / "gene_cna_summary.yaml"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(
            summary,
            handle,
            sort_keys=False,
        )

    return GeneCNAMapperResult(
        gene_bin_mapping=mapping,
        clone_gene_cna=clone_gene_cna,
        gene_cna_summary=gene_cna_summary,
        summary=summary,
    )
