from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterator

import pandas as pd


AUTOSOMES = tuple(
    str(i)
    for i in range(1, 23)
)


GENE_COLUMNS = [
    "gene_id",
    "gene_id_versioned",
    "gene_symbol",
    "gene_type",
    "chromosome",
    "start",
    "end",
    "strand",
    "source",
    "annotation_release",
    "genome_build",
]


def normalize_chromosome(
    value: str,
) -> str:
    chromosome = str(
        value
    ).strip()

    if chromosome.lower().startswith(
        "chr"
    ):
        chromosome = chromosome[
            3:
        ]

    chromosome = chromosome.upper()

    if not chromosome:
        raise ValueError(
            "Empty chromosome label."
        )

    return chromosome


def strip_ensembl_version(
    identifier: str,
) -> str:
    identifier = str(
        identifier
    ).strip()

    if not identifier:
        raise ValueError(
            "Empty Ensembl identifier."
        )

    return identifier.split(
        ".",
        maxsplit=1,
    )[0]


def parse_gtf_attributes(
    text: str,
) -> dict[str, str]:
    result = {}

    for raw_field in text.split(
        ";"
    ):
        field = raw_field.strip()

        if not field:
            continue

        pieces = field.split(
            maxsplit=1
        )

        if len(pieces) != 2:
            raise ValueError(
                "Malformed GTF attribute: "
                f"{field!r}"
            )

        key, value = pieces

        result[
            key
        ] = (
            value
            .strip()
            .strip('"')
        )

    return result


def iter_gencode_genes(
    path: Path,
    annotation_release: str = "50",
    genome_build: str = "GRCh38.p14",
) -> Iterator[
    dict[str, str | int]
]:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    opener = (
        gzip.open
        if path.suffix == ".gz"
        else open
    )

    with opener(
        path,
        "rt",
        encoding="utf-8",
    ) as handle:

        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if line.startswith(
                "#"
            ):
                continue

            fields = (
                line.rstrip(
                    "\n"
                )
                .split(
                    "\t"
                )
            )

            if len(fields) != 9:
                raise ValueError(
                    "Malformed GTF line "
                    f"{line_number}."
                )

            (
                sequence_name,
                source,
                feature_type,
                start,
                end,
                _score,
                strand,
                _frame,
                attributes_text,
            ) = fields

            if feature_type != "gene":
                continue

            chromosome = (
                normalize_chromosome(
                    sequence_name
                )
            )

            if chromosome not in AUTOSOMES:
                continue

            attributes = (
                parse_gtf_attributes(
                    attributes_text
                )
            )

            required = {
                "gene_id",
                "gene_name",
                "gene_type",
            }

            missing = (
                required
                - set(
                    attributes
                )
            )

            if missing:
                raise ValueError(
                    "Missing GENCODE attributes "
                    f"on line {line_number}: "
                    f"{sorted(missing)}"
                )

            start_i = int(
                start
            )

            end_i = int(
                end
            )

            if start_i < 1:
                raise ValueError(
                    "GENCODE start must "
                    "be >= 1."
                )

            if end_i < start_i:
                raise ValueError(
                    "GENCODE end < start."
                )

            versioned_id = (
                attributes[
                    "gene_id"
                ]
            )

            yield {
                "gene_id":
                    strip_ensembl_version(
                        versioned_id
                    ),

                "gene_id_versioned":
                    versioned_id,

                "gene_symbol":
                    attributes[
                        "gene_name"
                    ],

                "gene_type":
                    attributes[
                        "gene_type"
                    ],

                "chromosome":
                    chromosome,

                "start":
                    start_i,

                "end":
                    end_i,

                "strand":
                    strand,

                "source":
                    source,

                "annotation_release":
                    annotation_release,

                "genome_build":
                    genome_build,
            }


def parse_gencode_genes(
    path: Path,
    annotation_release: str = "50",
    genome_build: str = "GRCh38.p14",
) -> pd.DataFrame:

    frame = pd.DataFrame.from_records(
        iter_gencode_genes(
            path=path,
            annotation_release=(
                annotation_release
            ),
            genome_build=(
                genome_build
            ),
        ),
        columns=GENE_COLUMNS,
    )

    if frame.empty:
        raise ValueError(
            "No autosomal genes retained."
        )

    if frame[
        "gene_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate stable GENCODE "
            "gene IDs found."
        )

    chromosome_order = {
        chromosome: i
        for i, chromosome
        in enumerate(
            AUTOSOMES
        )
    }

    frame[
        "_chrom_order"
    ] = (
        frame[
            "chromosome"
        ].map(
            chromosome_order
        )
    )

    frame = (
        frame
        .sort_values(
            [
                "_chrom_order",
                "start",
                "end",
                "gene_id",
            ]
        )
        .drop(
            columns=[
                "_chrom_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return frame[
        GENE_COLUMNS
    ]
