#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ONCOSPOT_PYTHON:-python3}"

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <input_config.yaml> <new_run_directory>"
    exit 2
fi

cd "$PROJECT"

CONFIG=$("$PYTHON" -c \
    'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1")

RUN=$("$PYTHON" -c \
    'import os,sys; print(os.path.realpath(sys.argv[1]))' "$2")

[[ -f "$CONFIG" ]] || {
    echo "Missing input config: $CONFIG"
    exit 1
}

# Runs must remain under the designated results directory.
ALLOWED_RUN_ROOT=$("$PYTHON" -c     'import os,sys; print(os.path.realpath(sys.argv[1]))'     "$PROJECT/results/runs")

case "$RUN" in
    "$ALLOWED_RUN_ROOT/"*) ;;
    *)
        echo "ERROR: Run directory must be under results/runs/"
        echo "Resolved run: $RUN"
        echo "Allowed root: $ALLOWED_RUN_ROOT"
        exit 1
        ;;
esac

# Never overwrite an existing simulation.
if [[ -e "$RUN" ]]; then
    echo "ERROR: Run directory already exists: $RUN"
    echo "Choose a new run name."
    exit 1
fi

mkdir -p "$RUN"

export PYTHONPATH="$PROJECT/src:${PYTHONPATH:-}"

echo "======================================================================"
echo "ST SIMULATOR — CANCER-SPECIFIC RUN"
echo "======================================================================"
echo "Start:  $(date)"
echo "Host:   $(hostname)"
echo "Config: $CONFIG"
echo "Run:    $RUN"
echo "======================================================================"

run_stage () {
    local stage="$1"
    shift

    echo
    echo "======================================================================"
    echo "$stage"
    echo "======================================================================"
    echo "Start: $(date)"
    "$@"
    echo "Finished: $(date)"
}

# ----------------------------------------------------------------------
# 1. Resolve simulator configuration
# ----------------------------------------------------------------------

run_stage \
    "[1/16] RESOLVE CONFIGURATION" \
    "$PYTHON" scripts/pipeline/resolve_config.py \
        --config "$CONFIG" \
        --run-dir "$RUN"


# ----------------------------------------------------------------------
# Read Visium settings from resolved configuration
# ----------------------------------------------------------------------

echo
echo "Reading Visium observation settings..."

mapfile -t VISIUM_SETTINGS < <(
"$PYTHON" - "$RUN/resolved_config.yaml" "$PROJECT" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
project = Path(sys.argv[2])

cfg = yaml.safe_load(config_path.read_text())

obs = cfg.get("observation", {})
visium = obs.get("visium", {})

enabled = bool(visium.get("enabled", True))

geometry = visium.get("geometry", {})
target_cells = float(
    geometry.get("target_median_cells_per_spot", 2)
)

depth_ref = visium.get("depth_reference", {})
parameter_dir = depth_ref.get(
    "parameter_dir",
    "parameters/observation/visium/gse211956_sp4",
)

parameter_dir = Path(parameter_dir)
if not parameter_dir.is_absolute():
    parameter_dir = project / parameter_dir

depth_multiplier = float(
    visium.get("depth_multiplier", 1.0)
)

print("1" if enabled else "0")
print(target_cells)
print(parameter_dir)
print(depth_multiplier)
PY
)

VISIUM_ENABLED="${VISIUM_SETTINGS[0]}"
VISIUM_TARGET_MEDIAN_CELLS="${VISIUM_SETTINGS[1]}"
VISIUM_DEPTH_REFERENCE="${VISIUM_SETTINGS[2]}"
VISIUM_DEPTH_MULTIPLIER="${VISIUM_SETTINGS[3]}"

echo "Visium enabled:                  $VISIUM_ENABLED"
echo "Target median cells per spot:   $VISIUM_TARGET_MEDIAN_CELLS"
echo "Depth reference:                 $VISIUM_DEPTH_REFERENCE"
echo "Depth multiplier:                $VISIUM_DEPTH_MULTIPLIER"

if [[ "$VISIUM_ENABLED" == "1" ]]; then
    if [[ ! -f "$VISIUM_DEPTH_REFERENCE/depth_histogram.tsv" ]]; then
        echo
        echo "ERROR: Visium depth histogram not found:"
        echo "$VISIUM_DEPTH_REFERENCE/depth_histogram.tsv"
        exit 1
    fi

    if [[ ! -f "$VISIUM_DEPTH_REFERENCE/metadata.yaml" ]]; then
        echo
        echo "ERROR: Visium depth metadata not found:"
        echo "$VISIUM_DEPTH_REFERENCE/metadata.yaml"
        exit 1
    fi
fi


# ----------------------------------------------------------------------
# 2. Build DEMON config
# ----------------------------------------------------------------------

mkdir -p "$RUN/demon"

run_stage \
    "[2/16] BUILD DEMON CONFIGURATION" \
    "$PYTHON" scripts/pipeline/build_demon_config.py \
        --resolved-config "$RUN/resolved_config.yaml" \
        --output "$RUN/demon/config.dat"


# ----------------------------------------------------------------------
# 3. Run DEMON
# ----------------------------------------------------------------------

run_stage \
    "[3/16] RUN DEMON" \
    "$PYTHON" scripts/pipeline/run_demon.py \
        --run-dir "$RUN"


# ----------------------------------------------------------------------
# 4. Build canonical DEMON scaffold
# ----------------------------------------------------------------------

run_stage \
    "[4/16] BUILD DEMON SCAFFOLD" \
    "$PYTHON" scripts/pipeline/build_demon_scaffold.py \
        --run-dir "$RUN"


# ----------------------------------------------------------------------
# 5. Build cell scaffold
# ----------------------------------------------------------------------

run_stage \
    "[5/16] BUILD CELL SCAFFOLD" \
    "$PYTHON" scripts/pipeline/build_cell_scaffold.py \
        --run-dir "$RUN"


# ----------------------------------------------------------------------
# 6. Assign malignant / nonmalignant compartments
# ----------------------------------------------------------------------

run_stage \
    "[6/16] ASSIGN TISSUE COMPARTMENTS" \
    "$PYTHON" scripts/pipeline/assign_tissue_compartments.py \
        --run-dir "$RUN"


# ----------------------------------------------------------------------
# 7. Assign broad nonmalignant cell types
# ----------------------------------------------------------------------

run_stage \
    "[7/16] ASSIGN NONMALIGNANT CELL TYPES" \
    "$PYTHON" scripts/pipeline/assign_nonmalignant_cell_types.py \
        --run-dir "$RUN"


# ----------------------------------------------------------------------
# 8. Build malignant tree for CNAsim
#    Optional rare-clone collapsing uses a preliminary tree to measure
#    phylogenetic distance, then rebuilds the final tree from merged
#    benchmark clone labels.
# ----------------------------------------------------------------------

CLONE_MERGE_ENABLED="$(
"$PYTHON" - "$RUN/resolved_config.yaml" <<'PYCFG'
from pathlib import Path
import sys
import yaml

cfg = yaml.safe_load(
    Path(sys.argv[1]).read_text()
)

enabled = bool(
    cfg
    .get("malignant", {})
    .get("clone_merging", {})
    .get("enabled", False)
)

print("1" if enabled else "0")
PYCFG
)"

if [[ "$CLONE_MERGE_ENABLED" == "1" ]]; then

    run_stage \
        "[8a/16] BUILD PRE-MERGE MALIGNANT TREE" \
        "$PYTHON" scripts/pipeline/build_cnasim_tree.py \
            --run-dir "$RUN"

    run_stage \
        "[8b/16] MERGE RARE MALIGNANT SUBCLONES" \
        "$PYTHON" scripts/pipeline/merge_rare_malignant_subclones.py \
            --run-dir "$RUN"

    if [[ -e "$RUN/cna/bridge_pre_clone_merge" ]]; then
        echo "ERROR: pre-merge bridge backup already exists:"
        echo "$RUN/cna/bridge_pre_clone_merge"
        exit 1
    fi

    mv \
        "$RUN/cna/bridge" \
        "$RUN/cna/bridge_pre_clone_merge"

    run_stage \
        "[8c/16] BUILD FINAL CNASIM TREE" \
        "$PYTHON" scripts/pipeline/build_cnasim_tree.py \
            --run-dir "$RUN"

else

    run_stage \
        "[8/16] BUILD CNASIM TREE" \
        "$PYTHON" scripts/pipeline/build_cnasim_tree.py \
            --run-dir "$RUN"

fi


# ----------------------------------------------------------------------
# 9. Run CNAsim
# ----------------------------------------------------------------------

run_stage \
    "[9/16] RUN CNASIM" \
    "$PYTHON" scripts/pipeline/run_cnasim.py \
        --run-dir "$RUN"


# ----------------------------------------------------------------------
# 10. Process CNAsim output
# ----------------------------------------------------------------------

run_stage \
    "[10/16] PROCESS CNASIM OUTPUT" \
    "$PYTHON" scripts/pipeline/process_cnasim_output.py \
        --run-dir "$RUN"


# ----------------------------------------------------------------------
# 11. Map CNA profiles to genes
# ----------------------------------------------------------------------

run_stage \
    "[11/16] MAP CNA TO GENES" \
    "$PYTHON" scripts/pipeline/map_cna_to_genes.py \
        --run-dir "$RUN"


# ----------------------------------------------------------------------
# 12. Assign malignant tumor states
# ----------------------------------------------------------------------

run_stage \
    "[12/16] ASSIGN TUMOR STATES" \
    "$PYTHON" scripts/pipeline/assign_tumor_states.py \
        --run-dir "$RUN"


# ----------------------------------------------------------------------
# 13. Generate malignant expression
# ----------------------------------------------------------------------

run_stage \
    "[12b/16] RECONCILE EXPRESSION/CNA GENES" \
    "$PYTHON" scripts/pipeline/reconcile_expression_cna_genes.py \
        --run-dir "$RUN"


run_stage \
    "[13/16] GENERATE MALIGNANT EXPRESSION" \
    "$PYTHON" scripts/pipeline/generate_malignant_expression.py \
        --run-dir "$RUN"


# ----------------------------------------------------------------------
# 14. Generate nonmalignant expression
# ----------------------------------------------------------------------

run_stage \
    "[14/16] GENERATE NONMALIGNANT EXPRESSION" \
    "$PYTHON" scripts/pipeline/generate_nonmalignant_expression.py \
        --run-dir "$RUN"


# ----------------------------------------------------------------------
# 15. Merge latent expression + spatial outputs
# ----------------------------------------------------------------------

echo
echo "======================================================================"
echo "[15/16] FINALIZE LATENT + SPATIAL OUTPUTS"
echo "======================================================================"
echo "Start: $(date)"
echo

"$PYTHON" scripts/pipeline/merge_expression_outputs.py \
    --run-dir "$RUN"

"$PYTHON" scripts/pipeline/build_spatial_cell_table.py \
    --run-dir "$RUN"

"$PYTHON" scripts/visualization/plot_spatial_maps.py \
    --run-dir "$RUN"

echo
echo "Finished: $(date)"


# ----------------------------------------------------------------------
# 16. Visium-like observation layer
# ----------------------------------------------------------------------

echo
echo "======================================================================"
echo "[16/16] GENERATE VISIUM-LIKE OBSERVATION"
echo "======================================================================"
echo "Start: $(date)"
echo

if [[ "$VISIUM_ENABLED" == "1" ]]; then

    "$PYTHON" scripts/pipeline/build_visium_spots.py \
        --run-dir "$RUN" \
        --target-median-cells "$VISIUM_TARGET_MEDIAN_CELLS"

    "$PYTHON" scripts/pipeline/generate_visium_expression.py \
        --run-dir "$RUN" \
        --depth-reference-dir "$VISIUM_DEPTH_REFERENCE" \
        --depth-multiplier "$VISIUM_DEPTH_MULTIPLIER"

else

    echo "Visium observation disabled in resolved configuration."

fi

echo
echo "Finished: $(date)"


# ----------------------------------------------------------------------
# Final audit
# ----------------------------------------------------------------------

echo
echo "======================================================================"
echo "FINAL RUN AUDIT"
echo "======================================================================"

"$PYTHON" - "$RUN" <<'PY'
from pathlib import Path
import gzip
import sys

import numpy as np
import pandas as pd
from scipy import sparse
import yaml


run = Path(sys.argv[1])

resolved = yaml.safe_load(
    (run / "resolved_config.yaml").read_text()
)

cells = pd.read_csv(
    run / "cells/cells_with_cell_types.tsv",
    sep="\t",
)

spatial = pd.read_csv(
    run / "spatial/cells_spatial.tsv",
    sep="\t",
)


# ------------------------------------------------------------------
# Latent-cell audit
# ------------------------------------------------------------------

expected_total = int(
    resolved["simulation"]["n_cells"]
)

expected_malignant = int(
    resolved["tissue"]["n_malignant"]
)

expected_nonmalignant = int(
    resolved["tissue"]["n_nonmalignant"]
)

actual_total = len(cells)

actual_malignant = int(
    cells["is_malignant"].sum()
)

actual_nonmalignant = (
    actual_total
    - actual_malignant
)


print("LATENT TISSUE")
print("-------------")

print(
    "Expected total:",
    expected_total,
)

print(
    "Actual total:  ",
    actual_total,
)

print()

print(
    "Expected malignant:",
    expected_malignant,
)

print(
    "Actual malignant:  ",
    actual_malignant,
)

print()

print(
    "Expected nonmalignant:",
    expected_nonmalignant,
)

print(
    "Actual nonmalignant:  ",
    actual_nonmalignant,
)

print()

print("Cell types:")

print(
    cells["cell_type"]
    .value_counts()
    .to_string()
)

print()

print(
    "Observed malignant clones:",
    cells.loc[
        cells["is_malignant"],
        "subclone_id",
    ].nunique(),
)


assert actual_total == expected_total

assert (
    actual_malignant
    == expected_malignant
)

assert (
    actual_nonmalignant
    == expected_nonmalignant
)

assert len(spatial) == expected_total


# ------------------------------------------------------------------
# Visium audit
# ------------------------------------------------------------------

observation = resolved.get(
    "observation",
    {},
)

visium_cfg = observation.get(
    "visium",
    {},
)

visium_enabled = bool(
    visium_cfg.get(
        "enabled",
        True,
    )
)


if visium_enabled:

    print()
    print("VISIUM-LIKE OBSERVATION")
    print("-----------------------")

    visium_dir = (
        run
        / "observation"
        / "visium"
    )

    geometry_summary_path = (
        visium_dir
        / "geometry_summary.yaml"
    )

    expression_summary_path = (
        visium_dir
        / "expression"
        / "summary.yaml"
    )

    spot_observation_path = (
        visium_dir
        / "expression"
        / "spot_observation.tsv"
    )

    count_matrix_path = (
        visium_dir
        / "expression"
        / "counts.npz"
    )

    matrix_market_path = (
        visium_dir
        / "filtered_feature_bc_matrix"
        / "matrix.mtx.gz"
    )

    features_path = (
        visium_dir
        / "filtered_feature_bc_matrix"
        / "features.tsv.gz"
    )

    barcodes_path = (
        visium_dir
        / "filtered_feature_bc_matrix"
        / "barcodes.tsv.gz"
    )


    required = [
        geometry_summary_path,
        expression_summary_path,
        spot_observation_path,
        count_matrix_path,
        matrix_market_path,
        features_path,
        barcodes_path,
    ]

    for path in required:
        assert path.exists(), (
            f"Missing Visium output: {path}"
        )


    geometry = yaml.safe_load(
        geometry_summary_path.read_text()
    )

    expression = yaml.safe_load(
        expression_summary_path.read_text()
    )

    spot_obs = pd.read_csv(
        spot_observation_path,
        sep="\t",
    )

    counts = sparse.load_npz(
        count_matrix_path
    ).tocsr()


    n_spots = int(
        geometry["n_occupied_spots"]
    )

    target_median = float(
        geometry[
            "target_median_cells_per_spot"
        ]
    )

    observed_median = float(
        geometry[
            "observed_median_cells_per_spot"
        ]
    )


    print(
        "Occupied spots:",
        n_spots,
    )

    print(
        "Captured cells:",
        geometry[
            "n_captured_cells"
        ],
    )

    print(
        "Captured-cell fraction:",
        f"{geometry['captured_cell_fraction']:.4f}",
    )

    print(
        "Target median cells/spot:",
        target_median,
    )

    print(
        "Observed median cells/spot:",
        observed_median,
    )

    print(
        "Median UMI depth:",
        expression[
            "simulated_median_umis"
        ],
    )

    print(
        "Mean UMI depth:",
        expression[
            "simulated_mean_umis"
        ],
    )

    print(
        "Median genes detected:",
        expression[
            "median_genes_detected"
        ],
    )

    print(
        "Spot x gene shape:",
        counts.shape,
    )


    assert n_spots > 0

    assert len(spot_obs) == n_spots

    assert counts.shape[0] == n_spots

    assert counts.shape[1] == int(
        expression["n_genes"]
    )

    assert np.all(
        counts.data >= 0
    )

    assert np.issubdtype(
        counts.data.dtype,
        np.integer,
    )


    observed_totals = np.asarray(
        counts.sum(axis=1)
    ).ravel().astype(np.int64)

    target_totals = spot_obs[
        "target_umis"
    ].to_numpy(
        dtype=np.int64
    )

    assert np.array_equal(
        observed_totals,
        target_totals,
    ), (
        "One or more Visium spots do not "
        "sum to sampled target UMI depth."
    )


    assert np.array_equal(
        spot_obs[
            "observed_umis"
        ].to_numpy(
            dtype=np.int64
        ),
        target_totals,
    )


    # The scale-search stage is discrete, so equality is expected
    # for the current preset and is explicitly audited.
    assert (
        observed_median
        == target_median
    ), (
        "Observed median cells/spot does "
        "not match configured target."
    )


    # 10x-style dimensions.
    with gzip.open(
        barcodes_path,
        "rt",
    ) as handle:
        n_barcodes = sum(
            1 for _ in handle
        )

    with gzip.open(
        features_path,
        "rt",
    ) as handle:
        n_features = sum(
            1 for _ in handle
        )

    assert n_barcodes == n_spots

    assert n_features == counts.shape[1]


print()
print(
    "FINAL CONSISTENCY CHECK: PASS"
)
PY


echo
echo "======================================================================"
echo "CANCER ST SIMULATION COMPLETE"
echo "======================================================================"
echo "End time: $(date)"
echo
echo "Run directory:"
echo "$RUN"

echo
echo "Latent spatial plots:"
ls -lh "$RUN/spatial/plots/"

if [[ "$VISIUM_ENABLED" == "1" ]]; then
    echo
    echo "Visium plots:"
    ls -lh "$RUN/observation/visium/plots/"

    echo
    echo "Visium 10x-style matrix:"
    ls -lh \
        "$RUN/observation/visium/filtered_feature_bc_matrix/"
fi

echo
echo "======================================================================"
