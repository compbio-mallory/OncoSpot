# Extending OncoSpot to a New Cancer Type

OncoSpot currently provides calibrated simulation presets for:

- High-grade serous ovarian carcinoma (HGSOC)
- Colorectal cancer (CRC)
- Hepatocellular carcinoma (HCC)

This guide describes how to extend OncoSpot to a cancer type
that is not included in the existing presets.

## 1. Overview

OncoSpot separates its simulation into:

1. Spatial tumor evolution
2. Malignant lineage and copy-number alteration simulation
3. Malignant transcriptional states and gene expression
4. Nonmalignant cell types and tissue organization
5. Spatial transcriptomics observation

For an additional cancer type, the simulation framework can
be reused, but cancer-specific parameters must be supplied
or estimated from appropriate reference data.


## 2. Required Reference Data

Recommended inputs include:

### Malignant scRNA-seq reference

A gene-by-cell or cell-by-gene raw count matrix containing
malignant cells from the target cancer type.

Recommended accompanying information:

- Gene identifiers
- Cell identifiers
- Malignant cell annotations
- Sample or patient identifiers
- Copy-number information, if available
- Transcriptional state annotations, if available

### Nonmalignant scRNA-seq reference

A reference containing relevant nonmalignant populations,
such as:

- T cells
- B/plasma cells
- Myeloid/macrophage populations
- Fibroblasts
- Endothelial cells
- Normal epithelial cells

The relevant populations depend on the tissue of origin.

### Cell-of-origin reference

A normal-cell reference representing the suspected tissue
or cell of origin of the target malignancy.

### Spatial reference (recommended)

Spatial transcriptomics or histological data can help
inform tissue geometry, spatial organization, stromal
distribution, and tumor growth assumptions.

Expression data alone cannot uniquely determine
the underlying spatial growth mechanism.

## 3. Parameter Estimation

The new cancer parameter bundle should contain the
following components.

### 3.1 Cell-of-origin expression baseline

Estimate baseline expression from the selected normal
cell-of-origin reference.

Match gene identifiers to the annotation used by OncoSpot.

### 3.2 Malignant expression model

Estimate cancer-specific malignant expression parameters,
including baseline expression and expression variability.

OncoSpot currently uses a negative binomial expression
framework with CNA-associated expression effects and
malignant transcriptional states.

CNA-response coefficients require suitable evidence or
explicit assumptions. They cannot generally be identified
from expression counts alone.

### 3.3 Malignant transcriptional states

The current simulator uses six shared transcriptional axes:

- PROLIFERATION
- WNT
- MYC
- HYPOXIA
- EMT
- METABOLIC

Estimate cancer-specific state distributions and expression
effects while retaining these six axes for compatibility
with the current implementation.

### 3.4 Nonmalignant expression model

Estimate cell-type-specific expression baselines,
dispersion parameters, and empirical library-size
distributions from annotated nonmalignant references.

### 3.5 Tissue organization

Choose or calibrate:

- Malignant and nonmalignant compartment structure
- Spatial cell-type distributions
- Tumor interface behavior
- Cell proportions

These settings can be informed by spatial references and
histological information.

### 3.6 DEMON growth mode

DEMON models spatial tumor evolution.

The growth mode must be chosen based on the biological
characteristics of the cancer and its tissue architecture.

Existing OncoSpot examples include:

| Preset | DEMON mode         |
| ------ | ------------------ |
| HGSOC  | invasive_glandular |
| CRC    | gland_fission      |
| HCC    | boundary_growth    |

These are examples, not universal assignments.

For a new cancer, select and validate a suitable DEMON
mode using available biological evidence.

The selected mode must be supported by the installed
DEMON integration.

### 3.7 CNA simulation

Define the CNA process, including its burden and other
model assumptions.

The current implementation uses CNAsim.

Cancer-specific CNA parameters should be derived from
appropriate genomic references where possible.

## 4. Parameter Bundle Structure

The required organization follows the existing cancer
bundles under `parameters/`.

For a new cancer named CUSTOM:

parameters/CUSTOM/
    malignant/
        cell_of_origin/
        expression/
        demon_preset.yaml
    nonmalignant/
        expression/
    spatial/
    tumor_states/

Typical files include:

- Cell-of-origin expression baseline
- Malignant expression parameter table
- Malignant library-size reference
- Nonmalignant expression baseline table
- Nonmalignant library-size reference
- Nonmalignant dispersion metadata
- Malignant state reference
- DEMON parameter preset
- Spatial compartment preset

Existing HGSOC, CRC, and HCC parameter directories provide
examples of the expected organization and metadata.

The input schemas and resulting parameter tables must be
compatible with the corresponding OncoSpot pipeline stages.

## 5. Register a New Cancer

The configuration resolver locates cancer presets at:

    configs/cancers/<CANCER></cancer>.yaml

For a new cancer, create a configuration with the same
structure as an existing preset, but replace every
cancer-specific reference and assumption.

For example:

    configs/cancers/CUSTOM.yaml

At minimum, review:

- cancer_type
- simulation settings
- tissue tumor fraction
- cell of origin
- malignant transcriptional states
- DEMON growth mode and parameter file
- CNAsim parameter file
- spatial microenvironment configuration
- malignant expression parameter files
- nonmalignant expression parameter files
- tumor state reference files
- observation settings

Do not reuse another cancer's calibrated expression
parameters without documenting and validating that
transfer.

## 6. Run the New Cancer

After preparing the parameter bundle and registering
the cancer configuration, create a user configuration
such as:

    configs/examples/custom_run.yaml

Containing:

    cancer_type: CUSTOM
    simulation:
      n_cells: 15000
      seed: 42
    tissue:
      tumor_fraction: 0.5

Then run the main pipeline from the repository root:

    bash scripts/pipeline/run_cancer_simulation.sh
      configs/examples/custom_run.yaml
      results/runs/custom_test

## 7. Validate the New Cancer

Before using the new preset for research, verify:

- Configuration resolves successfully
- Every required parameter file exists
- DEMON executes with the selected mode
- CNAsim produces compatible CNA outputs
- Gene identifiers agree across reference tables
- Expression generation completes
- Spatial cell and spot outputs are consistent
- Simulated expression and spatial distributions are
  biologically plausible for the target cancer

Compare relevant simulated statistics against real
reference datasets where available.

## DEMON Growth Mode Selection

For cancer-specific growth-mode recommendations and
architecture-dependent exceptions, see the
[DEMON Mode Selection Guide](DEMON_MODE_SELECTION.md).

---

## Implemented custom-cancer workflow

OncoSpot supports reference-based parameter calibration and registration
of additional cancer types.

The workflow consists of four steps:

1. Prepare standardized cell annotations.
2. Estimate expression parameters from scRNA-seq reference counts.
3. Register a custom cancer using a supported DEMON growth mode.
4. Run an OncoSpot simulation.

The generalized workflow was tested end-to-end using colorectal cancer
(CRC, GSE132465), with 15,000 simulated cells.

This verifies software integration but does not establish biological
validity for other cancer types.

### Requirements

Install the Python dependencies listed in the repository's
`requirements.txt`.

DEMON and CNAsim must also be installed for the simulation pipeline.

See the main repository README for installation instructions.

### Required reference inputs

The calibration command currently requires four input files.

**1. Raw UMI count matrix**

A gzip-compressed, tab-separated gene-by-cell count matrix:

- The first column must be named `Index`.
- The first column contains gene symbols.
- All other columns contain reference-cell identifiers.
- Values must be nonnegative integer UMI counts.

Example:

    Index    CELL001    CELL002    CELL003
    A1BG     0          2          1
    A2M      3          0          5

Save the matrix as `raw_counts.tsv.gz`.

**2. Standardized cell annotations**

A tab-separated file containing:

| Column                 | Description                                   |
| ---------------------- | --------------------------------------------- |
| `reference_cell_id`  | Cell ID matching the count matrix             |
| `patient`            | Patient/donor identifier                      |
| `oncospot_cell_type` | Malignant or nonmalignant cell-type label     |
| `is_cell_origin`     | True for selected normal cell-of-origin cells |
| `sample`             | Optional sample identifier                    |

Use `malignant` for tumor cells.

Supported nonmalignant parameter labels in the current spatial
simulation configuration include:

- `T_NK`
- `B_plasma`
- `macrophage_myeloid`
- `fibroblast`
- `endothelial`
- `normal_epithelial`

Cell-of-origin cells must be nonmalignant.

The cell-type labels must match the runtime cell-type mapping in
the selected simulation preset.

**3. Gene mapping**

A tab-separated file containing:

| Column          | Description                       |
| --------------- | --------------------------------- |
| `gene_symbol` | Gene symbol from the count matrix |
| `gene_id`     | Ensembl gene identifier           |
| `status`      | Mapping status                    |

Only rows with `status=unique` are retained.

**4. Hallmark gene sets**

An MSigDB Hallmark GMT file containing the gene sets used by
OncoSpot's six malignant transcriptional state axes.

The six axes are:

- PROLIFERATION
- WNT
- MYC
- HYPOXIA
- EMT
- METABOLIC

### Step 1: Prepare reference annotations

Map the labels in your scRNA-seq dataset to the standardized
OncoSpot annotation columns described above.

For the CRC GSE132465 dataset, a reference adapter is included:

    python custom_reference/adapters/prepare_crc_labels.py
      --annotation /path/to/GSE132465_cell_annotation.txt.gz
      --output /path/to/standardized_cells.tsv

For other datasets, users must provide equivalent annotations.

The CRC adapter is dataset-specific and should not be used as a
universal cell-type classifier.

An AnnData preparation utility is also available:

    python custom_reference/calibration/01_prepare_reference.py --help

This utility validates and partitions AnnData references. However,
the unified parameter estimator currently expects a compressed
gene-by-cell TSV matrix, not a direct `.h5ad` input.

### Step 2: Estimate parameters

Run the following command from the OncoSpot repository root:

    python custom_reference/estimate_parameters.py
      --cancer BRCA
      --counts /path/to/raw_counts.tsv.gz
      --annotations /path/to/standardized_cells.tsv
      --gene-mapping /path/to/gene_mapping.tsv
      --gmt /path/to/hallmark.symbols.gmt
      --output results/calibration/BRCA

This generates:

- Donor-balanced cell-of-origin expression baselines
- Nonmalignant cell-type expression baselines
- Empirical malignant and nonmalignant library sizes
- Nonmalignant NB2 dispersion parameters
- Six empirical malignant Hallmark state scores
- Patient-centered ridge-regression state coefficients
- Malignant gene-level NB2 dispersion parameters

The output is organized into:

    results/calibration/BRCA/
      malignant/
        cell_of_origin/
          origin_baseline.tsv
        expression/
          malignant_expression_parameters.tsv
          malignant_library_sizes.tsv
          expression_metadata.yaml
      nonmalignant/
        expression/
          tme_expression_baselines.tsv
          library_sizes_by_cell_type.tsv
          tme_dispersion.yaml
      tumor_states/
        state_reference.tsv
        state_metadata.yaml

Note: The default malignant CNA dosage coefficient (beta = 0.4)
is provisional. It is not estimated from DNA copy-number data.

### Step 3: Select a DEMON growth mode

Choose a DEMON growth mode based on the spatial growth
architecture of the intended cancer type.

The currently supported modes are:

| DEMON mode             | Base preset |
| ---------------------- | ----------- |
| `invasive_glandular` | HGSOC       |
| `gland_fission`      | CRC         |
| `boundary_growth`    | HCC         |

See `DEMON_MODE_SELECTION.md` for additional guidance.

These mode selections describe broad spatial growth architectures.
They are not automatically inferred from scRNA-seq data.

### Step 4: Register the new cancer

For example, register BRCA using the invasive glandular mode:

    python custom_reference/register_cancer.py
      --cancer BRCA
      --parameters results/calibration/BRCA
      --demon-mode invasive_glandular
      --base-preset HGSOC

This creates:

    configs/cancers/BRCA.yaml
    configs/examples/brca_custom.yaml
    parameters/BRCA/

Registration does not modify the built-in HGSOC, CRC or HCC
configurations.

The selected base preset supplies provisional DEMON, CNA and
spatial microenvironment assumptions.

### Step 5: Verify the configuration

Check the configuration before running a simulation:

    python scripts/run_oncospot.py
      --cancer BRCA
      --cells 15000
      --tumor-fraction 0.5
      --seed 42
      --dry-run

A successful dry run should report the selected cancer type,
cell count and DEMON growth mode.

### Step 6: Generate simulated spatial transcriptomics

Run:

    python scripts/run_oncospot.py
      --cancer BRCA
      --cells 15000
      --tumor-fraction 0.5
      --seed 42
      --output results/runs/brca_custom_15k

The simulator generates:

- DEMON spatial tumor architecture
- Malignant subclone identities and lineage structure
- CNAsim copy-number profiles
- Malignant and nonmalignant single-cell expression
- Spatial cell coordinates and cell-type annotations
- Visium-like spot-level expression matrices
- Ground-truth annotations for downstream benchmarking

The executable paths can be configured using:

    export ONCOSPOT_PYTHON=/path/to/python
    export ONCOSPOT_DEMON_BIN=/path/to/demon
    export ONCOSPOT_CNASIM=/path/to/cnasim

A working DEMON and CNAsim installation is required.

### Limitations

- Reference gene identifiers must be compatible with the
  simulator's gene-level CNA annotation.
- scRNA-seq reference labels should be independently validated.
- The malignant CNA dosage coefficient remains provisional
  unless supported by additional genomic measurements.
- The chosen DEMON architecture may not describe every tumor
  within a cancer type.
- Spatial microenvironment parameters reused from a built-in
  cancer are provisional for newly registered cancer types.
- The current calibration interface accepts compressed
  gene-by-cell TSV matrices.
- Very small simulations may produce a single malignant
  clone, which is incompatible with the current CNAsim
  tree-input interface.
- Newly calibrated cancers require independent biological
  validation before use in scientific benchmarking.

### Data availability and reproducibility

The repository provides reusable calibration and simulation code..

Users must obtain the appropriate reference datasets and
provide the required parameter inputs.

For reproducibility, retain the source dataset identifiers,
gene mapping, Hallmark GMT version, simulation configuration,
random seed and generated parameter metadata.
