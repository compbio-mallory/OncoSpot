# OncoSpot

**OncoSpot: A cancer spatial transcriptomics simulator**

OncoSpot is a computational framework for generating synthetic spatial transcriptomics (ST) datasets with known tumor evolution, copy-number alterations (CNAs), malignant transcriptional states, nonmalignant microenvironment populations, and spatial organization.

The framework integrates DEMON-based tumor evolution, CNAsim-based copy-number simulation, reference-calibrated expression models, and Visium-like spot-level observation.

OncoSpot generates cell-level and spot-level ground truth for benchmarking computational methods in cancer spatial transcriptomics.

## 1. Features

- Spatially organized malignant and nonmalignant populations
- Cancer-specific tumor growth using DEMON
- Clone-specific CNA simulation using CNAsim
- Malignant transcriptional state heterogeneity
- CNA-associated malignant gene expression
- Nonmalignant cell-type composition and spatial organization
- Reference-calibrated gene-expression models
- Visium-like spot placement and UMI count generation
- Cell-level and spot-level simulation ground truth
- Configurable cell count, tumor fraction, CNA burden, and random seed
- Automatic selection of cancer-specific DEMON growth modes

## 2. Supported Cancer Types

The current implementation provides presets for three cancer types:

| Cancer | Code | DEMON Growth Mode |
|---|---|---|
| High-grade serous ovarian carcinoma | HGSOC | Invasive glandular |
| Colorectal cancer | CRC | Gland fission |
| Hepatocellular carcinoma | HCC | Boundary growth |

OncoSpot automatically selects the appropriate DEMON growth mode and associated parameter presets based on the selected cancer.

Users do not need to manually configure or execute DEMON for each simulation.

Additional cancer types require cancer-specific reference calibration and validated parameter presets.

## 3. Simulation Workflow

OncoSpot integrates tumor evolution, genomic alterations, spatial organization, and gene expression into a single simulation pipeline.

The main workflow consists of the following stages:

1. **Configuration:** Load and resolve cancer-specific simulation parameters.
2. **Tumor evolution:** Simulate spatial tumor growth using DEMON.
3. **Cell scaffold:** Construct the malignant and nonmalignant cellular populations.
4. **Spatial organization:** Assign tissue compartments and spatial cell locations.
5. **Malignant lineage:** Construct the tumor lineage for CNA simulation.
6. **CNA simulation:** Generate clone-specific copy-number alterations using CNAsim.
7. **Gene-level CNA mapping:** Map copy-number alterations to genomic regions and genes.
8. **Clonal refinement:** Merge rare malignant subclones according to the configured merging procedure.
9. **Transcriptional states:** Assign malignant transcriptional programs.
10. **Gene expression:** Generate malignant and nonmalignant expression counts.
11. **Spatial integration:** Assemble the final spatial cell-level dataset.
12. **Visium observation:** Generate Visium-like spots and aggregate cell-level expression.
13. **Ground truth:** Export the true malignant identities, clone structure, CNA profiles, and cellular composition.

DEMON and CNAsim are called automatically by the OncoSpot pipeline once installed.

## 4. Installation

### 4.1 System Requirements

The current implementation is designed for Linux.

Recommended requirements:

- Linux operating system
- Python 3.10
- Git
- C++ compiler (GCC/G++)
- Boost C++ headers
- GNUplot (DEMON dependency)
- The `file` utility
- Sufficient memory and disk storage for simulation outputs

Large simulations are recommended to run on an HPC cluster or a sufficiently provisioned workstation.

### 4.2 Clone OncoSpot

```bash
git clone https://github.com/compbio-mallory/OncoSpot.git
cd OncoSpot
```

### 4.3 Create Python Environment

Create a dedicated environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The main Python dependencies include NumPy, pandas, SciPy, PyYAML, and Matplotlib.

### 4.4 Install CNAsim

OncoSpot uses CNAsim to simulate clone-specific copy-number profiles.

Install CNAsim version 1.3.5:

```bash
python -m pip install cnasim==1.3.5
```

Verify the installation:

```bash
command -v cnasim
```

CNAsim source repository:

https://github.com/samsonweiner/CNAsim

### 4.5 Install DEMON

OncoSpot uses DEMON (Deme-based Oncology Model) to simulate spatial tumor evolution.

DEMON is developed independently and must be installed separately.

Source repository:

https://github.com/robjohnnoble/demon_model

Clone the DEMON source:

```bash
mkdir -p external

git clone \
  https://github.com/robjohnnoble/demon_model.git \
  external/demon_reference
```

Compile the Linux executable:

```bash
mkdir -p external/demon_reference/bin

g++ -O2 \
  -o external/demon_reference/bin/demon \
  external/demon_reference/src/demon.cpp \
  -lm
```

The required Boost development headers must be installed and discoverable by the compiler.

If Boost is installed in a nonstandard location, add its include path using `-I`.

Verify the executable:

```bash
file external/demon_reference/bin/demon
```

The output should identify a Linux ELF executable.

### 4.6 Configure External Executables

From the OncoSpot root directory:

```bash
export ONCOSPOT_PYTHON="$(command -v python)"

export ONCOSPOT_CNASIM="$(command -v cnasim)"

export ONCOSPOT_DEMON_BIN="$(pwd)/external/demon_reference/bin/demon"
```

These variables may instead point to existing installations.

For HPC environments, the executable paths can be set in a SLURM submission script.

### 4.7 Reference Parameter Files

OncoSpot requires calibrated parameters for the selected cancer.

Parameter files are organized under:

```text
parameters/
├── HGSOC/
│   ├── malignant/
│   ├── nonmalignant/
│   ├── tumor_states/
│   └── spatial/
├── CRC/
│   ├── malignant/
│   ├── nonmalignant/
│   ├── tumor_states/
│   └── spatial/
├── HCC/
│   ├── malignant/
│   ├── nonmalignant/
│   ├── tumor_states/
│   └── spatial/
└── observation/
    └── visium/
```

The parameter bundles contain:

- Cell-of-origin expression baselines
- Malignant expression parameters
- Nonmalignant expression baselines
- Cell-type-specific library-size distributions
- Gene-expression dispersion parameters
- Malignant transcriptional state references
- Spatial compartment parameters
- DEMON growth presets
- CNAsim parameters
- Visium UMI-depth reference distributions

**Important:** The initial source-code repository does not contain all reference-derived parameter TSV files.

A compatible parameter bundle must be supplied before running an end-to-end simulation.

Parameter distribution instructions will be updated as the reference bundles are finalized for release.

### 4.8 GENCODE Gene Annotation

OncoSpot uses GENCODE gene annotations to map simulated copy-number alterations to genes.

The current implementation uses GENCODE release 50 (GRCh38).

Download the corresponding GENCODE GTF from:

https://www.gencodegenes.org/human/

Then prepare the autosomal gene annotation:

```bash
python scripts/setup/prepare_gencode_genes.py \
  --input /path/to/gencode.v50.annotation.gtf.gz \
  --output data/annotations/gencode/v50/processed/genes_autosomes.tsv \
  --summary data/annotations/gencode/v50/processed/summary.json
```

The resulting annotation is used by the CNA-to-gene mapping stage.

## 5. Running OncoSpot

### 5.1 Quick Start: Select a Cancer

OncoSpot provides a cancer-selection wrapper.

For HGSOC:

```bash
python scripts/run_oncospot.py --cancer HGSOC
```

For colorectal cancer:

```bash
python scripts/run_oncospot.py --cancer CRC
```

For hepatocellular carcinoma:

```bash
python scripts/run_oncospot.py --cancer HCC
```

The wrapper automatically selects the corresponding cancer configuration and DEMON growth mode.

Users do not need to execute DEMON or CNAsim manually.

The wrapper calls the existing end-to-end simulation pipeline internally.

### 5.2 Customize Simulation Parameters

Users can configure the total number of cells, malignant-cell fraction, and random seed.

Example:

```bash
python scripts/run_oncospot.py \
  --cancer HGSOC \
  --cells 15000 \
  --tumor-fraction 0.5 \
  --seed 42
```

Another example:

```bash
python scripts/run_oncospot.py \
  --cancer CRC \
  --cells 10000 \
  --tumor-fraction 0.3 \
  --seed 123
```

The tumor fraction is the requested proportion of malignant cells. The realized proportion can differ because of the stochastic simulation and spatial construction procedures.

### 5.3 CNA Burden

The wrapper accepts CNA-burden settings:

```bash
python scripts/run_oncospot.py \
  --cancer HGSOC \
  --cna-burden default
```

The available options are:

- `low`
- `default`
- `high`

The influence of these settings should be verified against the resolved CNAsim configuration before using them in comparative experiments.

### 5.4 Configuration Dry Run

To inspect the selected cancer configuration and DEMON mode without starting a simulation:

```bash
python scripts/run_oncospot.py \
  --cancer HGSOC \
  --dry-run
```

### 5.5 Advanced: Run with a YAML Configuration

Advanced users can directly execute the main simulation pipeline:

```bash
bash scripts/pipeline/run_cancer_simulation.sh \
  configs/examples/hgsoc_validation_15k_hallmark_v1.yaml \
  results/runs/hgsoc_example
```

This permits detailed customization through YAML configuration files.

The output directory must be new and located inside `results/runs/`.

Cancer-specific examples are available in:

```text
configs/examples/
```

## 6. Simulation Outputs

Each simulation creates a dedicated run directory under:

```text
results/runs/
```

Depending on the configuration, the output contains:

### 6.1 Tumor Evolution

- DEMON simulation outputs
- Tumor lineage information
- Malignant clone identities
- Spatial cell organization

### 6.2 Copy-Number Alterations

- Clone-specific CNA profiles
- Gene-level copy-number states
- CNA simulation metadata
- Malignant clone assignments

### 6.3 Gene Expression

- Malignant cell expression counts
- Nonmalignant cell expression counts
- Cell-type annotations
- Malignant transcriptional states
- Cell-level expression data

### 6.4 Spatial Transcriptomics

- Cell spatial coordinates
- Visium-like spot coordinates
- Cell-to-spot assignments
- Spot-level gene-expression counts
- Malignant fractions in individual spots
- Spot-level cellular composition

### 6.5 Ground Truth

OncoSpot preserves ground-truth information for evaluating downstream computational methods, including:

- Malignant and nonmalignant identities
- Malignant subclone labels
- True CNA states
- Cell-type composition
- Spatial organization

### 6.6 10x-Style Output

Visium-like expression matrices are exported in Matrix Market format, including:

```text
matrix.mtx.gz
barcodes.tsv.gz
features.tsv.gz
```

These files can be imported into standard single-cell and spatial transcriptomics analysis workflows.

## 7. Reference-Based Parameter Estimation

OncoSpot uses reference-derived parameters to reproduce cancer-specific transcriptional characteristics.

The current implementation includes calibrated parameter sets for HGSOC, CRC, and HCC.

Reference calibration includes estimation of:

- Cell-of-origin gene-expression baselines
- Malignant gene-expression distributions
- CNA-associated expression responses
- Malignant transcriptional state effects
- Nonmalignant cell-type expression distributions
- Library-size distributions
- Dispersion parameters
- Spatial and tissue organization presets

The current cancer-specific calibration procedures are based on empirical reference datasets.

A unified parameter-estimation interface for new user-provided scRNA-seq references is under development.

For new cancer types, users will need suitable reference data and compatible cancer-specific parameter presets.

Tumor-growth modes and CNA process parameters may require additional biological assumptions or calibration beyond scRNA-seq expression alone.

## 8. Reproducibility

OncoSpot supports reproducible simulation through explicit configuration files and random seeds.

Each run records its resolved configuration.

For reproducibility, users should preserve:

- Simulation configuration
- Random seed
- Cancer-specific parameter bundle
- Reference gene annotation
- DEMON and CNAsim versions
- Python dependency versions

The HGSOC end-to-end workflow has been tested using a 15,000-cell simulation.

In that run, OncoSpot generated:

| Metric | Value |
|---|---:|
| Simulated cells | 15,000 |
| Observed malignant clones | 8 |
| Visium-like spots | 1,461 |
| Genes | 24,640 |
| Median spot UMI depth | 6,125 |
| Final consistency check | PASS |

These results represent one completed simulation run rather than a benchmark of all supported parameter combinations.

## 9. Repository Structure

```text
OncoSpot/
├── README.md
├── requirements.txt
├── configs/
│   ├── defaults/
│   ├── cancers/
│   ├── examples/
│   └── schemas/
├── parameters/
│   ├── HGSOC/
│   ├── CRC/
│   ├── HCC/
│   └── observation/
├── scripts/
│   ├── run_oncospot.py
│   ├── pipeline/
│   ├── setup/
│   └── visualization/
├── src/
│   └── cancer_st_sim/
│       ├── annotation/
│       ├── cna/
│       ├── config/
│       ├── core/
│       ├── expression/
│       ├── io/
│       ├── observation/
│       ├── spatial/
│       └── tumor/
└── tests/
```

Generated simulation outputs, raw datasets, local software environments, and experimental validation artifacts are excluded from version control.

## 10. External Software and Attribution

OncoSpot integrates the following independently developed software.

### DEMON

Deme-based Oncology Model, used for spatial tumor evolution.

Repository:
https://github.com/robjohnnoble/demon_model

### CNAsim

Used for simulation of clone-specific copy-number alterations.

Repository:
https://github.com/samsonweiner/CNAsim

Users should cite the original publications describing these tools when using OncoSpot in research.

The third-party software retains its respective licensing terms.

## 11. Citation

If you use OncoSpot in your research, please cite the OncoSpot manuscript once the citation information is available.

Manuscript details will be updated following publication.

## 12. License

Third-party software, including DEMON and CNAsim, remains subject to its original license.
