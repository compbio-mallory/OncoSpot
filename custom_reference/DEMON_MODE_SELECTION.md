# Selecting DEMON Growth Modes for New Cancer Types

OncoSpot selects a spatial tumor growth mode according to
the biological architecture of the target cancer.

For the three currently implemented cancers, OncoSpot uses
their validated configuration presets:

| Cancer | DEMON mode         |
| ------ | ------------------ |
| HGSOC  | invasive_glandular |
| CRC    | gland_fission      |
| HCC    | boundary_growth    |

These assignments are specific to the implemented presets
and should not be interpreted as universal tumor-wide rules.

## Reference-Based Selection Guide

The following guidance summarizes the cancer-category
mapping and literature evidence collected for OncoSpot.

| Cancer / architecture                           | Candidate DEMON mode | Notes                              |
| ----------------------------------------------- | -------------------- | ---------------------------------- |
| Colorectal adenoma / crypt-fission architecture | gland_fission        | Direct crypt-fission evidence      |
| Selected invasive breast or lung carcinoma      | invasive_glandular   | Supported in selected solid tumors |
| Expansile HCC                                   | boundary_growth      | Not universal across HCC           |
| Infiltrative soft-tissue sarcoma                | invasive_glandular   | Architecture-specific analogy      |
| Pushing / expansile sarcoma                     | boundary_growth      | Candidate; lower confidence        |
| Follicular lymphoma                             | invasive_glandular   | Spatial-deme analogy               |
| Multiple myeloma                                | invasive_glandular   | Marrow-niche analogy               |
| Expansile pancreatic NET                        | boundary_growth      | Depends on tumor border            |
| Infiltrative pancreatic NET                     | invasive_glandular   | Depends on tumor border            |
| Leukemia                                        | non_spatial          | Supported by the DEMON framework   |

## Selection Procedure

1. Identify the target cancer and histological subtype.
2. Determine whether the tumor has an infiltrative,
   gland-fission, expansile, or non-spatial growth mechanism.
3. Consult the published evidence for that architecture.
4. Choose a supported DEMON mode.
5. Record the selection rationale and confidence.
6. Calibrate the remaining cancer-specific parameters.
7. Validate simulated spatial and evolutionary statistics.

## Important Limitations

Spatial heterogeneity alone does not establish the tumor
growth mechanism.

Diversity, driver burden, and phylogenetic patterns can
support model validation, but should not automatically
determine the DEMON spatial mode.

Mode selection does not replace cancer-specific expression,
CNA, or tumor microenvironment calibration.

Some entries represent mechanistic analogies rather than
direct validations of the DEMON model.

The mapping is a decision guide, not a universal
cancer-type classifier.

## Key Reference

Noble et al. Spatial structure governs the mode of tumour
evolution. Nature Ecology & Evolution (2022).

https://doi.org/10.1038/s41559-021-01615-9
