# Automated Longitudinal Lesion Tracking in Quantitative SPECT/CT for Radiopharmaceutical Therapy

Automated, topology-aware lesion tracking framework for longitudinal SPECT/CT in radiopharmaceutical therapy (RPT), enabling scalable lesion-level dosimetry and response assessment. Code accompanying the manuscript submitted to the *Journal of Nuclear Medicine (JNM)*.

## Overview

Lesion-level dosimetry and response assessment in RPT (e.g., [¹⁷⁷Lu]Lu-PSMA-617, [¹⁷⁷Lu]Lu-HTK03170, [¹⁷⁷Lu]Lu-DOTATATE) require reliable correspondence of individual lesions across serial SPECT/CT scans — within a treatment cycle (intra-cycle, for time-activity curve fitting) and across cycles (inter-cycle, for cumulative dose and longitudinal response). Manual correspondence is impractical at scale, particularly in high metastatic burden disease such as mCRPC.

This repository implements an automated lesion tracking pipeline validated on 193 SPECT scans from 34 patients across three cohorts (mCRPC, NETs), comprising 3,006 annotated lesion correspondences.

## Method

The pipeline processes longitudinal SPECT/CT acquisitions organized by cycle and timepoint (e.g., `c1t1`, `c1t2`, `c2t1`) through five stages:

1. **Lesion mask extraction** from RTSTRUCT DICOM (with NIfTI fallback)
2. **Registration** — CT→CT rigid + affine alignment (Mattes mutual information) to a cohort-specific anchor timepoint, applied to SPECT and lesion masks; anatomy-driven CT registration is more robust than SPECT→SPECT when uptake changes strongly across cycles
3. **Feature extraction** per lesion — centroid, volume, radiomic heterogeneity (PyRadiomics first-order + GLCM entropy)
4. **Correspondence** — spatially gated, cost-based assignment (Hungarian algorithm) over standardized feature vectors, run across all consecutive intra- and inter-cycle pairs plus dedicated missed/new-lesion reference pairs
5. **Canonical lesion identification** — global lesion IDs propagated across cycles, RTSTRUCT renaming, and longitudinal PDF report generation

Tracking performance was evaluated using Connected Lesion Accuracy (CLA), precision, and recall against physician-reviewed reference correspondences.

## Features

- DICOM-native I/O (CT, SPECT/NM, RTSTRUCT) with per-timepoint caching
- Geometry-safe mask alignment (header copy vs. nearest-neighbor resample)
- Cohort auto-detection (HTK / PR21 / CAVA) with cohort-specific anchor timepoint logic
- Cost-colored coronal MIP visualizations with explicit missed/new lesion markers
- Registration QC (Pearson correlation, MAD) with automatic flagging
- Deterministic seeds, notebook- and CLI-friendly

## Installation

```bash
pip install SimpleITK rt-utils pydicom pyradiomics numpy pandas scipy scikit-learn matplotlib
```

## Usage

Set `PATIENT_ID` and `BASE_ROOT` at the top of the pipeline cell/script, pointing to a directory of timepoint subfolders (`c1t1`, `c1t2`, ...) containing CT, SPECT, and RTSTRUCT DICOM series. Run the pipeline to generate:

- Registered SPECT/CT volumes and aligned lesion masks
- `matched_lesions_<t1>_to_<t2>.csv` — lesion correspondence tables per timepoint pair
- Coronal MIP tracking visualizations (`lesion_tracking_<t1>_to_<t2>.png`)
- Registration QC plots

## Citation

If you use this pipeline, please cite:

> Yousefirizi F, Esquinas PL, Kurkowska S, Colpo N, Williams C, Hou X, Soleimani M, Wilson D, Beauregard J-M, Bénard F, Rahmim A, Uribe C. Automated Longitudinal Lesion Tracking in Quantitative SPECT/CT for Radiopharmaceutical Therapy. *Journal of Nuclear Medicine* (submitted).

## Contact

Corresponding author: Carlos Uribe, PhD, MCCPM — carlos.uribe@bccancer.bc.ca
BC Cancer / University of British Columbia, Vancouver, BC, Canada
