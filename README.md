# Automated Pipeline for Longitudinal Lesion Tracking in SPECT Imaging using Morphology and Texture Aware Cost Function


# Longitudinal Lesion Tracking (SPECT/CT)

Robust **longitudinal lesion tracking** for nuclear medicine imaging (SPECT/CT):

- **Rigid registration `T2 → T1`** (SimpleITK) with optional histogram matching and foreground body masks  
- **Geometry-safe mask alignment** (header fix vs. NN resample) to prevent silent mis-registrations  
- **Feature construction**: centroids (mm), volumes (mm³), simple heterogeneity (PyRadiomics first-order + GLCM)  
- **Partial Hungarian matching** with **spatial gating** + **cosine feature similarity**  
- **Visualization**: joint-window coronal MIPs, links colored by match strength, explicit missed/new lesions  
- **QC**: blended MIPs with Pearson correlation and MAD metrics  
- **CLI & notebook-friendly**: clean logging, deterministic seeds, clear outputs

> **Use case**: Longitudinal SPECT/CT (e.g., RPT monitoring) where consistent lesion correspondence is needed to form lesion-level TACs, estimate TIA, and enable dose–response modeling.


![image](https://github.com/qurit/Lesion_Tracking/blob/main/LT_approach1.png)

---



### Install  necessary libraries
SimpleITK>=2.3.1
pyradiomics>=3.1.0
numpy>=1.26
pandas>=2.0
scipy>=1.11
scikit-learn>=1.3
matplotlib>=3.8

