# Automated Pipeline for Longitudinal Lesion Tracking in SPECT Imaging using Morphology and Texture Aware Cost Function



The pipeline begins with rigid registration of the two time-point SPECT scans using Mattes mutual information and an Euler3DTransform, ensuring spatial alignment of the images. This transformation is then applied to the lesion masks from the second time-point to align them with the first time point. For each lesion, we calculate key features at both time points, including centroid, volume (voxel count), and heterogeneity metrics derived from first-order statistics and GLCM-based analysis. To quantify lesion dissimilarity, we designed a custom cost function that accounts for spatial distance between lesion centroids (D), overlap of lesion volumes (O), and differences in texture heterogeneity ΔH ( weighting parameters α, β, γ were set to 0.5, 0.4, and 0.1, respectively)

![image](https://github.com/qurit/Lesion_Tracking/blob/main/LT_approach1.png)


### Install  necessary libraries
SimpleITK>=2.3.1
pyradiomics>=3.1.0
numpy>=1.26
pandas>=2.0
scipy>=1.11
scikit-learn>=1.3
matplotlib>=3.8

## Folder layout (per patient)


# To be added
Second approach: graph-based lesion matching
