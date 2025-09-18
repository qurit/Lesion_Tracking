# SPDX-License-Identifier: MIT
"""
Longitudinal Lesion Tracking (SPECT/CT)

Rigid registration (T2 → T1), mask–image geometry reconciliation,
feature-based partial matching with spatial gating, and QC/visualization.

Author: Fereshteh Yousefirizi
License: MIT
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from radiomics import featureextractor
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------
# Logging
# -------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("lesion_tracking")

# Quiet PyRadiomics logs
import os as _os

_os.environ["PYRAD_LOGLEVEL"] = "WARNING"
logging.getLogger("radiomics").setLevel(logging.ERROR)


# -------------------------
# Configuration
# -------------------------
@dataclass
class Config:
    # I/O
    patient_folder: Path = Path("05")
    time1: str = "2024-06-29"  # fixed (T1)
    time2: str = "2024-10-05"  # moving (T2)
    spect_name: str = "spect.nii.gz"  # change to 'SPECT.nii.gz' if needed
    spect_subdir: str = "spect"       # '' if directly under time folder
    mask_subdir: str = "lesions"

    # Registration
    use_histogram_matching: bool = True
    fixed_mask_pct: float = 55.0  # body mask lower percentile on T1
    moving_mask_pct: float = 55.0  # body mask lower percentile on T2
    resample_t2_masks: bool = True  # resample T2 masks → T1 (recommended)

    # Matching
    spatial_gate_mm: float = 40.0
    spatial_scale_mm: float = 60.0
    cost_weight_spatial: float = 0.6  # 0..1

    # MIP and output
    mip_percentiles: Tuple[float, float] = (1.0, 99.0)
    output_tag: str = "rigidT2toT1"
    random_seed: int = 13

    def t1_dir(self) -> Path:
        return self.patient_folder / self.time1

    def t2_dir(self) -> Path:
        return self.patient_folder / self.time2

    def spect1_path(self) -> Path:
        return (self.t1_dir() / self.spect_subdir / self.spect_name) if self.spect_subdir else (self.t1_dir() / self.spect_name)

    def spect2_path(self) -> Path:
        return (self.t2_dir() / self.spect_subdir / self.spect_name) if self.spect_subdir else (self.t2_dir() / self.spect_name)

    def mask1_dir(self) -> Path:
        return self.t1_dir() / self.mask_subdir

    def mask2_dir(self) -> Path:
        return self.t2_dir() / self.mask_subdir


# -------------------------
# Utility helpers
# -------------------------
def _nearly_equal(a: Sequence[float], b: Sequence[float], atol: float = 1e-5) -> bool:
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=atol)


def _image_geom(img: sitk.Image) -> Tuple[Tuple[int, int, int], Tuple[float, float, float], Tuple[float, float, float], Tuple[float, ...]]:
    return img.GetSize(), img.GetSpacing(), img.GetOrigin(), img.GetDirection()


def _list_nii(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix in {".nii", ".gz"} or p.name.endswith(".nii.gz")])


def _print_geom(tag: str, img: sitk.Image) -> None:
    sz, sp, org, direc = _image_geom(img)
    logger.info(f"{tag}: size={sz}, spacing={np.round(sp, 3)}, origin={np.round(org, 2)}, dir={np.round(direc, 3)}")


def _ensure_exists(path: Path, kind: str = "path") -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected {kind} not found: {path}")


def print_transform_info(T: sitk.Transform) -> None:
    """Print transform(s) robustly (works for Composite and Euler)."""
    try:
        logger.info("Transform name: %s", T.GetName())
        if isinstance(T, sitk.CompositeTransform):
            n = T.GetNumberOfTransforms()
            logger.info(" Composite with %d sub-transforms:", n)
            for i in range(n):
                Ti = T.GetNthTransform(i)
                params = np.round(np.array(Ti.GetParameters(), float), 4)
                logger.info("  [%d] %s params: %s", i, Ti.GetName(), params)
                if Ti.GetName() == "Euler3DTransform":
                    center = sitk.Euler3DTransform(Ti).GetCenter()
                    logger.info("      center: %s", np.round(np.array(center, float), 4))
        else:
            params = np.round(np.array(T.GetParameters(), float), 4)
            logger.info(" Parameters: %s", params)
            if T.GetName() == "Euler3DTransform":
                center = sitk.Euler3DTransform(T).GetCenter()
                logger.info(" Center: %s", np.round(np.array(center, float), 4))
    except Exception as e:
        logger.warning("Could not fully describe transform: %s", e)


# -------------------------
# Mask alignment to own SPECT
# -------------------------
def copy_geometry(m: sitk.Image, r: sitk.Image) -> sitk.Image:
    out = sitk.Image(m)
    out.SetOrigin(r.GetOrigin())
    out.SetSpacing(r.GetSpacing())
    out.SetDirection(r.GetDirection())
    return out


def resample_nn(m: sitk.Image, r: sitk.Image, tx: Optional[sitk.Transform] = None) -> sitk.Image:
    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(r)
    rf.SetInterpolator(sitk.sitkNearestNeighbor)
    rf.SetDefaultPixelValue(0)
    rf.SetTransform(tx if tx is not None else sitk.Transform(3, sitk.sitkIdentity))
    return rf.Execute(m)


def align_mask_img_to_ref(m: sitk.Image, r: sitk.Image) -> sitk.Image:
    m_sz, m_sp, m_org, m_dir = _image_geom(m)
    r_sz, r_sp, r_org, r_dir = _image_geom(r)
    sizes_match = (m_sz == r_sz)
    spacings_eq = _nearly_equal(m_sp, r_sp, 1e-4)
    dirs_close = _nearly_equal(m_dir, r_dir, 1e-3)
    if sizes_match and spacings_eq and dirs_close and not _nearly_equal(m_org, r_org, 1e-3):
        # header-only mismatch
        return copy_geometry(m, r)
    return resample_nn(m, r)


def align_mask_dir_to_spect(mask_dir: Path, spect_img: sitk.Image, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in _list_nii(mask_dir):
        aligned = align_mask_img_to_ref(sitk.ReadImage(str(p)), spect_img)
        out_path = out_dir / (p.name if p.name.endswith(".gz") else f"{p.name}.gz")
        sitk.WriteImage(aligned, str(out_path))
    return out_dir


# -------------------------
# Foreground/body masks for registration
# -------------------------
def spect_body_mask(img: sitk.Image, lower_percentile: float = 55.0) -> sitk.Image:
    """Create a simple body mask from SPECT using a percentile threshold + largest component."""
    a = sitk.GetArrayFromImage(img).astype(np.float32)
    vals = a[a > 0]
    if vals.size == 0:
        return sitk.Image(img.GetSize(), sitk.sitkUInt8)
    thr = float(np.percentile(vals, lower_percentile))
    bin_img = sitk.BinaryThreshold(img, thr, float(np.max(vals)))
    cc = sitk.ConnectedComponent(bin_img)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    if not stats.GetLabels():
        return sitk.Cast(bin_img, sitk.sitkUInt8)
    largest = max(stats.GetLabels(), key=lambda L: stats.GetPhysicalSize(L))
    body = sitk.BinaryThreshold(cc, largest, largest, insideValue=1, outsideValue=0)
    body = sitk.BinaryMorphologicalClosing(body, (2, 2, 2))
    return sitk.Cast(body, sitk.sitkUInt8)


# -------------------------
# Registration & resampling
# -------------------------
def resample_linear(moving_img: sitk.Image, reference_img: sitk.Image, transform: sitk.Transform, default: float = 0.0) -> sitk.Image:
    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(reference_img)
    rf.SetInterpolator(sitk.sitkLinear)
    rf.SetDefaultPixelValue(default)
    rf.SetTransform(transform)
    return rf.Execute(moving_img)


def register_rigid_t2_to_t1(
    fixed_img: sitk.Image,
    moving_img: sitk.Image,
    use_hist_match: bool = True,
    fixed_mask: Optional[sitk.Image] = None,
    moving_mask: Optional[sitk.Image] = None,
) -> sitk.Transform:
    mov = moving_img
    if use_hist_match:
        hm = sitk.HistogramMatchingImageFilter()
        hm.SetNumberOfHistogramLevels(128)
        hm.SetNumberOfMatchPoints(10)
        hm.SetThresholdAtMeanIntensity(True)
        mov = hm.Execute(moving_img, fixed_img)

    init = sitk.CenteredTransformInitializer(
        fixed_img, mov, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(32)
    if fixed_mask is not None:
        reg.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        reg.SetMetricMovingMask(moving_mask)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed=42)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-3, numberOfIterations=400, relaxationFactor=0.5
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(init, inPlace=False)
    return reg.Execute(fixed_img, mov)


# -------------------------
# Centroids/volumes (physical) + conversions
# -------------------------
def _largest_component_centroid_volume(mask_img: sitk.Image) -> Tuple[Optional[np.ndarray], float]:
    bin_img = mask_img > 0
    cc = sitk.ConnectedComponent(bin_img)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    labels = stats.GetLabels()
    if not labels:
        return None, 0.0
    largest = max(labels, key=lambda L: stats.GetPhysicalSize(L))
    cx, cy, cz = stats.GetCentroid(largest)  # (x,y,z) in mm
    return np.array([cz, cy, cx], float), float(stats.GetPhysicalSize(largest))


def calculate_centroids_and_volumes_from_dir(mask_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    cents: Dict[str, np.ndarray] = {}
    vols: Dict[str, float] = {}
    n_total = 0
    for p in _list_nii(mask_dir):
        n_total += 1
        c, v = _largest_component_centroid_volume(sitk.ReadImage(str(p)))
        if c is None:
            continue
        # Extract first integer from filename as ID; fall back to stem
        m = re.findall(r"\d+", p.stem)
        lesion_id = (m or [p.stem])[0]
        cents[lesion_id] = c
        vols[lesion_id] = v
    logger.info("Lesions in %s: %d (from %d masks)", mask_dir, len(cents), n_total)
    return cents, vols


def transform_centroids_to_fixed_space(centroids_t2_zyx_mm: Mapping[str, np.ndarray], transform: sitk.Transform) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, (z, y, x) in centroids_t2_zyx_mm.items():
        xw, yw, zw = transform.TransformPoint((float(x), float(y), float(z)))
        out[k] = np.array([zw, yw, xw], float)
    return out


def physical_to_voxel_zyx(centroids_zyx_mm: Mapping[str, np.ndarray], ref_img: sitk.Image) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, (z, y, x) in centroids_zyx_mm.items():
        ix, iy, iz = ref_img.TransformPhysicalPointToIndex((float(x), float(y), float(z)))
        out[k] = np.array([iz, iy, ix], float)  # z,y,x indices
    return out


# -------------------------
# Radiomics (simple heterogeneity)
# -------------------------
def build_extractor() -> featureextractor.RadiomicsFeatureExtractor:
    E = featureextractor.RadiomicsFeatureExtractor()
    E.disableAllFeatures()
    E.enableFeatureClassByName("firstorder")
    E.enableFeatureClassByName("glcm")
    return E


def compute_heterogeneity_for_dir(mask_dir: Path, spect_path: Path, extractor: featureextractor.RadiomicsFeatureExtractor) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for p in _list_nii(mask_dir):
        m = re.findall(r"\d+", p.stem)
        lesion_id = (m or [p.stem])[0]
        try:
            r = extractor.execute(str(spect_path), str(p))
            fo = float(r.get("original_firstorder_Entropy", 0.0))
            gl = r.get("original_glcm_JointEntropy", r.get("original_glcm_Entropy", 0.0))
            feats[lesion_id] = fo + float(gl)
        except Exception as e:
            logger.warning("Radiomics failed for %s (%s) — defaulting to 0.0", p.name, e)
            feats[lesion_id] = 0.0
    return feats


# -------------------------
# Features + matching
# -------------------------
def build_feature_matrix(
    centroids: Mapping[str, np.ndarray], volumes: Mapping[str, float], heterogeneity: Mapping[str, float]
) -> Tuple[np.ndarray, List[str]]:
    ids = list(centroids.keys())
    X = []
    for i in ids:
        z, y, x = centroids[i]
        vol = float(volumes.get(i, 0.0))
        het = float(heterogeneity.get(i, 0.0))
        X.append([z, y, x, vol, het])
    X = np.asarray(X, float)
    if X.size and len(X) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            X = (X - np.nanmean(X, 0)) / np.nanstd(X, 0)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, ids


def compute_cost_matrix(
    ids1: List[str],
    ids2: List[str],
    X1: np.ndarray,
    X2: np.ndarray,
    C1_mm: Mapping[str, np.ndarray],
    C2_mm: Mapping[str, np.ndarray],
    spatial_gate_mm: float,
    spatial_scale_mm: float,
    w_spatial: float,
) -> np.ndarray:
    n1, n2 = len(ids1), len(ids2)
    cost = np.full((n1, n2), np.inf, float)
    for i, a in enumerate(ids1):
        for j, b in enumerate(ids2):
            d = float(np.linalg.norm(C1_mm[a] - C2_mm[b]))
            if d > spatial_gate_mm:
                continue
            spatial_cost = min(1.0, d / max(spatial_scale_mm, 1e-8))
            spec_sim = float(cosine_similarity([X1[i]], [X2[j]])[0, 0])
            spec_cost = 0.5 * (1.0 - spec_sim)  # map sim∈[-1,1] → cost∈[0,1]
            cost[i, j] = w_spatial * spatial_cost + (1.0 - w_spatial) * spec_cost
    return cost


def hungarian_assign_partial(cost: np.ndarray, infeasible_fill: float = 1e6) -> List[Tuple[int, int, float]]:
    if cost.size == 0:
        return []
    C = cost.copy().astype(float)
    C[~np.isfinite(C)] = infeasible_fill
    n1, n2 = C.shape
    N = max(n1, n2)
    if n1 != n2:
        C = np.pad(C, ((0, N - n1), (0, N - n2)), constant_values=infeasible_fill)
    ri, ci = linear_sum_assignment(C)
    return [(i, j, float(cost[i, j])) for i, j in zip(ri, ci) if i < n1 and j < n2 and math.isfinite(cost[i, j])]


# -------------------------
# MIPs (joint window) + viz
# -------------------------
def _mip_coronal(a: np.ndarray) -> np.ndarray:
    return np.max(a, axis=1)


def joint_coronal_mips(spect1_path: Path, spect2_path: Path, percentiles: Tuple[float, float] = (1, 99), invert: bool = True):
    img1 = sitk.ReadImage(str(spect1_path))
    img2 = sitk.ReadImage(str(spect2_path))
    a1 = sitk.GetArrayFromImage(img1)
    a2 = sitk.GetArrayFromImage(img2)
    m1 = _mip_coronal(a1)
    m2 = _mip_coronal(a2)
    lo, hi = np.percentile(np.hstack([m1.ravel(), m2.ravel()]), percentiles)

    def norm(m):
        m = np.clip(m, lo, hi)
        m = (m - lo) / max(hi - lo, 1e-8)
        m = np.flipud(m)
        return 1.0 - m if invert else m

    return norm(m1), norm(m2), (float(lo), float(hi))


def visualize_matches_with_unmatched(
    mip1: np.ndarray,
    mip2: np.ndarray,
    centroids1_zyx: Mapping[str, np.ndarray],
    centroids2_zyx: Mapping[str, np.ndarray],
    matches: Mapping[str, Tuple[str, Optional[float]]],
    gap: int = 5,
    image_path: Optional[Path] = None,
    global_vmin: Optional[float] = None,
    global_vmax: Optional[float] = None,
    normalize_costs_to_01: bool = True,
) -> None:
    h1, w1 = mip1.shape
    h2, w2 = mip2.shape
    H = max(h1, h2)
    m1 = np.pad(mip1, ((0, H - h1), (0, 0)), constant_values=0)
    m2 = np.pad(mip2, ((0, H - h2), (0, 0)), constant_values=0)
    combined = np.hstack((m1, np.ones((H, gap)) * 0.5, m2))

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(combined, cmap="gray")
    ax.axis("off")
    x_off = w1 + gap

    costs = [c for (_, c) in matches.values() if c is not None]
    if (global_vmin is not None) and (global_vmax is not None):
        norm = Normalize(global_vmin, global_vmax)
    else:
        norm = Normalize(0.0, 1.0) if (normalize_costs_to_01 or not costs) else Normalize(min(costs), max(costs))
    cmap = LinearSegmentedColormap.from_list("match_strength", [(1, 0, 0), (1, 0.5, 0), (1, 1, 0), (0, 1, 0)][::-1])

    def clamp(z: float, x: float, h: int, w: int) -> Tuple[float, float]:
        return float(np.clip(z, 0, h - 1)), float(np.clip(x, 0, w - 1))

    matched_T1, matched_T2 = set(), set()

    for id1, (id2, c) in matches.items():
        if id2 is None or id1 not in centroids1_zyx or id2 not in centroids2_zyx:
            continue
        matched_T1.add(id1)
        matched_T2.add(id2)
        z1, x1 = centroids1_zyx[id1][0], centroids1_zyx[id1][2]
        z1, x1 = clamp(z1, x1, h1, w1)
        y1 = h1 - z1
        z2, x2 = centroids2_zyx[id2][0], centroids2_zyx[id2][2]
        z2, x2 = clamp(z2, x2, h2, w2)
        y2 = h2 - z2
        color = cmap(norm(1.0 if c is None else c))
        ax.plot([x1, x2 + x_off], [y1, y2], color=color, linewidth=2)
        ax.scatter([x1, x2 + x_off], [y1, y2], edgecolors=color, facecolors="none", s=120)
        ax.text(x1, y1 - 5, str(id1), color="cyan", fontsize=9, ha="center")
        ax.text(x2 + x_off, y2 - 5, str(id2), color="magenta", fontsize=9, ha="center")

    for id1 in set(centroids1_zyx.keys()) - matched_T1:
        z, x = centroids1_zyx[id1][0], centroids1_zyx[id1][2]
        z, x = clamp(z, x, h1, w1)
        y = h1 - z
        ax.scatter(x, y, color="cyan", edgecolors="black", s=120, linewidth=1.5)
        ax.text(x, y - 5, str(id1), color="cyan", fontsize=9, ha="center")

    for id2 in set(centroids2_zyx.keys()) - matched_T2:
        z, x = centroids2_zyx[id2][0], centroids2_zyx[id2][2]
        z, x = clamp(z, x, h2, w2)
        y = h2 - z
        ax.scatter(x + x_off, y, color="magenta", edgecolors="black", s=120, linewidth=1.5)
        ax.text(x + x_off, y - 5, str(id2), color="magenta", fontsize=9, ha="center")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Matching Cost (Lower is Better)", fontsize=10)
    legend = [
        Line2D([0], [0], color="red", lw=2, label="Weak Match"),
        Line2D([0], [0], color="green", lw=2, label="Strong Match"),
        Line2D([0], [0], marker="o", color="cyan", label="Missed", markersize=8, linewidth=0),
        Line2D([0], [0], marker="o", color="magenta", label="New", markersize=8, linewidth=0),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
    plt.tight_layout()
    if image_path is not None:
        plt.savefig(str(image_path), dpi=300)
    plt.show()


# -------------------------
# QC helpers
# -------------------------
def _mip_from_img(img: sitk.Image, pmin: float = 1, pmax: float = 99) -> np.ndarray:
    a = sitk.GetArrayFromImage(img)
    m = _mip_coronal(a)
    lo, hi = np.percentile(m, [pmin, pmax])
    m = (np.clip(m, lo, hi) - lo) / max(hi - lo, 1e-8)
    return np.flipud(m)


def qc_overlay_and_stats(
    fixed_img: sitk.Image,
    moving_img_reg: sitk.Image,
    pct: Tuple[float, float] = (1, 99),
    title: str = "Registration QC",
    save_path: Optional[Path] = None,
) -> Tuple[float, float]:
    m1 = _mip_from_img(fixed_img, *pct)
    m2 = _mip_from_img(moving_img_reg, *pct)
    H, W = max(m1.shape[0], m2.shape[0]), max(m1.shape[1], m2.shape[1])

    def pad(m: np.ndarray) -> np.ndarray:
        return np.pad(m, ((0, H - m.shape[0]), (0, W - m.shape[1])), constant_values=0)

    m1p, m2p = pad(m1), pad(m2)
    lo, hi = np.percentile(np.hstack([m1p.ravel(), m2p.ravel()]), pct)
    a1 = (np.clip(m1p, lo, hi) - lo) / max(hi - lo, 1e-8)
    a2 = (np.clip(m2p, lo, hi) - lo) / max(hi - lo, 1e-8)
    corr = float(np.corrcoef(a1.ravel(), a2.ravel())[0, 1])
    mad = float(np.mean(np.abs(a1 - a2)))
    blend = 0.5 * m1p + 0.5 * m2p

    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    axs[0].imshow(m1p, cmap="gray")
    axs[0].set_title("T1 MIP")
    axs[0].axis("off")
    axs[1].imshow(m2p, cmap="gray")
    axs[1].set_title("T2 (rigid→T1) MIP")
    axs[1].axis("off")
    axs[2].imshow(blend, cmap="gray")
    axs[2].set_title(f"Blend (corr={corr:.3f}, MAD={mad:.3f})")
    axs[2].axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=200)
    plt.show()
    logger.info("QC corr=%.4f, MAD=%.4f", corr, mad)
    return corr, mad


# -------------------------
# Main pipeline
# -------------------------
def run(cfg: Config) -> None:
    np.random.seed(cfg.random_seed)

    # Resolve paths and check existence
    spect1_path = cfg.spect1_path()
    spect2_path = cfg.spect2_path()
    mask_dir1_in = cfg.mask1_dir()
    mask_dir2_in = cfg.mask2_dir()

    _ensure_exists(spect1_path, "SPECT T1")
    _ensure_exists(spect2_path, "SPECT T2")
    _ensure_exists(mask_dir1_in, "T1 mask dir")
    _ensure_exists(mask_dir2_in, "T2 mask dir")

    # Read SPECTs
    spect1 = sitk.ReadImage(str(spect1_path))
    spect2 = sitk.ReadImage(str(spect2_path))
    _print_geom("SPECT T1", spect1)
    _print_geom("SPECT T2", spect2)

    # Align masks to their own SPECTs
    mask_dir1 = cfg.t1_dir() / f"{cfg.mask_subdir}_aligned"
    mask_dir2 = cfg.t2_dir() / f"{cfg.mask_subdir}_aligned"
    align_mask_dir_to_spect(mask_dir1_in, spect1, mask_dir1)
    align_mask_dir_to_spect(mask_dir2_in, spect2, mask_dir2)

    # Body masks & registration
    fixed_mask = spect_body_mask(spect1, cfg.fixed_mask_pct)
    moving_mask = spect_body_mask(spect2, cfg.moving_mask_pct)
    logger.info("Running rigid registration (T2 → T1)...")
    tx_2to1 = register_rigid_t2_to_t1(
        spect1,
        spect2,
        use_hist_match=cfg.use_histogram_matching,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
    )
    print_transform_info(tx_2to1)

    # Resample SPECT2 → T1
    spect2_reg_img = resample_linear(spect2, spect1, tx_2to1, default=0.0)
    spect2_reg_path = cfg.patient_folder / f"{cfg.time2}_spect_rigid2{cfg.time1}.nii.gz"
    sitk.WriteImage(spect2_reg_img, str(spect2_reg_path))
    _print_geom("SPECT T2 (registered→T1)", spect2_reg_img)

    # (Optional) resample T2 masks → T1 and recompute centroids/volumes THERE
    if cfg.resample_t2_masks:
        mask_dir2_T1 = cfg.patient_folder / f"{cfg.time2}_{cfg.mask_subdir}_rigid2{cfg.time1}"
        mask_dir2_T1.mkdir(parents=True, exist_ok=True)
        for p in _list_nii(mask_dir2):
            m_img = sitk.ReadImage(str(p))
            sitk.WriteImage(resample_nn(m_img, spect1, tx_2to1), str(mask_dir2_T1 / p.name))
        C2_mm_T1, V2_T1 = calculate_centroids_and_volumes_from_dir(mask_dir2_T1)  # in T1 geometry
        mask_dir2_for_radiomics = mask_dir2_T1
    else:
        # Native T2 centroids/volumes + transform centroids → T1
        C2_mm_native, V2_T1 = calculate_centroids_and_volumes_from_dir(mask_dir2)
        C2_mm_T1 = transform_centroids_to_fixed_space(C2_mm_native, tx_2to1)
        mask_dir2_for_radiomics = mask_dir2  # radiomics will use native SPECT2

    # QC overlay figure
    qc_png = cfg.patient_folder / f"registration_QC_{cfg.time2}_to_{cfg.time1}.png"
    qc_overlay_and_stats(spect1, spect2_reg_img, pct=cfg.mip_percentiles, title="Rigid registration QC", save_path=qc_png)

    # T1 centroids/volumes
    C1_mm, V1 = calculate_centroids_and_volumes_from_dir(mask_dir1)

    # Radiomics
    extractor = build_extractor()
    H1 = compute_heterogeneity_for_dir(mask_dir1, spect1_path, extractor)
    if cfg.resample_t2_masks:
        H2 = compute_heterogeneity_for_dir(mask_dir2_for_radiomics, spect2_reg_path, extractor)
    else:
        H2 = compute_heterogeneity_for_dir(mask_dir2_for_radiomics, spect2_path, extractor)

    # Features
    X1, ids1 = build_feature_matrix(C1_mm, V1, H1)
    X2, ids2 = build_feature_matrix(C2_mm_T1, V2_T1, H2)

    # Distance sanity
    if ids1 and ids2:
        dists = [np.min([np.linalg.norm(C1_mm[a] - C2_mm_T1[b]) for b in ids2]) for a in ids1]
        logger.info(
            "Nearest T2 distance (mm): min=%.1f, median=%.1f, max=%.1f",
            float(np.min(dists)),
            float(np.median(dists)),
            float(np.max(dists)),
        )

    # Matching
    C = compute_cost_matrix(
        ids1,
        ids2,
        X1,
        X2,
        C1_mm,
        C2_mm_T1,
        spatial_gate_mm=cfg.spatial_gate_mm,
        spatial_scale_mm=cfg.spatial_scale_mm,
        w_spatial=cfg.cost_weight_spatial,
    )
    feasible = int(np.isfinite(C).sum())
    logger.info("Feasible pairs within gate: %d", feasible)
    assign_list = hungarian_assign_partial(C)

    # Build table
    rows: List[List[Optional[float]]] = []
    matched_T1, matched_T2 = set(), set()
    for i, j, cost in assign_list:
        id1, id2 = ids1[i], ids2[j]
        dist = float(np.linalg.norm(C1_mm[id1] - C2_mm_T1[id2]))
        rows.append([id1, id2, cost, dist])
        matched_T1.add(id1)
        matched_T2.add(id2)

    for id1 in ids1:
        if id1 not in matched_T1:
            rows.append([id1, "None", None, None])
    for id2 in ids2:
        if id2 not in matched_T2:
            rows.append(["None", id2, None, None])

    out_csv = cfg.patient_folder / f"matched_lesions_{cfg.time1}_to_{cfg.time2}_{cfg.output_tag}.csv"
    pd.DataFrame(rows, columns=["Lesion_T1", "Lesion_T2", "Cost", "Distance_mm"]).to_csv(out_csv, index=False)
    logger.info("Saved matches: %s", out_csv)

    # Joint-window MIPs (both panels in T1 geometry!)
    mip1, mip2, _ = joint_coronal_mips(spect1_path, spect2_reg_path, percentiles=cfg.mip_percentiles)

    # Convert BOTH centroid sets to T1 voxel indices for plotting
    C1_vox_T1 = physical_to_voxel_zyx(C1_mm, spect1)
    C2_vox_T1 = physical_to_voxel_zyx(C2_mm_T1, spect1)

    matches_by_id: Dict[str, Tuple[str, Optional[float]]] = {ids1[i]: (ids2[j], c) for i, j, c in assign_list}
    out_png = cfg.patient_folder / f"lesion_tracking_{cfg.time1}_to_{cfg.time2}_{cfg.output_tag}.png"
    visualize_matches_with_unmatched(
        mip1,
        mip2,
        C1_vox_T1,
        C2_vox_T1,
        matches_by_id,
        image_path=out_png,
        global_vmin=0.0,
        global_vmax=1.0,
    )
    logger.info("Tracking figure saved: %s", out_png)
    logger.info("Done.")


# -------------------------
# CLI
# -------------------------
def cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Longitudinal lesion tracking (SPECT/CT).")
    p.add_argument("--patient-folder", type=Path)
    p.add_argument("--time1")
    p.add_argument("--time2")
    p.add_argument("--spect-name")
    p.add_argument("--spect-subdir")
    p.add_argument("--mask-subdir")
    p.add_argument("--use-histogram-matching", type=lambda s: s.lower() in {"1", "true", "yes"})
    p.add_argument("--fixed-mask-pct", type=float)
    p.add_argument("--moving-mask-pct", type=float)
    p.add_argument("--resample-t2-masks", type=lambda s: s.lower() in {"1", "true", "yes"})
    p.add_argument("--spatial-gate-mm", type=float)
    p.add_argument("--spatial-scale-mm", type=float)
    p.add_argument("--cost-weight-spatial", type=float)
    p.add_argument("--mip-percentiles", nargs=2, type=float, metavar=("PLOW", "PHIGH"))
    p.add_argument("--output-tag")
    p.add_argument("--random-seed", type=int)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = {k: v for k, v in vars(p.parse_args()).items() if v is not None}
    logging.getLogger().setLevel(args.pop("log-level", "INFO"))

    cfg = Config()
    if "mip_percentiles" in args:
        args["mip_percentiles"] = (args["mip_percentiles"][0], args["mip_percentiles"][1])
    cfg = Config(**{**asdict(cfg), **args})
    run(cfg)


if __name__ == "__main__":
    cli()
