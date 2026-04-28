#!/usr/bin/env python3
"""
uncertainty_analysis.py — Consolidated MC Dropout Uncertainty Analysis

Generates uncertainty visualizations and metrics from MC dropout sample frames.
Designed to be run for each model (SR-MedT++, SOTA models, etc.) by pointing
--mc_dir at that model's ENSEMBLE_RAW_FRAMES folder.

Outputs saved to --output_dir/:
  probability_profiles/          1D probability profile plots per patient
  uncertainty_overlays/          Band overlays on original images (requires --image_dir)
  thickness_profiles/            Radial thickness CSVs + plots per patient
  <model_name>_uncertainty_metrics.csv   Brier, NLL, ECE, band area per patient
  <model_name>_thickness_summary.csv     Summary radial thickness statistics

Pixel-to-mm conversion:
  Pass --conversion_csv pointing to a headerless CSV with one mm/pixel value per
  line, ordered to match the sorted patient folder list (first patient's first
  slice on row 1, last patient's last slice on the final row).
  If omitted, --pixel_spacing (default 1.0) is used as a constant fallback.

Usage examples:
  # SR-MedT++ (full run with GT masks, overlays, and per-slice mm conversion):
  python uncertainty_analysis.py \\
      --mc_dir  /path/to/SR-MedT/ENSEMBLE_RAW_FRAMES \\
      --gt_dir  /path/to/dataset/Masks \\
      --image_dir /path/to/dataset/Images \\
      --conversion_csv uncertainty_scripts/conversion_factors.csv \\
      --output_dir results/SR-MedT \\
      --model_name SR-MedT

  # SOTA model (metrics only, no overlays):
  python uncertainty_analysis.py \\
      --mc_dir  /path/to/UNetPP/ENSEMBLE_RAW_FRAMES \\
      --gt_dir  /path/to/dataset/Masks \\
      --conversion_csv uncertainty_scripts/conversion_factors.csv \\
      --output_dir results/UNetPP \\
      --model_name UNetPP
"""

import argparse
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
from sklearn.metrics import brier_score_loss, log_loss
from tqdm import tqdm

# ==================== CONSTANTS ====================
# Uncertainty band thresholds (probability ratios):
#   Outer  = lower confidence bound  (-1 std dev proxy)
#   Inner  = upper confidence bound  (+1 std dev proxy)
# Pixels with  THRESH_OUTER <= P <= THRESH_INNER  form the "uncertainty band".
THRESH_OUTER = 0.16
THRESH_INNER = 0.84
N_BINS = 10  # bins used in Expected Calibration Error


# ==================== SHARED UTILITIES ====================

def load_mc_stack(patient_dir: str) -> np.ndarray | None:
    """
    Load all BMP MC dropout frames from a patient folder.

    Supports two layouts:
      Flat:   <patient_dir>/*.bmp
      Nested: <patient_dir>/<fold_*>/*.bmp   (e.g. 5 folds × 100 samples = 500 total)

    Returns a float32 array of shape (N_samples, H, W) with values in [0, 1],
    or None if no BMP files are found.
    """
    files = sorted(glob.glob(os.path.join(patient_dir, "*.png")))
    if not files:
        files = sorted(glob.glob(os.path.join(patient_dir, "*.bmp")))
    if not files:
        # Try one level deeper (fold subfolders)
        files = sorted(glob.glob(os.path.join(patient_dir, "*", "*.png")))
    if not files:
        files = sorted(glob.glob(os.path.join(patient_dir, "*", "*.bmp")))
    if not files:
        return None
    first = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    if first is None:
        return None
    H, W = first.shape
    stack = np.zeros((len(files), H, W), dtype=np.float32)
    for i, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if img.shape != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            stack[i] = img / 255.0
    return stack


def compute_prob_maps(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_map, std_map) from an MC sample stack."""
    return np.mean(stack, axis=0), np.std(stack, axis=0)


def get_safe_centroid(binary_mask: np.ndarray) -> tuple[int, int] | None:
    """
    Return (cX, cY) centroid of the largest connected component in a binary mask.
    Accepts a uint8 mask (0/255) or a boolean array.
    Returns None if the mask is empty.
    """
    if binary_mask.dtype != np.uint8:
        mask_u8 = (binary_mask.astype(bool) * 255).astype(np.uint8)
    else:
        mask_u8 = binary_mask
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] != 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return None


def _index_directory(directory: str, extensions=("*.bmp", "*.png", "*.jpg", "*.jpeg", "*.bmp")) -> dict[str, str]:
    """
    Build a {stem -> full_path} map for all files in directory matching extensions.
    """
    file_map: dict[str, str] = {}
    for ext in extensions:
        for path in glob.glob(os.path.join(directory, ext)):
            key = os.path.splitext(os.path.basename(path))[0].strip()
            file_map[key] = path
    return file_map


def _lookup(file_map: dict[str, str], patient_id: str) -> str | None:
    """Exact match first, then partial substring fallback."""
    if patient_id in file_map:
        return file_map[patient_id]
    stripped = patient_id.strip()
    if stripped in file_map:
        return file_map[stripped]
    # Partial fallback
    return next((v for k, v in file_map.items() if patient_id in k), None)


# ==================== VISUALIZATION: POLAR UNCERTAINTY PLOT ====================

def generate_polar_plot(
    patient_id: str,
    mean_map: np.ndarray,
    output_dir: str,
    mm_per_px: float,
) -> dict | None:
    """
    Generate a polar bar chart of uncertainty band width (in mm) at every degree.
    Each bar represents the radial thickness of the uncertainty band at that angle.

    Returns a summary dict with mean/max thickness, or None if the prediction is empty.
    """
    H, W = mean_map.shape
    binary_center = mean_map > 0.5
    if binary_center.sum() == 0:
        return None

    cy, cx = center_of_mass(binary_center)
    center = (int(cx), int(cy))
    max_radius = float(np.sqrt((H / 2) ** 2 + (W / 2) ** 2))

    polar_img = cv2.linearPolar(mean_map, center, max_radius, cv2.WARP_FILL_OUTLIERS)
    polar_img = cv2.resize(polar_img, (int(max_radius), 360),
                           interpolation=cv2.INTER_LINEAR)

    widths_mm = []
    for angle in range(360):
        row = polar_img[angle, :]
        band_px = np.sum((row >= THRESH_OUTER) & (row <= THRESH_INNER))
        widths_mm.append(float(band_px) * mm_per_px)

    widths_mm = np.array(widths_mm)
    mean_thick = float(np.mean(widths_mm))
    max_thick  = float(np.max(widths_mm))
    max_angle  = int(np.argmax(widths_mm))

    theta = np.radians(np.arange(360))

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("E")
    ax.bar(theta, widths_mm, width=np.radians(1.0), bottom=0.0,
           color="crimson", alpha=0.8)

    max_w = float(np.max(widths_mm)) if np.max(widths_mm) > 0 else 5.0
    ticks = np.linspace(0, max_w, 5)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.2f} mm" for t in ticks], fontsize=9, fontweight="bold")
    ax.set_rlabel_position(45)
    ax.set_title(
        f"{patient_id}\nUncertainty Band Width (mm) — Mean: {mean_thick:.2f} mm",
        va="bottom", fontsize=11,
    )

    plt.savefig(os.path.join(output_dir, f"{patient_id}_polar.png"),
                dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {
        "Patient_ID":          patient_id,
        "Mean_Thickness_mm":   round(mean_thick, 4),
        "Max_Thickness_mm":    round(max_thick, 4),
        "Max_Thickness_Angle": max_angle,
        "Conv_Factor_mm_px":   round(mm_per_px, 8),
    }


# ==================== VISUALIZATION: UNCERTAINTY OVERLAYS ====================

def _draw_legend(image: np.ndarray) -> np.ndarray:
    """Draw a semi-transparent legend in the top-right corner (in-place)."""
    h, w = image.shape[:2]
    box_w, box_h, pad = 240, 135, 10
    font = cv2.FONT_HERSHEY_SIMPLEX

    items = [
        ((255, 255, 255), "Mean Prediction",   "solid"),
        ((0,   255,   0), "Mean + 1 Std Dev",  "dashed"),
        ((0,   100, 255), "Mean - 1 Std Dev",  "dashed"),
        ((0,   140, 255), "Uncertainty Band",  "fill"),
        ((0,   255, 255), "Centroid",          "marker"),
    ]

    x1, y1 = w - box_w - pad, pad
    x2, y2 = w - pad, pad + box_h
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

    start_y = y1 + 25
    lx1, lx2, tx = x1 + 10, x1 + 40, x1 + 50
    for color, label, style in items:
        if style == "fill":
            cv2.rectangle(image, (lx1, start_y - 6), (lx2, start_y + 6), color, -1)
        elif style == "solid":
            cv2.line(image, (lx1, start_y), (lx2, start_y), color, 2)
        elif style == "dashed":
            for dx in (5, 15, 25):
                cv2.circle(image, (lx1 + dx, start_y), 2, color, -1)
        elif style == "marker":
            cv2.drawMarker(image, ((lx1 + lx2) // 2, start_y),
                           color, cv2.MARKER_CROSS, 10, 2)
        cv2.putText(image, label, (tx, start_y + 4), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        start_y += 25

    return image


def generate_uncertainty_overlay(
    patient_id: str,
    mean_map: np.ndarray,
    orig_img_path: str,
    output_dir: str,
) -> None:
    """
    Overlay the uncertainty band + contours on the original ultrasound image
    and save as a JPG.  Contours:
      - White  solid  = mean prediction boundary (P = 0.5)
      - Green  dashed = inner confidence boundary (P = THRESH_INNER)
      - Blue   dashed = outer confidence boundary (P = THRESH_OUTER)
    """
    orig = cv2.imread(orig_img_path)
    if orig is None:
        return

    H_o, W_o = orig.shape[:2]
    prob = cv2.resize(mean_map, (W_o, H_o), interpolation=cv2.INTER_LINEAR)
    prob_u8 = (prob * 255).astype(np.uint8)

    band_mask = ((prob >= THRESH_OUTER) & (prob <= THRESH_INNER))
    band_color = np.zeros_like(orig)
    band_color[:, :] = (0, 140, 255)  # BGR orange
    result = orig.copy()
    alpha = 0.45
    where_band = band_mask
    result[where_band] = cv2.addWeighted(
        orig, 1 - alpha, band_color, alpha, 0
    )[where_band]

    def draw_contours(threshold_val, color, thickness=2):
        _, thresh = cv2.threshold(prob_u8, threshold_val, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, cnts, -1, color, thickness)

    draw_contours(127,                    (255, 255, 255), 1)  # mean boundary
    draw_contours(int(THRESH_INNER * 255),(0,   255,   0), 1)  # inner bound (green)
    draw_contours(int(THRESH_OUTER * 255),(0,   100, 255), 1)  # outer bound (blue)

    center_u8 = (prob > 0.5).astype(np.uint8) * 255
    cp = get_safe_centroid(center_u8)
    if cp:
        cv2.drawMarker(result, cp, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    result = _draw_legend(result)
    cv2.imwrite(os.path.join(output_dir, f"{patient_id}_overlay.jpg"), result)


# ==================== METRICS: CALIBRATION ====================

def _calculate_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = N_BINS) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probs)
    for i in range(n_bins):
        bm = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        bc = int(np.sum(bm))
        if bc > 0:
            ece += (bc / total) * abs(float(np.mean(y_true[bm])) - float(np.mean(probs[bm])))
    return ece


def calculate_calibration_metrics(
    patient_id: str,
    mean_map: np.ndarray,
    gt_mask_path: str,
    pixel_spacing_mm: float,
) -> dict | None:
    """
    Compute uncertainty band area, Brier score, NLL, and ECE vs. ground truth.
    ROI is defined as pixels where P > 1% OR GT is positive.
    Returns a metrics dict or None on failure.
    """
    gt = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return None
    gt_binary = (gt > 0).astype(np.float32)
    H_gt, W_gt = gt_binary.shape

    prob = cv2.resize(mean_map, (W_gt, H_gt), interpolation=cv2.INTER_LINEAR)

    # Uncertainty band geometry
    mask_total = prob >= THRESH_OUTER
    mask_core  = prob >= THRESH_INNER
    mask_band  = np.logical_xor(mask_total, mask_core)
    area_band_mm2  = float(np.sum(mask_band)) * (pixel_spacing_mm ** 2)
    area_total_px  = float(np.sum(mask_total))
    unc_ratio      = (float(np.sum(mask_band)) / area_total_px) if area_total_px > 0 else 0.0

    # Calibration metrics on ROI
    flat_probs = prob.flatten()
    flat_gt    = gt_binary.flatten()
    roi_mask   = (flat_probs > 0.01) | (flat_gt > 0.5)
    if roi_mask.sum() == 0:
        return None

    roi_probs = flat_probs[roi_mask]
    roi_gt    = flat_gt[roi_mask]
    roi_probs_c = np.clip(roi_probs, 1e-15, 1 - 1e-15)

    return {
        "Patient_ID":           patient_id,
        "Uncertainty_Area_mm2": round(area_band_mm2, 2),
        "Uncertainty_Ratio":    round(unc_ratio, 4),
        "Brier_Score":          round(float(brier_score_loss(roi_gt, roi_probs)), 5),
        "NLL":                  round(float(log_loss(roi_gt, roi_probs_c)), 5),
        "ECE":                  round(_calculate_ece(roi_probs, roi_gt), 5),
    }


# ==================== MAIN ====================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MC Dropout Uncertainty Analysis — visualizations and metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mc_dir", required=True,
        help="Directory of patient subfolders, each containing MC dropout BMP frames.",
    )
    parser.add_argument(
        "--gt_dir", default=None,
        help="(Optional) Ground truth masks directory (BMP/PNG). "
             "Required for calibration metrics (Brier, NLL, ECE, band area).",
    )
    parser.add_argument(
        "--image_dir", default=None,
        help="(Optional) Original ultrasound images directory. "
             "Required for uncertainty band overlay visualizations.",
    )
    parser.add_argument(
        "--output_dir", default="uncertainty_output",
        help="Root output directory (created if missing). Default: uncertainty_output/",
    )
    parser.add_argument(
        "--conversion_csv", default=None,
        help="Path to a headerless CSV with one mm/pixel value per line, ordered to "
             "match the sorted patient folder list. "
             "Use uncertainty_scripts/conversion_factors.csv for the test set.",
    )
    parser.add_argument(
        "--pixel_spacing", type=float, default=1.0,
        help="Fallback mm/pixel constant used when --conversion_csv is not provided. "
             "Default: 1.0",
    )
    parser.add_argument(
        "--model_name", default="model",
        help="Short model label used in output CSV filenames. Default: model",
    )
    args = parser.parse_args()

    conv_factors: list[float] = []
    if args.conversion_csv:
        with open(args.conversion_csv) as f:
            for line in f:
                line = line.strip()
                if line:
                    conv_factors.append(float(line))
        print(f"Loaded {len(conv_factors)} conversion factors from: {args.conversion_csv}")
    else:
        print(f"No --conversion_csv provided. Using constant pixel_spacing={args.pixel_spacing} mm/px for all slices.")

    sub_dirs = {
        "overlays": os.path.join(args.output_dir, "uncertainty_overlays"),
        "polar":    os.path.join(args.output_dir, "polar_plots"),
    }
    for d in sub_dirs.values():
        os.makedirs(d, exist_ok=True)

    mask_map: dict[str, str] = {}
    if args.gt_dir:
        mask_map = _index_directory(args.gt_dir)
        print(f"Indexed {len(mask_map)} GT masks from: {args.gt_dir}")

    image_map: dict[str, str] = {}
    if args.image_dir:
        image_map = _index_directory(args.image_dir)
        print(f"Indexed {len(image_map)} original images from: {args.image_dir}")

    patient_folders = sorted(
        p for p in glob.glob(os.path.join(args.mc_dir, "*")) if os.path.isdir(p)
    )
    if not patient_folders:
        print(f"No patient subfolders found in: {args.mc_dir}")
        return
    print(f"Found {len(patient_folders)} patient folders in: {args.mc_dir}")

    if conv_factors and len(conv_factors) != len(patient_folders):
        print(
            f"[WARN] conversion_csv has {len(conv_factors)} entries but "
            f"{len(patient_folders)} patient folders were found. "
            "Indices that exceed the CSV length will fall back to pixel_spacing."
        )
    print()

    calibration_rows: list[dict] = []
    polar_rows:       list[dict] = []

    for idx, folder in enumerate(tqdm(patient_folders, desc="Patients")):
        patient_id = os.path.basename(folder)

        if conv_factors and idx < len(conv_factors):
            mm_per_px = conv_factors[idx]
        else:
            mm_per_px = args.pixel_spacing

        stack = load_mc_stack(folder)
        if stack is None:
            tqdm.write(f"  [SKIP] No PNG/BMP frames: {patient_id}")
            continue
        mean_map, std_map = compute_prob_maps(stack)

        polar_row = generate_polar_plot(patient_id, mean_map, sub_dirs["polar"], mm_per_px)
        if polar_row:
            polar_rows.append(polar_row)

        if image_map:
            orig_path = _lookup(image_map, patient_id)
            if orig_path:
                generate_uncertainty_overlay(patient_id, mean_map, orig_path,
                                             sub_dirs["overlays"])
            else:
                tqdm.write(f"  [WARN] No matching original image for: {patient_id}")

        if mask_map:
            gt_path = _lookup(mask_map, patient_id)
            if gt_path:
                metrics = calculate_calibration_metrics(
                    patient_id, mean_map, gt_path, mm_per_px
                )
                if metrics:
                    metrics["Conv_Factor_mm_px"] = round(mm_per_px, 8)
                    calibration_rows.append(metrics)
            else:
                tqdm.write(f"  [WARN] No matching GT mask for: {patient_id}")

    print()
    if calibration_rows:
        df_cal = pd.DataFrame(calibration_rows).sort_values("Brier_Score", ascending=False)
        cal_path = os.path.join(args.output_dir, f"{args.model_name}_uncertainty_metrics.csv")
        df_cal.to_csv(cal_path, index=False)
        print(f"Calibration metrics  -> {cal_path}")
        print(f"  Patients matched : {len(calibration_rows)}")
        print(f"  Mean ECE         : {df_cal['ECE'].mean():.4f}")
        print(f"  Mean Brier       : {df_cal['Brier_Score'].mean():.4f}")
        print(f"  Mean NLL         : {df_cal['NLL'].mean():.4f}")

    if polar_rows:
        df_polar = pd.DataFrame(polar_rows)
        polar_path = os.path.join(args.output_dir, f"{args.model_name}_polar_summary.csv")
        df_polar.to_csv(polar_path, index=False)
        print(f"Polar summary        -> {polar_path}")
        print(f"  Mean thickness   : {df_polar['Mean_Thickness_mm'].mean():.4f} mm")

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
