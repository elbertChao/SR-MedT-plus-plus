import os
import cv2
import pandas as pd
import numpy as np
from glob import glob

csv_path = "results_5folds/test_predictions/metrics_results_Fold4 - metrics_results.csv"
img_dir = "test_data_2/img"
label_dir = "test_data_2/labelcol"
pred_dir = "results_5folds/test_predictions"
save_dir = "results_5folds/bad_slices_visualizations"

os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} entries from metrics file")
print(df.columns.tolist())
print(df.head())

f1_col = "mean_f1"
img_col = "slice_name"

bad_slices = df[df[f1_col] < 0.5]
print(f"Found {len(bad_slices)} slices with F1 < 0.5")

for i, row in bad_slices.iterrows():
    fname = os.path.basename(row[img_col])
    base_name = os.path.splitext(fname)[0]
    
    img_path = os.path.join(img_dir, fname)
    gt_path = os.path.join(label_dir, fname)
    pred_path = os.path.join(pred_dir, fname)

    if not (os.path.exists(img_path) and os.path.exists(gt_path) and os.path.exists(pred_path)):
        print(f"⚠️ Missing file for {fname}, skipping")
        continue

    img = cv2.imread(img_path)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    if img is None or gt is None or pred is None:
        print(f"⚠️ Could not read {fname}, skipping")
        continue

    _, gt_bin = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
    _, pred_bin = cv2.threshold(pred, 127, 255, cv2.THRESH_BINARY)

    contours_gt, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_pred, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = img.copy()
    cv2.drawContours(overlay, contours_gt, -1, (0, 255, 0), 1)   # green = ground truth
    cv2.drawContours(overlay, contours_pred, -1, (0, 0, 255), 1) # red = prediction

    save_path = os.path.join(save_dir, f"{base_name}_overlay.png")
    cv2.imwrite(save_path, overlay)

print(f"✅ Saved all overlays to: {save_dir}")
