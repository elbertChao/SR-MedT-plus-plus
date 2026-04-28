import os
import cv2
import glob
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import time
import matplotlib
matplotlib.use('Agg')

def get_args():
    parser = argparse.ArgumentParser(description='Generate mm-scaled Uncertainty and Whisker Plots')
    parser.add_argument('--mc_dir', type=str, default='./MC_SAMPLES', help='Path to MC_SAMPLES directory')
    parser.add_argument('--mask_dir', type=str, default='./dataset/Masks', help='Path to ground truth masks')
    parser.add_argument('--img_dir', type=str, default='./dataset/Images', help='Path to original contrast enhanced images')
    parser.add_argument('--coords_excel', type=str, default='annotate_coords.xlsx', help='Path to annotate_coords.xlsx')
    parser.add_argument('--conversion_excel', type=str, default="./conversion_factors.xlsx", help='Path to conversion_factors.xlsx')
    parser.add_argument('--remote_out_dir', type=str, default='./uncert_results/inner_arc_visualizations', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--thresh_inner', type=float, default=0.84)
    parser.add_argument('--thresh_outer', type=float, default=0.16)

    return parser.parse_args()

def load_all_points(excel_path):
    df = pd.read_excel(excel_path)
    points_dict = {}
    for _, r in df.iterrows():
        fname = str(r['Filename']).strip()
        
        if fname.startswith("outline_"):
            fname = fname.replace("outline_", "")
        fname = os.path.splitext(fname)[0]
            
        points_dict[fname] = (
            (float(r['X1']), float(r['Y1'])), 
            (float(r['X2']), float(r['Y2'])), 
            (float(r['X3']), float(r['Y3']))
        )
    return points_dict

def get_contour_from_array(mask_array):
    mask_uint8 = (mask_array > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return None
    return max(contours, key=cv2.contourArea).squeeze()

def get_inner_arc(contour, p1, p2, p3):
    if len(contour.shape) == 1: contour = np.expand_dims(contour, axis=0)
    dists_p1 = np.linalg.norm(contour - np.array(p1), axis=1)
    dists_p2 = np.linalg.norm(contour - np.array(p2), axis=1)
    idx1, idx2 = np.argmin(dists_p1), np.argmin(dists_p2)
    
    if idx1 > idx2: idx1, idx2 = idx2, idx1
        
    arc_a = contour[idx1:idx2+1]
    arc_b = np.concatenate((contour[idx2:], contour[:idx1+1]))
    
    dist_a = np.min(np.linalg.norm(arc_a - np.array(p3), axis=1)) if len(arc_a) > 0 else float('inf')
    dist_b = np.min(np.linalg.norm(arc_b - np.array(p3), axis=1)) if len(arc_b) > 0 else float('inf')
    
    return arc_a if dist_a < dist_b else arc_b

def calculate_signed_distances_mm(gt_arc, pred_arc, pred_mask, conv_factor):
    d_matrix = cdist(gt_arc, pred_arc)
    distances_mm, signs, nearest_indices = [], [], []
    
    for i, gt_point in enumerate(gt_arc):
        min_idx = np.argmin(d_matrix[i])
        min_dist_px = d_matrix[i][min_idx]
        
        x, y = int(gt_point[0]), int(gt_point[1])
        h, w = pred_mask.shape
        
        is_inside = False
        if 0 <= x < w and 0 <= y < h:
            is_inside = pred_mask[y, x] > 0
            
        sign = 1 if is_inside else -1
        
        distances_mm.append(min_dist_px * conv_factor)
        signs.append(sign)
        nearest_indices.append(min_idx)
        
    return np.array(distances_mm), np.array(signs), np.array(nearest_indices)

def plot_whiskers_with_legend(ax, fig, gt_arc, pred_arc, dists_mm, signs, nearest_indices):
    signed_dists = dists_mm * signs
    vmax = max(np.abs(signed_dists).max(), 0.01)

    cmap = plt.get_cmap('seismic')
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)

    ax.plot(gt_arc[:, 0],   gt_arc[:, 1],   'k-',  linewidth=2,   label='Ground Truth', zorder=3)
    ax.plot(pred_arc[:, 0], pred_arc[:, 1],  'w--', linewidth=1,   alpha=0.6, label='Mean Prediction', zorder=2)

    for i in range(0, len(gt_arc), 5):
        p_gt   = gt_arc[i]
        p_pred = pred_arc[nearest_indices[i]]
        color  = cmap(norm(signed_dists[i]))
        ax.plot([p_gt[0], p_pred[0]], [p_gt[1], p_pred[1]],
                color=color, linewidth=1.2, alpha=0.9, zorder=4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Signed Distance (mm)\n+ over-seg  |  − under-seg', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.legend(loc='upper right', fontsize=8, framealpha=0.7)

if __name__ == "__main__":
    args = get_args()
    total_start = time.time()

    dirs_to_make = ["WHISKERS/ZOOMED", "WHISKERS/FULLSIZE", "EXCEL_METRICS"]
    out_paths = {d: os.path.join(args.remote_out_dir, d) for d in dirs_to_make}
    for p in out_paths.values(): os.makedirs(p, exist_ok=True)

    print("Loading Annotation Coordinates and Conversion Factors...")
    points_dict = load_all_points(args.coords_excel)
    
    conv_df = pd.read_excel(args.conversion_excel)
    conv_dict = {}
    for _, r in conv_df.iterrows():
        fname = str(r['Filename']).strip()
        
        if fname.startswith("outline_"):
            fname = fname.replace("outline_", "")
        fname = os.path.splitext(fname)[0]
            
        conv_dict[fname] = float(r['mm/pixels'])

    slice_dirs = sorted(glob.glob(os.path.join(args.mc_dir, "*")))
    if not slice_dirs: raise ValueError(f"No MC folders found in {args.mc_dir}")

    for slice_dir in tqdm(slice_dirs, desc="Processing Slices"):
        if not os.path.isdir(slice_dir): continue
        
        slice_name = os.path.basename(slice_dir)
        
        if slice_name not in points_dict:
            print(f"\nSkipping {slice_name}: No points in Excel.")
            continue
        if slice_name not in conv_dict:
            print(f"\nSkipping {slice_name}: No conversion factor in Excel.")
            continue
            
        p1, p2, p3 = points_dict[slice_name]
        conv_factor = float(conv_dict[slice_name])

        sample_paths = sorted(glob.glob(os.path.join(slice_dir, "*.png")))
        if not sample_paths:
            sample_paths = sorted(glob.glob(os.path.join(slice_dir, "*.bmp")))
        if not sample_paths:
            continue
        samples_list = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) / 255.0 for p in sample_paths]
        mean_prob_map = np.mean(samples_list, axis=0)
        pred_bin = (mean_prob_map > args.threshold).astype(np.uint8)

        gt_path = os.path.join(args.mask_dir, f"{slice_name}.bmp")
        img_path = os.path.join(args.img_dir, f"{slice_name}.bmp")
        
        if not os.path.exists(gt_path): continue
        
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        bg_img = cv2.imread(img_path)
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB) if bg_img is not None else np.ones_like(cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2RGB)) * 255
        
        gt_contour = get_contour_from_array(gt_mask)
        pred_contour = get_contour_from_array(pred_bin)
        
        if gt_contour is None or pred_contour is None: continue

        gt_inner = get_inner_arc(gt_contour, p1, p2, p3)
        pred_inner = get_inner_arc(pred_contour, p1, p2, p3)
        
        dists_mm, signs, nearest_indices = calculate_signed_distances_mm(gt_inner, pred_inner, pred_bin, conv_factor)

        df_metrics = pd.DataFrame({
            'Point_Index': range(len(gt_inner)),
            'X': gt_inner[:, 0], 'Y': gt_inner[:, 1],
            'Abs_Dist_mm': dists_mm, 'Sign': signs, 'Signed_Dist_mm': dists_mm * signs
        })
        df_metrics.to_excel(os.path.join(out_paths["EXCEL_METRICS"], f"{slice_name}_metrics.xlsx"), index=False)

        all_points = np.vstack((gt_inner, pred_inner))
        min_x, min_y = np.min(all_points, axis=0)
        max_x, max_y = np.max(all_points, axis=0)
        pad = 40
        zoom_xlim = (min_x - pad, max_x + pad)
        zoom_ylim = (max_y + pad, min_y - pad)  # inverted Y

        img_h, img_w = bg_img.shape[:2]

        for variant, xlim, ylim, folder_key in [
            ("zoomed",   zoom_xlim,    zoom_ylim,    "WHISKERS/ZOOMED"),
            ("fullsize", (0, img_w),   (img_h, 0),   "WHISKERS/FULLSIZE"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(bg_img)
            plot_whiskers_with_legend(ax, fig, gt_inner, pred_inner, dists_mm, signs, nearest_indices)
            ax.invert_yaxis()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.axis('off')
            ax.set_title(f"{slice_name} — Whisker Overlay ({variant})", fontsize=10)
            fig.savefig(
                os.path.join(out_paths[folder_key], f"{slice_name}_whisker_{variant}.png"),
                bbox_inches='tight', pad_inches=0.1, dpi=150,
            )
            plt.close(fig)

    print(f"\n All slices processed and saved to {args.remote_out_dir}")
    print(f"Total Time: {(time.time() - total_start) / 60:.2f} mins.")