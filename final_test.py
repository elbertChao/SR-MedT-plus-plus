import argparse
import torch
import lib
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os
import pandas as pd
import cv2
import glob
from tqdm import tqdm
import random
import re
import time
from utils_gray import JointTransform2D 
from torch.utils.data.dataset import Dataset
import matplotlib
matplotlib.use('Agg')

def get_args():
    parser = argparse.ArgumentParser(description='Ensemble Test Script on Fixed CSV')
    parser.add_argument('--modelname', type=str, default='MedT',
                        choices=['MedT', 'TransAttUnet', 'axialunet', 'gatedaxialunet', 'SwinUnet', 'UnetPlusPlus'])
    parser.add_argument('--direc', type=str, required=True,
                        help='Directory containing "checkpoints" and "fixed_test_patients.csv"')
    parser.add_argument('--img_path', type=str, default='local_dataset/Images_contrast_enhanced')
    parser.add_argument('--mask_path', type=str, default='local_dataset/Masks')
    parser.add_argument('--heatmaps', action='store_true',
                        help='Generate layer-wise heatmaps (fold 1 model only)')
    parser.add_argument('--uncertainty', action='store_true',
                        help='Enable MC Dropout uncertainty estimation')
    parser.add_argument('--postprocess', action='store_true',
                        help='Apply morphological closing + spatial consistency post-processing')
    parser.add_argument('--mc_samples', type=int, default=10,
                        help='MC forward passes per model (total = 5 * mc_samples)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--imgsize', type=int, default=256)
    parser.add_argument('--gray', type=str, default='yes', choices=['yes', 'no'])
    parser.add_argument('--seed', type=int, default=3000)
    parser.add_argument('--workers', type=int, default=4)

    return parser.parse_args()


def set_mc_dropout(model):
    # keep BN in eval so running stats don't shift during stochastic passes
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()

def set_standard_eval(model):
    model.eval()

class DynamicKFoldDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, gray=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.gray = gray
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        if self.gray: img = Image.open(img_path).convert('L')
        else: img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        original_size = img.size
        filename = os.path.basename(img_path)
        
        if self.transform: 
            img, mask = self.transform(img, mask)
        
        return img, mask, filename, original_size

def extract_key_from_filename(filename):
    match = re.search(r'(Pt \d{4}).*?(\d{1,3}-\d{1,3})\..*$', filename)
    if match: return (match.group(1), match.group(2))
    else: return None

def get_patient_id(filename):
    match = re.search(r'(Pt \d{4})', filename)
    if match: return match.group(1)
    return None

def create_legend_strip(height, top_label, bottom_label):
    width = 60
    gradient = np.linspace(0, 255, height, dtype=np.uint8)[::-1]
    legend_grad = np.tile(gradient, (width, 1)).T
    legend_color = cv2.applyColorMap(legend_grad, cv2.COLORMAP_JET)
    
    cv2.rectangle(legend_color, (0, 0), (width, 20), (0,0,0), -1)
    cv2.rectangle(legend_color, (0, height-20), (width, height), (0,0,0), -1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.35
    thickness = 1
    color = (255, 255, 255)
    
    cv2.putText(legend_color, top_label, (5, 15), font, scale, color, thickness, cv2.LINE_AA)
    cv2.putText(legend_color, bottom_label, (5, height - 5), font, scale, color, thickness, cv2.LINE_AA)
    return legend_color

def generate_heatmap_from_tensor(activation_tensor, img_size=256):
    heatmap_tensor = activation_tensor.squeeze(0).mean(dim=0)
    heatmap_np = heatmap_tensor.cpu().numpy()

    if heatmap_np.shape != (img_size, img_size):
        heatmap_resized = cv2.resize(heatmap_np, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    else:
        heatmap_resized = heatmap_np

    heatmap_normalized = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    heatmap_coloured = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    return heatmap_coloured

def heatmap_overlay(base_img_np, heatmap_coloured, alpha=0.4):
    return cv2.addWeighted(base_img_np, 1 - alpha, heatmap_coloured, alpha, 0)

if __name__ == "__main__":
    args = get_args()
    total_start = time.time()

    models_dir = os.path.join(args.direc, "checkpoints")
    csv_path = os.path.join(args.direc, "fixed_test_patients.csv")
    
    if args.postprocess:
        final_save_dir = os.path.join(args.direc, "POST_P_PREDICTIONS")
    else:
        final_save_dir = os.path.join(args.direc, "ENSEMBLE_PREDICTIONS")
    
    os.makedirs(final_save_dir, exist_ok=True)
    
    if args.uncertainty:
        uncertainty_save_dir = os.path.join(args.direc, "ENSEMBLE_UNCERTAINTY")
        probability_save_dir = os.path.join(args.direc, "ENSEMBLE_PROB_MAPS")
        mc_samples_dir = os.path.join(args.direc, "MC_SAMPLES")
        os.makedirs(uncertainty_save_dir, exist_ok=True)
        os.makedirs(probability_save_dir, exist_ok=True)
        os.makedirs(mc_samples_dir, exist_ok=True)
        
    if args.heatmaps:
        heatmap_save_dir = os.path.join(args.direc, "ENSEMBLE_HEATMAPS")
        os.makedirs(heatmap_save_dir, exist_ok=True)

    imgchant = 1 if args.gray == "yes" else 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Device: {device}!! | Model: {args.modelname}!! | Ensemble Mode: 5 Folds")

    img_exts = ['.bmp', '.png', '.jpg']
    all_image_paths_raw = sorted([f for f in glob.glob(os.path.join(args.img_path, "*.*")) if os.path.splitext(f)[1].lower() in img_exts])
    all_mask_paths_raw = sorted([f for f in glob.glob(os.path.join(args.mask_path, "*.*")) if os.path.splitext(f)[1].lower() in img_exts])

    mask_map = {}
    for path in all_mask_paths_raw:
        key = extract_key_from_filename(os.path.basename(path))
        if key: mask_map[key] = path

    paired_image_paths = []
    paired_mask_paths = []
    for path in all_image_paths_raw:
        key = extract_key_from_filename(os.path.basename(path))
        if key and key in mask_map:
            paired_image_paths.append(path)
            paired_mask_paths.append(mask_map[key])

    all_image_paths = np.array(paired_image_paths)
    all_mask_paths = np.array(paired_mask_paths)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f" ERROR: fixed_test_patients.csv not found in {args.direc}")

    test_df = pd.read_csv(csv_path)
    col_name = 'Patient_ID' if 'Patient_ID' in test_df.columns else test_df.columns[0]
    fixed_test_patients = test_df[col_name].unique().tolist()
    print(f" Loaded {len(fixed_test_patients)} patients from CSV.")

    patient_groups = [get_patient_id(os.path.basename(p)) for p in all_image_paths]
    patient_groups_array = np.array(patient_groups)
    test_idx = np.isin(patient_groups_array, fixed_test_patients)
    
    test_x_paths = all_image_paths[test_idx]
    test_y_paths = all_mask_paths[test_idx]
    
    print(f" Processing {len(test_x_paths)} slices (Fixed Test Set).")
    
    tf_test = JointTransform2D(crop=None, train=False, long_mask=True, img_size=args.imgsize)
    test_dataset = DynamicKFoldDataset(test_x_paths, test_y_paths, transform=tf_test, gray=(args.gray == "yes"))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    ensemble_models = []
    print("\n Loading Models...")

    for fold in range(1, 6):
        model_path = os.path.join(models_dir, f"best_model_fold{fold}.pth")
        if not os.path.exists(model_path):
            print(f"   WARNING: Fold {fold} missing. Skipping.")
            continue
            
        if args.modelname == "TransAttUnet":
            from lib.models.TransAttUnet.TransAttUnet import UNet_Attention_Transformer_Multiscale
            model = UNet_Attention_Transformer_Multiscale(n_channels=imgchant, n_classes=2)
        elif args.modelname == "MedT":
            model = lib.models.axialnet.MedT(img_size=args.imgsize, imgchan=imgchant)
        elif args.modelname == "axialunet":
            model = lib.models.axialunet(img_size=args.imgsize, imgchan=imgchant)
        elif args.modelname == "gatedaxialunet":
            model = lib.models.gatedaxialunet(img_size=args.imgsize, imgchan=imgchant)
        elif args.modelname == "SwinUnet":
            from lib.models.SwinUnet.vision_transformer import SwinUnet
            model = SwinUnet(img_size=args.imgsize, num_classes=2, imgchan=imgchant)
        elif args.modelname == "UnetPlusPlus": 
            from lib.models.UNetPlusPlus.unetplusplus import UNetPlusPlus
            model = UNetPlusPlus(
                in_channels=imgchant,
                num_classes=2,
                deep_supervision=True
        )
        else:
            model = lib.models.axialnet.MedT(img_size=args.imgsize, imgchan=imgchant)

        if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
        model.to(device)
        
        state_dict = torch.load(model_path)
        if list(state_dict.keys())[0].startswith('module.'): 
             state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            print(f"   WARNING Fold {fold}: Missing keys (random init): {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"   WARNING Fold {fold}: Unexpected keys (ignored): {load_result.unexpected_keys}")
        model.eval()
        ensemble_models.append(model)
        print(f"  Fold {fold} loaded.")

    if not ensemble_models:
        raise RuntimeError("No models loaded! Check directory path.")

    activations = {}
    handles = []
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    if args.heatmaps and len(ensemble_models) > 0:
        print("   Heatmaps enabled (Using Fold 1 model for visualization)")
        model_to_hook = ensemble_models[0].module if isinstance(ensemble_models[0], nn.DataParallel) else ensemble_models[0]
        
        layers_to_hook = {}
        if args.modelname == "TransAttUnet":
            layers_to_hook = {
                'Enc_Down3': model_to_hook.down3,
                'Att_SDPA': model_to_hook.sdpa,
                'Dec_Up1': model_to_hook.up1
            }
        elif args.modelname == "MedT":
            layers_to_hook = {
                'Global_Enc1':    model_to_hook.layer1,
                'Global_Enc2':    model_to_hook.layer2,
                'Global_Dec1':    model_to_hook.decoder4,
                'Global_Dec2':    model_to_hook.decoder5,
                'Local_Enc1':     model_to_hook.layer1_p,
                'Local_Enc2':     model_to_hook.layer2_p,
                'Local_Enc3':     model_to_hook.layer3_p,
                'Local_Enc4':     model_to_hook.layer4_p,
                'Local_Dec1':     model_to_hook.decoder1_p,
                'Local_Dec2':     model_to_hook.decoder2_p,
                'Local_Dec3':     model_to_hook.decoder3_p,
                'Local_Dec4':     model_to_hook.decoder4_p,
                'Local_Dec5':     model_to_hook.decoder5_p,
                'CrossPatch_Comm': model_to_hook.cross_patch_comm,
                'Fusion_Block':   model_to_hook.fusion_block,
                'Final_Fusion':   model_to_hook.decoderf
            }
        elif args.modelname == "UnetPlusPlus":
            layers_to_hook = {
                'Bottleneck':  model_to_hook.conv5_1,
                'Final_Conv':  model_to_hook.output_4,
            }
            
        for name, layer in layers_to_hook.items():
            handles.append(layer.register_forward_hook(get_activation(name)))

    records = []
    # Per-patient centroid anchor for spatial-consistency post-processing.
    # Anchored on the first clean single-component prediction for each patient,
    # then used on ambiguous slices to pick the component closest to it.
    patient_ref_centroids = {}
    print("\n Starting Ensemble Inference...")

    with torch.no_grad():
        for X_batch, y_batch, filenames, original_sizes in tqdm(test_loader, desc="Test"):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            image_filename = filenames[0]
            image_name_base = os.path.splitext(image_filename)[0]
            orig_w, orig_h = original_sizes[0].item(), original_sizes[1].item()
            pt_id = get_patient_id(image_filename)
            
            if args.uncertainty:
                slice_samples_dir = os.path.join(mc_samples_dir, image_name_base)
                os.makedirs(slice_samples_dir, exist_ok=True)

            all_probs = []
            iters_per_model = args.mc_samples if args.uncertainty else 1
            
            for m_idx, model in enumerate(ensemble_models):
                fold_num = m_idx + 1
                if args.uncertainty:
                    set_mc_dropout(model)
                else:
                    set_standard_eval(model)
                
                for s_idx in range(iters_per_model):
                    logits = model(X_batch)
                    
                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]
                    # UNet++ output_4 has no activation, so softmax is applied uniformly here
                    probs = F.softmax(logits, dim=1)

                    p_map = probs[:, 1, :, :].cpu()
                    all_probs.append(p_map)

                    if args.uncertainty:
                        sample_np = p_map[0].numpy()
                        sample_resized = cv2.resize(sample_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                        sample_filename = f"fold{fold_num}_sample{s_idx}.png"
                        cv2.imwrite(os.path.join(slice_samples_dir, sample_filename), (sample_resized * 255).astype(np.uint8))

            all_probs_stack = torch.stack(all_probs)
            mean_prob = all_probs_stack.mean(dim=0)
            
            prob_np = mean_prob[0].numpy()
            prob_resized = cv2.resize(prob_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            pred_mask_np = (prob_resized > 0.5).astype(np.uint8)

            if args.postprocess and pred_mask_np.sum() > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                pred_mask_np = cv2.morphologyEx(pred_mask_np, cv2.MORPH_CLOSE, kernel)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask_np, connectivity=8)
                # label 0 is background; 1 = one component; >2 = ambiguous
                if num_labels == 1:
                    pred_mask_np = np.zeros_like(pred_mask_np)
                elif num_labels == 2:
                    pred_mask_np = (labels == 1).astype(np.uint8)
                    patient_ref_centroids[pt_id] = (centroids[1][0], centroids[1][1])
                else:
                    fg_labels = list(range(1, num_labels))
                    if pt_id not in patient_ref_centroids:
                        # no anchor yet — fall back to largest component
                        best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    else:
                        ref_cx, ref_cy = patient_ref_centroids[pt_id]
                        distances = [
                            (centroids[lbl][0] - ref_cx) ** 2 + (centroids[lbl][1] - ref_cy) ** 2
                            for lbl in fg_labels
                        ]
                        best_label = fg_labels[int(np.argmin(distances))]
                    pred_mask_np = (labels == best_label).astype(np.uint8)
                    patient_ref_centroids[pt_id] = (centroids[best_label][0], centroids[best_label][1])

            pred_out_path = os.path.join(final_save_dir, image_filename)
            if not os.path.exists(pred_out_path):
                cv2.imwrite(pred_out_path, (pred_mask_np * 255).astype(np.uint8))
            # skip if already written by a prior deterministic run

            if args.uncertainty:
                std_dev = torch.std(all_probs_stack, dim=0)
                unc_np = std_dev[0].numpy()
                unc_resized = cv2.resize(unc_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                unc_norm = (unc_resized - unc_resized.min()) / (unc_resized.max() - unc_resized.min() + 1e-8)
                unc_color = cv2.applyColorMap((unc_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                legend_unc = create_legend_strip(orig_h, "High", "Low")
                cv2.imwrite(os.path.join(uncertainty_save_dir, f"{image_name_base}_unc.bmp"), np.hstack((unc_color, legend_unc)))

                prob_viz = (prob_resized * 255).astype(np.uint8)
                prob_color = cv2.applyColorMap(prob_viz, cv2.COLORMAP_JET)
                legend_prob = create_legend_strip(orig_h, "100%", "0%")
                cv2.imwrite(os.path.join(probability_save_dir, f"{image_name_base}_prob.bmp"), np.hstack((prob_color, legend_prob)))

            if args.heatmaps:
                set_standard_eval(ensemble_models[0])
                _ = ensemble_models[0](X_batch)
                
                img_tensor = X_batch.squeeze(0).squeeze(0)
                img_viz = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
                orig_img_bgr = cv2.cvtColor((img_viz.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                
                heatmap_out_dir = os.path.join(heatmap_save_dir, image_name_base)
                os.makedirs(heatmap_out_dir, exist_ok=True)
                
                for name, act in activations.items():
                    if isinstance(act, list): act = act[0]
                    hm_color = generate_heatmap_from_tensor(act, img_size=args.imgsize)
                    overlay = heatmap_overlay(orig_img_bgr, hm_color)
                    cv2.imwrite(os.path.join(heatmap_out_dir, f"{name}.bmp"), overlay)

                activations = {}

            y_true = (y_batch.cpu() > 0).float()
            pred_tensor = (mean_prob > 0.5).float()
            intersection = (pred_tensor * y_true).sum()
            dice = (2. * intersection) / (pred_tensor.sum() + y_true.sum() + 1e-6)
            iou = (intersection) / (pred_tensor.sum() + y_true.sum() - intersection + 1e-6)
            
            records.append({
                'filename': image_filename,
                'patient_id': pt_id,
                'dice': dice.item(),
                'iou': iou.item()
            })

    if args.heatmaps:
        for h in handles: h.remove()

    total_end = time.time()
    total_duration = total_end - total_start
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = total_duration % 60

    print(f"\nTotal Inference Time: {total_duration:.2f} seconds ({total_duration/len(test_loader):.2f} sec/image)")

    # separate filename for MC runs so deterministic results aren't overwritten
    if len(records) > 0:
        df = pd.DataFrame(records)
        csv_filename = "ensemble_metrics_mc.csv" if args.uncertainty else "ensemble_metrics_fixed.csv"
        save_path = os.path.join(args.direc, csv_filename)
        df.to_csv(save_path, index=False)
        print(f"\n Ensemble Complete!!!")
        print(f"   Predictions: {final_save_dir}")
        print(f"   Metrics:     {save_path}")
        print(f"   Avg Dice:    {df['dice'].mean():.4f}")
        print(f"   Avg IoU:     {df['iou'].mean():.4f}")
        
        print("-" * 30)
        print(f" Total Execution Time: {hours}h {minutes}m {seconds:.2f}s")
        print("-" * 30)
        
        if args.uncertainty:
            print(f"   Uncertainty Maps: {uncertainty_save_dir}")
            print(f"   Probability Maps:  {probability_save_dir}")
            print(f"   MC Samples saved by slice in: {mc_samples_dir}")
        if args.heatmaps:
            print(f"   Heatmaps: {heatmap_save_dir}")
    else:
        print("No records to save! Check if test set is empty or if there were errors during inference.")