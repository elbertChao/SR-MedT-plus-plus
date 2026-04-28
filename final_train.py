import argparse
import torch
import lib
import torchvision
from torchinfo import summary
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import torch.nn.init as init
from utils_gray import JointTransform2D, ImageToImage2D, Image2D
from metrics import jaccard_index, f1_score, LogNLLLoss, classwise_f1, classwise_iou, CombinedDiceFocalLoss
import cv2
from functools import partial
from random import randint
import timeit
import time
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
import copy
import glob
from tqdm import tqdm
import random
import re
import torch.optim.lr_scheduler as lr_scheduler

# Try importing custom models
try:
    from lib.models.TransAttUnet.TransAttUnet import UNet_Attention_Transformer_Multiscale
except ImportError:
    pass

try:
    from lib.models.SwinUnet.vision_transformer import SwinUnet
except ImportError:
    pass

try:
    from lib.models.UNetPlusPlus.unetplusplus import UNetPlusPlus
except ImportError:
    print("Warning: UNetPlusPlus architecture not found.")

# METRIC_FREQ = 5
ACCUMULATION_STEPS = 2  # effective batch = batch_size * 2

def get_args():
    parser = argparse.ArgumentParser(description='Flexible Training Script with Logging')
    parser.add_argument('--modelname', type=str, default='MedT',
                        choices=['MedT', 'TransAttUnet', 'SwinUnet', 'axialunet', 'gatedaxialunet', 'logo', 'UnetPlusPlus'])
    parser.add_argument('--direc', type=str, default='./Tuned_MedT_results')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1.36e-3)
    parser.add_argument('--weight_decay', type=float, default=3.35e-5)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--seed', type=int, default=3000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--imgsize', type=int, default=256)
    parser.add_argument('--gray', type=str, default='yes', choices=['yes', 'no'])
    parser.add_argument('--cuda', type=str, default='on')
    parser.add_argument('--aug', type=str, default='on')
    parser.add_argument('--img_path', type=str, default='local_dataset/Images_contrast_enhanced')
    parser.add_argument('--mask_path', type=str, default='local_dataset/Masks')

    return parser.parse_args()

args = get_args()


HARD_CASES = [
    "Pt 0121", "Pt 0102", "Pt 0096", "Pt 0095", "Pt 0091", 
    "Pt 0086", "Pt 0075", "Pt 0064", "Pt 0062", "Pt 0057", 
    "Pt 0054", "Pt 0050", "Pt 0039", "Pt 0026", "Pt 0005"
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(args.seed)

gray_ = args.gray
aug = args.aug
direc = args.direc
modelname = args.modelname
imgsize = args.imgsize

if gray_ == "yes":
    from utils_gray import JointTransform2D, ImageToImage2D, Image2D
    imgchant = 1
else:
    from utils import JointTransform2D, ImageToImage2D, Image2D
    imgchant = 3

crop = None 

if not os.path.exists(direc): os.makedirs(direc)

aspect_check_dir = os.path.join(direc, "aspect_ratio_checks")
os.makedirs(aspect_check_dir, exist_ok=True)

aug_save_dir = os.path.join(direc, "augmented_samples")
os.makedirs(aug_save_dir, exist_ok=True)
print(f" Augmented samples will be saved to: {aug_save_dir}")

tf_train      = JointTransform2D(crop=crop, train=True,  long_mask=True, img_size=imgsize)
tf_train_hard = JointTransform2D(crop=crop, train=True,  long_mask=True, img_size=imgsize, hard_case_mode=True) 
tf_val        = JointTransform2D(crop=crop, train=False, long_mask=True, img_size=imgsize)

device = torch.device("cuda" if args.cuda == "on" else "cpu")
print(f" Using device: {device}")

class DeepSupervisionLossWrapper(nn.Module):
    def __init__(self, base_criterion, weights=[1.0, 0.5, 0.25, 0.125]):
        super().__init__()
        self.base_criterion = base_criterion
        self.weights = weights

    def forward(self, y_out, y):
        if isinstance(y_out, (tuple, list)):
            total_loss = 0
            for i, out in enumerate(y_out):
                #print(f"  DS output {i}: {out.shape}, target: {y.shape}")
                weight = self.weights[i] if i < len(self.weights) else 0.0
                if weight == 0.0:
                    continue
                if out.shape[2:] != y.shape[1:]:
                    y_resized = F.interpolate(
                        y.float().unsqueeze(1),
                        size=out.shape[2:],
                        mode='nearest'
                    ).squeeze(1).long()
                else:
                    y_resized = y
                total_loss += weight * self.base_criterion(out, y_resized)
            return total_loss
        else:
            return self.base_criterion(y_out, y)

base_criterion = CombinedDiceFocalLoss(
    dice_weight=0.5,
    focal_weight=0.5,
    start_gamma=1.5,
    max_gamma=2.56,
    # UNet++ already has deep supervision providing strong early gradients, so dynamic gamma isn't needed
    dynamic_gamma=(args.modelname != 'UnetPlusPlus'),
    class_weights=[0.2, 0.8],
    apply_softmax=(args.modelname != 'UnetPlusPlus')
)

criterion = DeepSupervisionLossWrapper(base_criterion, weights=[1.0, 0.3, 0.1, 0.0])

class DynamicKFoldDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, hard_transform=None, 
                 hard_flags=None, gray=False):
        self.image_paths   = image_paths
        self.mask_paths    = mask_paths
        self.transform     = transform
        self.hard_transform = hard_transform
        self.hard_flags    = hard_flags if hard_flags is not None else np.zeros(len(image_paths), dtype=bool)
        self.gray          = gray
        self.img_size      = transform.img_size if hasattr(transform, 'img_size') else (256, 256)
        self.long_mask     = transform.long_mask if hasattr(transform, 'long_mask') else True

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img      = Image.open(img_path).convert('L') if self.gray else Image.open(img_path).convert('RGB')
        mask     = Image.open(self.mask_paths[idx])
        filename = os.path.basename(img_path)

        tf = (self.hard_transform 
              if (self.hard_flags[idx] and self.hard_transform is not None) 
              else self.transform)
        if tf:
            img, mask = tf(img, mask)
        return img, mask, filename

def compute_metrics(model, dataloader, device, desc=""):
    model.eval()
    iou_scores, f1_scores = [], []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"  {desc} Metrics", leave=False)
        for X_batch, y_batch, *_ in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                y_out = model(X_batch)
                if isinstance(y_out, tuple): y_out = y_out[0]
            iou = classwise_iou(y_out, y_batch)
            f1 = classwise_f1(y_out, y_batch)
            if iou.numel() > 1:
                iou = iou[1:]
                f1 = f1[1:]
            iou_scores.append(iou.mean().item())
            f1_scores.append(f1.mean().item())
    model.train() 
    return np.mean(iou_scores), np.mean(f1_scores)

global_start_time = time.time()

print(f"Loading files from {args.img_path}...")
img_exts = ['.bmp', '.png', '.jpg']
all_image_paths_raw = sorted([os.path.join(args.img_path, f) for f in os.listdir(args.img_path) if os.path.splitext(f)[1].lower() in img_exts])
all_mask_paths_raw = sorted([os.path.join(args.mask_path, f) for f in os.listdir(args.mask_path) if os.path.splitext(f)[1].lower() in img_exts])

if len(all_image_paths_raw) == 0:
    print(f"Error: No images found.")
    exit()

def extract_key(filename):
    match = re.search(r'(Pt \d{4}).*?(\d{1,3}-\d{1,3})\..*$', filename)
    if match: return (match.group(1), match.group(2))
    return None

mask_map = {}
for path in all_mask_paths_raw:
    key = extract_key(os.path.basename(path))
    if key: mask_map[key] = path

paired_image_paths, paired_mask_paths = [], []
for path in all_image_paths_raw:
    key = extract_key(os.path.basename(path))
    if key and key in mask_map:
        paired_image_paths.append(path)
        paired_mask_paths.append(mask_map[key])

temp_all_image_paths = np.array(paired_image_paths)
temp_all_mask_paths = np.array(paired_mask_paths)

def get_patient_id(filename):
    match = re.search(r'(Pt \d{4})', filename)
    if match: return match.group(1)
    return None

csv_path = "fixed_test_patients.csv"
if os.path.exists(csv_path):
    test_df = pd.read_csv(csv_path)
    col_name = 'Patient_ID' if 'Patient_ID' in test_df.columns else test_df.columns[0]
    fixed_test_patients = test_df[col_name].unique().tolist()
    print(f" Excluding {len(fixed_test_patients)} patients found in fixed_test_patients.csv")
else:
    print(" Warning: fixed_test_patients.csv not found. Using all patients for Cross-Validation.")
    fixed_test_patients = []

filtered_image_paths = []
filtered_mask_paths = []

for img, mask in zip(temp_all_image_paths, temp_all_mask_paths):
    pt_id = get_patient_id(os.path.basename(img))
    if pt_id not in fixed_test_patients:
        filtered_image_paths.append(img)
        filtered_mask_paths.append(mask)

all_image_paths = np.array(filtered_image_paths)
all_mask_paths = np.array(filtered_mask_paths)

print(f" Training Pool: {len(all_image_paths)} slices (Filtered from {len(temp_all_image_paths)} original)")

patient_groups = [get_patient_id(os.path.basename(p)) for p in all_image_paths]
patient_groups_array = np.array(patient_groups)
unique_patients = np.unique(patient_groups_array)

patient_labels = np.array([1 if pt in HARD_CASES else 0 for pt in unique_patients])

kfold_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
all_fold_results = []

print("Calculating Model Complexity...")
try:
    dummy_model = None
    if modelname == "axialunet": dummy_model = lib.models.axialunet(img_size=imgsize, imgchan=imgchant)
    elif modelname == "MedT": dummy_model = lib.models.axialnet.MedT(img_size=imgsize, imgchan=imgchant)
    elif modelname == "TransAttUnet": dummy_model = UNet_Attention_Transformer_Multiscale(n_channels=imgchant, n_classes=2)
    elif modelname == "SwinUnet": dummy_model = SwinUnet(img_size=imgsize, num_classes=2, imgchan=imgchant)
    elif modelname == "UnetPlusPlus": 
        dummy_model = UNetPlusPlus(
            in_channels=imgchant,
            num_classes=2,
            deep_supervision=True
        )
    else: exit()

    dummy_input = torch.randn(1, imgchant, imgsize, imgsize)
    model_stats = summary(dummy_model, input_data=dummy_input, verbose=0)
    
    total_params = model_stats.total_params
    trainable_params = model_stats.trainable_params
    total_mult_adds = model_stats.total_mult_adds 
    
    print(f"  Total Params: {total_params:,}")
    print(f"  Trainable Params: {trainable_params:,}")
    print(f"  FLOPs: {total_mult_adds:,}")
    print(f"  MACs: {total_mult_adds/2:,}")
    
    del dummy_model
except Exception as e:
    print(f"Warning: Could not calculate FLOPs: {e}")
    total_params = 0
    total_mult_adds = 0

for fold, (train_patient_indices, val_patient_indices) in enumerate(kfold_splitter.split(unique_patients, patient_labels)):
    fold_num = fold + 1
    print(f"\n  Training Fold {fold_num}/5")
    set_seed(args.seed + fold_num) 

    train_patients_active = unique_patients[train_patient_indices]
    val_patients_active = unique_patients[val_patient_indices]

    print(f"    [Split Info] Train Patients ({len(train_patients_active)}): {train_patients_active.tolist()}")
    print(f"    [Split Info] Val Patients   ({len(val_patients_active)}): {val_patients_active.tolist()}")
    
    leakage_check = [p for p in train_patients_active if p in fixed_test_patients]
    if leakage_check:
        print(f"    CRITICAL WARNING: LEAKAGE DETECTED! Test patients found in train: {leakage_check}")
    else:
        print("    Verified: 0 fixed test patients are in this training split.")

    train_idx = np.isin(patient_groups_array, train_patients_active)
    val_idx = np.isin(patient_groups_array, val_patients_active)

    train_x_paths, train_y_paths = all_image_paths[train_idx], all_mask_paths[train_idx]
    val_x_paths, val_y_paths = all_image_paths[val_idx], all_mask_paths[val_idx]
    
    extra_x, extra_y, extra_flags = [], [], []
    for i, path in enumerate(train_x_paths):
        if get_patient_id(os.path.basename(path)) in HARD_CASES:
            extra_x.extend([path] * 2)
            extra_y.extend([train_y_paths[i]] * 2)
            extra_flags.extend([True, True])  

    base_flags = np.zeros(len(train_x_paths), dtype=bool)

    if extra_x:
        train_x_paths = np.concatenate([train_x_paths, np.array(extra_x)])
        train_y_paths = np.concatenate([train_y_paths, np.array(extra_y)])
        hard_flags    = np.concatenate([base_flags, np.array(extra_flags)])
    else:
        hard_flags = base_flags

    train_loader = DataLoader(
        DynamicKFoldDataset(train_x_paths, train_y_paths,
                            transform=tf_train, hard_transform=tf_train_hard,
                            hard_flags=hard_flags, gray=(gray_ == "yes")),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(
        DynamicKFoldDataset(val_x_paths, val_y_paths, transform=tf_val, gray=(gray_ == "yes")),
        batch_size=4,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True
    )

    if modelname == "axialunet": model = lib.models.axialunet(img_size=imgsize, imgchan=imgchant)
    elif modelname == "MedT": model = lib.models.axialnet.MedT(img_size=imgsize, imgchan=imgchant)
    elif modelname == "gatedaxialunet": model = lib.models.axialnet.gated(img_size=imgsize, imgchan=imgchant)
    elif modelname == "logo": model = lib.models.axialnet.logo(img_size=imgsize, imgchan=imgchant)
    elif modelname == "TransAttUnet": model = UNet_Attention_Transformer_Multiscale(n_channels=imgchant, n_classes=2)
    elif modelname == "SwinUnet": model = SwinUnet(img_size=imgsize, num_classes=2, imgchan=imgchant)
    elif modelname == "UnetPlusPlus": 
        model = UNetPlusPlus(
            in_channels=imgchant,
            num_classes=2,
            deep_supervision=True
        )
    else: exit()


    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    scaler = torch.amp.GradScaler(
        'cuda',
        enabled=(device.type == 'cuda'),
        init_scale=2.**14,     # lower than default 2^16 — reduces early overflow risk
        growth_factor=1.5,
        backoff_factor=0.5,
        growth_interval=200
    )

    train_losses, val_losses, train_ious, val_ious, train_f1s, val_f1s = [], [], [], [], [], []
    best_val_loss = float('inf')
    best_epoch_iou, best_epoch_f1 = 0.0, 0.0
    epochs_no_improve = 0
    best_model_path = os.path.join(direc, "checkpoints", f"best_model_fold{fold_num}.pth")
    if not os.path.exists(os.path.dirname(best_model_path)): os.makedirs(os.path.dirname(best_model_path))

    for epoch in range(args.epochs):
        current_gamma = criterion.base_criterion.update_gamma(epoch, args.epochs)

        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"F{fold_num} E{epoch+1}", leave=False)
        for step, (X, y, _) in enumerate(pbar):
            X, y = X.to(device), y.to(device)
            
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, 
                                    enabled=(device.type == 'cuda')):
                y_out = model(X)
                loss = criterion(y_out, y) / ACCUMULATION_STEPS
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n NaN/Inf at step {step}, epoch {epoch+1}")
                print(f"  Input  — nan: {torch.isnan(X).any()}, inf: {torch.isinf(X).any()}")
                print(f"  Output — nan: {torch.isnan(y_out).any()}, inf: {torch.isinf(y_out).any()}")
                for name, param in model.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        print(f"  NaN/Inf grad in: {name}")
                break

            scaler.scale(loss).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * ACCUMULATION_STEPS  
            pbar.set_postfix(loss=f"{loss.item() * ACCUMULATION_STEPS:.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y, _ in val_loader:
                X, y = X.to(device), y.to(device)
                
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                    y_out = model(X)
                
                if isinstance(y_out, (tuple, list)):
                    y_final = y_out[0].float()
                else:
                    y_final = y_out.float()
                
                val_loss += base_criterion(y_final, y).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        tr_iou, tr_f1   = compute_metrics(model, train_loader, device, "Train")
        val_iou, val_f1 = compute_metrics(model, val_loader,   device, "Val")
        
        train_ious.append(tr_iou); val_ious.append(val_iou)
        train_f1s.append(tr_f1); val_f1s.append(val_f1)
        
        print(f"  F{fold_num} E{epoch+1} | TL: {avg_train_loss:.4f} VL: {avg_val_loss:.4f} | VIoU: {val_iou:.4f} VF1: {val_f1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            best_epoch_iou = val_iou
            best_epoch_f1 = val_f1
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"    Early STOPPING at epoch {epoch+1}")
            break

        if (epoch % args.save_freq) == 0 and epoch > 0:
            viz_dir = os.path.join(direc, f"fold_{fold_num}_epoch_{epoch}")
            os.makedirs(viz_dir, exist_ok=True)
            model.eval()
            with torch.no_grad():
                for i, (X, y, fnames) in enumerate(val_loader):
                    if i >= 5: break
                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                        y_out = model(X.to(device))
                        if isinstance(y_out, tuple): y_out = y_out[0]
                        pred = torch.argmax(y_out, dim=1)[0].cpu().numpy()
                    cv2.imwrite(os.path.join(viz_dir, fnames[0]), (pred*255).astype(np.uint8))

    # Plots
    plot_dir = os.path.join(direc, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label="Train Loss"); plt.plot(val_losses, label="Val Loss")
    plt.legend(); plt.savefig(os.path.join(plot_dir, f"fold_{fold_num}_loss.png")); plt.close()
    
    plt.figure()
    plt.plot(train_ious, label="Train IoU"); plt.plot(val_ious, label="Val IoU", linestyle='--')
    plt.legend(); plt.savefig(os.path.join(plot_dir, f"fold_{fold_num}_metrics.png")); plt.close()

    all_fold_results.append((best_epoch_iou, best_epoch_f1))

total_duration = time.time() - global_start_time
hours = int(total_duration // 3600)
minutes = int((total_duration % 3600) // 60)

iou_list = [x[0] for x in all_fold_results]
f1_list = [x[1] for x in all_fold_results]

final_mean_iou = np.mean(iou_list)
final_std_iou = np.std(iou_list)
final_mean_f1 = np.mean(f1_list)
final_std_f1 = np.std(f1_list)

print("\n--- Final Results ---")
print(f"Mean IoU: {final_mean_iou:.4f} ± {final_std_iou:.4f}")
print(f"Mean F1:  {final_mean_f1:.4f} ± {final_std_f1:.4f}")
print(f"Time:     {hours}h {minutes}m")
print(f"Params:   {total_params:,}")
print(f"FLOPs:    {total_mult_adds:,}")

results_data = []
for i, (iou, f1) in enumerate(all_fold_results):
    results_data.append({
        'Fold': f"Fold {i+1}", 
        'Best Val IoU': iou, 
        'Best Val F1': f1,
        'Model': modelname,
        'Params': total_params,
        'Trainable Params': trainable_params,
        'FLOPs': total_mult_adds,
        'Training Time (s)': total_duration
    })

df = pd.DataFrame(results_data)

df.loc[len(df)] = ['Mean', final_mean_iou, final_mean_f1, modelname, total_params, trainable_params, total_mult_adds, total_duration]
df.loc[len(df)] = ['StdDev', final_std_iou, final_std_f1, '', '', '', '', '']

csv_path = os.path.join(direc, "final_5_fold_cv_results_detailed.csv")
df.to_csv(csv_path, index=False)
print(f"Detailed results saved to: {csv_path}")