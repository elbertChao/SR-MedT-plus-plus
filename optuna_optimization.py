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
from sklearn.model_selection import StratifiedKFold
import copy
import glob
from tqdm import tqdm
import random
import re
import torch.optim.lr_scheduler as lr_scheduler
import optuna
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc

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
    import segmentation_models_pytorch as smp
except ImportError:
    print("Warning: segmentation_models_pytorch not found.")

def get_args():
    parser = argparse.ArgumentParser(description='Flexible Training Script with Logging')
    parser.add_argument('--modelname', type=str, default='MedT', 
                        choices=['MedT', 'TransAttUnet', 'SwinUnet', 'axialunet', 'gatedaxialunet', 'logo', 'UnetPlusPlus'])
    parser.add_argument('--direc', type=str, default='./MedT_optuna_results')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=3000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--imgsize', type=int, default=256)
    parser.add_argument('--gray', type=str, default='yes', choices=['yes', 'no'])
    parser.add_argument('--cuda', type=str, default='on')
    parser.add_argument('--img_path', type=str, default='local_dataset/Images_contrast_enhanced')
    parser.add_argument('--mask_path', type=str, default='local_dataset/Masks')
    parser.add_argument('--n_trials', type=int, default=30, help='Number of Optuna trials to run')
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

if args.gray == "yes":
    from utils_gray import JointTransform2D
    imgchant = 1
else:
    from utils import JointTransform2D
    imgchant = 3

if not os.path.exists(args.direc): os.makedirs(args.direc)

tf_train = JointTransform2D(crop=None, train=True, long_mask=True, img_size=args.imgsize)
tf_val   = JointTransform2D(crop=None, train=False, long_mask=True, img_size=args.imgsize)

device = torch.device("cuda" if args.cuda == "on" else "cpu")

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
        if self.gray: img = Image.open(img_path).convert('L')
        else: img = Image.open(img_path).convert('RGB')
        mask = Image.open(self.mask_paths[idx])
        filename = os.path.basename(img_path)
        if self.transform: img, mask = self.transform(img, mask)
        return img, mask, filename

def compute_metrics(model, dataloader, device):
    model.eval()
    iou_scores = []
    with torch.no_grad():
        for X_batch, y_batch, *_ in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                y_out = model(X_batch)
            iou = classwise_iou(y_out, y_batch)
            if iou.numel() > 1:
                iou = iou[1:]
            iou_scores.append(iou.mean().item())
    model.train() 
    return np.mean(iou_scores)

print(f"Loading files from {args.img_path}...")
img_exts = ['.bmp', '.png', '.jpg']
all_image_paths_raw = sorted([os.path.join(args.img_path, f) for f in os.listdir(args.img_path) if os.path.splitext(f)[1].lower() in img_exts])
all_mask_paths_raw = sorted([os.path.join(args.mask_path, f) for f in os.listdir(args.mask_path) if os.path.splitext(f)[1].lower() in img_exts])

def extract_key(filename):
    match = re.search(r'(Pt \d{4}).*?(\d{1,3}-\d{1,3})\..*$', filename)
    if match: return (match.group(1), match.group(2))
    return None

mask_map = {extract_key(os.path.basename(p)): p for p in all_mask_paths_raw if extract_key(os.path.basename(p))}
paired_image_paths, paired_mask_paths = [], []
for path in all_image_paths_raw:
    key = extract_key(os.path.basename(path))
    if key and key in mask_map:
        paired_image_paths.append(path)
        paired_mask_paths.append(mask_map[key])

def get_patient_id(filename):
    match = re.search(r'(Pt \d{4})', filename)
    if match: return match.group(1)
    return None

csv_path = os.path.join(args.direc, "fixed_test_patients.csv")
fixed_test_patients = pd.read_csv(csv_path)[pd.read_csv(csv_path).columns[0]].unique().tolist() if os.path.exists(csv_path) else []

filtered_image_paths, filtered_mask_paths = [], []
for img, mask in zip(paired_image_paths, paired_mask_paths):
    if get_patient_id(os.path.basename(img)) not in fixed_test_patients:
        filtered_image_paths.append(img)
        filtered_mask_paths.append(mask)

all_image_paths = np.array(filtered_image_paths)
all_mask_paths = np.array(filtered_mask_paths)

patient_groups_array = np.array([get_patient_id(os.path.basename(p)) for p in all_image_paths])
unique_patients = np.unique(patient_groups_array)
patient_labels = np.array([1 if pt in HARD_CASES else 0 for pt in unique_patients])

kfold_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)


def objective(trial):
    search_lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    search_wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    search_focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)

    micro_batch_size = 2  # physical batch size to fit 12 GB VRAM
    search_effective_bs = trial.suggest_categorical("effective_batch_size", [4, 8, 16])
    accumulation_steps = search_effective_bs // micro_batch_size
    
    criterion = CombinedDiceFocalLoss(dice_weight=0.5, focal_weight=0.5, focal_gamma=search_focal_gamma, class_weights=[0.2, 0.8])
    
    # only run one fold per trial to keep HPO fast
    fold_generator = kfold_splitter.split(unique_patients, patient_labels)
    train_patient_indices, val_patient_indices = next(fold_generator)
    
    set_seed(args.seed + trial.number) 

    train_patients_active = unique_patients[train_patient_indices]
    val_patients_active = unique_patients[val_patient_indices]

    train_idx = np.isin(patient_groups_array, train_patients_active)
    val_idx = np.isin(patient_groups_array, val_patients_active)

    train_x_paths, train_y_paths = all_image_paths[train_idx], all_mask_paths[train_idx]
    val_x_paths, val_y_paths = all_image_paths[val_idx], all_mask_paths[val_idx]
    
    extra_x, extra_y = [], []
    for i, path in enumerate(train_x_paths):
        if get_patient_id(os.path.basename(path)) in HARD_CASES:
            extra_x.extend([path] * 2); extra_y.extend([train_y_paths[i]] * 2)
    if extra_x:
        train_x_paths = np.concatenate([train_x_paths, np.array(extra_x)])
        train_y_paths = np.concatenate([train_y_paths, np.array(extra_y)])

    train_loader = DataLoader(DynamicKFoldDataset(train_x_paths, train_y_paths, transform=tf_train, gray=(args.gray == "yes")), batch_size=micro_batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(DynamicKFoldDataset(val_x_paths, val_y_paths, transform=tf_val, gray=(args.gray == "yes")), batch_size=1, shuffle=False, num_workers=args.workers)

    if args.modelname == "axialunet": model = lib.models.axialunet(img_size=args.imgsize, imgchan=imgchant)
    elif args.modelname == "MedT": model = lib.models.axialnet.MedT(img_size=args.imgsize, imgchan=imgchant)
    elif args.modelname == "TransAttUnet": model = UNet_Attention_Transformer_Multiscale(n_channels=imgchant, n_classes=2)
    elif args.modelname == "SwinUnet": model = SwinUnet(img_size=args.imgsize, num_classes=2, imgchan=imgchant)
    elif args.modelname == "UnetPlusPlus": model = smp.UnetPlusPlus(encoder_name="efficientnet-b0", encoder_weights=None, in_channels=imgchant, classes=2)
    else: exit()

    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=search_lr, weight_decay=search_wd)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_loss = float('inf')
    best_epoch_iou = 0.0
    epochs_no_improve = 0

    try:
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            
            for batch_idx, (X, y, _) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)
                
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                    y_out = model(X)
                    loss = criterion(y_out, y)
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                if ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(train_loader)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X, y, _ in val_loader:
                    X, y = X.to(device), y.to(device)
                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                        val_loss += criterion(model(X), y).item()
            
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            val_iou = compute_metrics(model, val_loader, device)
            print(f"  Trial {trial.number} | Epoch {epoch+1}/{args.epochs} | Val IoU: {val_iou:.4f} | Eff. BS: {search_effective_bs}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch_iou = val_iou
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            trial.report(val_iou, epoch)
            if trial.should_prune():
                print(f"  Trial {trial.number} pruned at epoch {epoch+1}.") 
                raise optuna.exceptions.TrialPruned()

            if epochs_no_improve >= args.patience:
                break
                
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"   WARNING: Trial {trial.number} ran out of GPU memory. Safely pruning config.")
            del model, optimizer, scaler
            torch.cuda.empty_cache()
            gc.collect()
            raise optuna.exceptions.TrialPruned()
        else:
            raise e

    del model, optimizer, scaler
    torch.cuda.empty_cache()
    gc.collect()

    return best_epoch_iou


def save_live_results(study, trial):
    df = study.trials_dataframe()
    csv_path = os.path.join(args.direc, f"{args.modelname}_optuna_live_results.csv")
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    print(f"Starting Hyperparameter Optimization for {args.modelname}...")

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study_name = f"{args.modelname}_HPO"
    storage_name = f"sqlite:///{os.path.join(args.direc, study_name)}.db"

    # load_if_exists resumes an interrupted study without losing prior trials
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        pruner=pruner
    )

    study.optimize(
        objective, 
        n_trials=args.n_trials, 
        show_progress_bar=True,
        callbacks=[save_live_results]
    )
    
    print("\n=========================================")
    print("*** OPTIMIZATION FINISHED ***")
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value (Max Val IoU): {trial.value:.4f}")
    print("  Best Hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    df = study.trials_dataframe()
    csv_path = os.path.join(args.direc, f"{args.modelname}_optuna_results_FINAL.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nFinal Detailed HPO results saved to: {csv_path}")

    print("\nGenerating performance plots...")
    completed_trials = df[df['state'] == 'COMPLETE']
    
    if not completed_trials.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(completed_trials['number'], completed_trials['value'], marker='o', linestyle='-', alpha=0.7, label='Trial Value (IoU)')
        plt.plot(completed_trials['number'], completed_trials['value'].cummax(), marker='', linestyle='--', color='red', label='Best Value So Far')
        plt.xlabel('Trial Number')
        plt.ylabel('Validation IoU')
        plt.title(f'Optimization History - {args.modelname}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.direc, f"{args.modelname}_optimization_history.png"))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(completed_trials['params_learning_rate'], completed_trials['value'], c=completed_trials['number'], cmap='viridis', s=100, alpha=0.8)
        plt.xscale('log') 
        plt.colorbar(label='Trial Number (Darker = Earlier, Lighter = Later)')
        plt.xlabel('Learning Rate (Log Scale)')
        plt.ylabel('Validation IoU')
        plt.title(f'Learning Rate impact on Performance - {args.modelname}')
        plt.grid(True)
        plt.savefig(os.path.join(args.direc, f"{args.modelname}_learning_rate_impact.png"))
        plt.close()
        
        print(f"Plots saved to: {args.direc}")
    else:
        print("No completed trials to plot (all were pruned or failed).")