# SR-MedT++

A medical image segmentation project extending the **Medical Transformer (MedT)** backbone with ensemble training, improved loss formulation, and uncertainty estimation.

## Overview

SR-MedT++ performs binary segmentation on contrast-enhanced medical images. It supports multiple transformer-based and hybrid segmentation architectures:

| Model            | Description                                                                |
| ---------------- | -------------------------------------------------------------------------- |
| `MedT`           | Medical Transformer with Gated Axial-Attention and LoGo training (default) |
| `axialunet`      | Axial-Attention U-Net (standard, non-gated)                                |
| `gatedaxialunet` | Axial-Attention U-Net with learnable gating factors                        |
| `logo`           | LoGo local-global training variant                                         |
| `TransAttUnet`   | U-Net with bottleneck transformer attention                                |
| `SwinUnet`       | Swin Transformer U-Net                                                     |
| `UnetPlusPlus`   | Nested U-Net++ with deep supervision                                       |

The default model is **MedT**, which runs two parallel branches — one over the full image and one over 64×64 patches — then combines them at the end.

### Key Features

- **5-fold stratified cross-validation** at the patient level; hard cases are oversampled 2× with stronger augmentation
- **CombinedDiceFocalLoss** — Dice + Focal with Focal gamma annealed from 1.5 → 2.56 over the first 60% of training
- **Deep supervision** with output weights `[1.0, 0.3, 0.1, 0.0]`
- **Mixed-precision training** with gradient accumulation (effective batch size doubles)
- **Ensemble inference** — probability maps from all 5 fold checkpoints are averaged
- **MC Dropout uncertainty** — optional; runs multiple stochastic forward passes and saves variance maps
- **Morphological post-processing** — connected-component selection with per-patient centroid anchoring
- **Layer activation heatmaps** via forward hooks

---

## Environment Setup

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate medt_env
```

> `mkl=2023.1.0` is pinned to fix a known `iJIT_NotifyEvent` crash on Windows. `numpy<2.0` avoids breaking changes introduced in NumPy 2.

### pip

```bash
pip install -r requirements.txt
```

For SwinUnet support, additionally install:

```bash
pip install -r swin_requirements.txt
```

**Tested with:** Python 3.10, PyTorch with CUDA 12.1

---

## Dataset Preparation

Organize your data in the following directory layout. Images and their corresponding segmentation masks must share the same filename stem.

```
/path/to/dataset/
├── images/
│   ├── Pt_0001_slice_001.png
│   ├── Pt_0001_slice_002.png
│   └── ...
└── masks/
    ├── Pt_0001_slice_001.png
    ├── Pt_0001_slice_002.png
    └── ...
```

- **Images**: grayscale or RGB `.png` files (resized to 256×256 during loading)
- **Masks**: binary `.png` files — pixel values must be `0` (background) or `255` (foreground)
- **Filename convention**: filenames must include a patient ID matching the pattern `Pt XXXX` (e.g., `Pt 0001`) — the data loader uses this for patient-level CV splits

You also need a `fixed_test_patients.csv` listing the patient IDs reserved for the fixed test set. Place it in the same directory as `--direc`.

---

## Training

Run 5-fold cross-validation training with:

```bash
python final_train.py \
  --modelname MedT \
  --img_path /path/to/dataset/images \
  --mask_path /path/to/dataset/masks \
  --direc ./results \
  --epochs 400 \
  --batch_size 4 \
  --learning_rate 1.36e-3 \
  --weight_decay 3.35e-5 \
  --imgsize 256 \
  --gray yes \
  --aug on \
  --patience 50 \
  --save_freq 10 \
  --workers 8
```

### Key Arguments

| Argument          | Default           | Description                                                             |
| ----------------- | ----------------- | ----------------------------------------------------------------------- |
| `--modelname`     | `MedT`            | Architecture to train (see table above)                                 |
| `--img_path`      | `/path/to/images` | Path to image directory                                                 |
| `--mask_path`     | `/path/to/masks`  | Path to mask directory                                                  |
| `--direc`         | `./results`       | Output directory for checkpoints and logs                               |
| `--epochs`        | `400`             | Maximum training epochs per fold                                        |
| `--batch_size`    | `4`               | Per-GPU batch size (effective batch=8 via 2-step gradient accumulation) |
| `--learning_rate` | `1.36e-3`         | Initial Adam learning rate                                              |
| `--weight_decay`  | `3.35e-5`         | Adam weight decay                                                       |
| `--imgsize`       | `256`             | Input image spatial size                                                |
| `--gray`          | `yes`             | `yes` = grayscale (1 channel), `no` = RGB (3 channels)                  |
| `--aug`           | `on`              | Enable joint image-mask augmentation                                    |
| `--patience`      | `50`              | Early stopping patience (epochs)                                        |
| `--save_freq`     | `10`              | Checkpoint and visualization save frequency (epochs)                    |
| `--workers`       | `8`               | DataLoader worker processes                                             |
| `--seed`          | `3000`            | Random seed for reproducibility                                         |

### Outputs

After training, `--direc` will contain:

```
results/
├── checkpoints/
│   ├── best_model_fold1.pth
│   ├── best_model_fold2.pth
│   ├── best_model_fold3.pth
│   ├── best_model_fold4.pth
│   └── best_model_fold5.pth
├── fold_metrics.csv          # Per-fold IoU/F1 + mean ± std
├── loss_curves_fold*.png
└── sample_predictions_fold*.png
```

---

## Testing

Run ensemble inference over the fixed test set with:

```bash
python final_test.py \
  --modelname MedT \
  --direc ./results \
  --img_path /path/to/dataset/images \
  --mask_path /path/to/dataset/masks \
  --imgsize 256 \
  --gray yes \
  --workers 4
```

Loads all five fold checkpoints from `--direc/checkpoints/` and averages their probability maps per image.

### Optional Flags

| Flag              | Description                                                    |
| ----------------- | -------------------------------------------------------------- |
| `--postprocess`   | Morphological closing + spatial consistency post-processing    |
| `--uncertainty`   | MC Dropout uncertainty estimation                              |
| `--mc_samples 10` | Stochastic forward passes per model (requires `--uncertainty`) |
| `--heatmaps`      | Save layer activation heatmaps                                 |

### Outputs

```
results/
├── ENSEMBLE_PREDICTIONS/        # binary segmentation masks
├── ENSEMBLE_PROB_MAPS/          # probability maps (JET colormap)
├── POST_P_PREDICTIONS/          # post-processed masks (--postprocess)
├── ENSEMBLE_UNCERTAINTY/        # uncertainty maps (--uncertainty)
├── MC_SAMPLES/                  # per-model dropout samples (--uncertainty)
├── ENSEMBLE_HEATMAPS/           # layer activation maps (--heatmaps)
└── test_metrics.csv
```

---

## Hyperparameter Optimization

To search for optimal hyperparameters using Optuna:

```bash
python optuna_optimization.py \
  --modelname MedT \
  --img_path /path/to/dataset/images \
  --mask_path /path/to/dataset/masks \
  --direc ./optuna_results \
  --n_trials 100
```

Results go into `--direc/MedT_HPO.db` (SQLite). Parameter importance is plotted to `MedT_parameter_importance.png`.

---

## Repository Structure

```
SR-MedT_plus_plus/
├── final_train.py                  # 5-fold CV training script
├── final_test.py                   # Ensemble inference script
├── optuna_optimization.py          # Hyperparameter optimization
├── metrics.py                      # Loss functions (Dice, Focal, Combined)
├── utils.py                        # RGB dataset and augmentation utilities
├── utils_gray.py                   # Grayscale dataset and augmentation utilities
├── extractors.py                   # Auxiliary backbone extractors (ResNet/DenseNet/SqueezeNet)
├── outlier_check.py                # Visualize worst-performing predictions
├── feature_importance_plots.py     # Plot Optuna parameter importance
├── environment.yml                 # Conda environment
├── requirements.txt                # pip dependencies
├── swin_requirements.txt           # SwinUnet additional dependencies
└── lib/
    ├── models/
    │   ├── axialnet.py             # MedT, axialunet, gatedaxialunet, logo
    │   ├── memory_efficient_attention.py
    │   ├── TransAttUnet/
    │   │   └── TransAttUnet.py
    │   ├── SwinUnet/
    │   │   └── vision_transformer.py
    │   └── UNetPlusPlus/
    │       └── unetplusplus.py
    └── datasets/
```

---

## Backbone: Medical Transformer (MedT)

The Gated Axial-Attention U-Net and LoGo training strategy come from the **Medical Transformer** paper:

> Valanarasu, J. M. J., Oza, P., Hacihaliloglu, I., & Patel, V. M. (2021).
> **Medical Transformer: Gated Axial-Attention for Medical Image Segmentation.**
> In _Medical Image Computing and Computer Assisted Intervention – MICCAI 2021_,
> Lecture Notes in Computer Science, vol. 12901, pp. 36–46.
> Springer, Cham. https://doi.org/10.1007/978-3-030-87193-2_4

BibTeX:

```bibtex
@InProceedings{valanarasu2021medical,
  author    = {Valanarasu, Jeya Maria Jose and Oza, Poojan and Hacihaliloglu, Ilker and Patel, Vishal M.},
  title     = {Medical Transformer: Gated Axial-Attention for Medical Image Segmentation},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2021},
  year      = {2021},
  publisher = {Springer International Publishing},
  address   = {Cham},
  pages     = {36--46},
  isbn      = {978-3-030-87193-2},
  doi       = {10.1007/978-3-030-87193-2_4}
}
```

Axial attention code adapted from [axial-deeplab](https://github.com/csrhddlam/axial-deeplab).

---

## Dataset Access

The dataset used to train SR-MedT++ is private. To request access, contact Robarts Research Institute, London, Ontario.

---

## License

See [LICENSE](LICENSE).
