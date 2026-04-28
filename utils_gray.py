import os
import numpy as np
import torch

from skimage import io,color
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from typing import Callable
import os
import cv2
import pandas as pd
import re
import random

from numbers import Number
from typing import Container
from collections import defaultdict


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if img is None:
            raise ValueError("cv2.imread returned None. Check file paths and file integrity.")
        
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class JointTransform2D:
    """
    Synchronized augmentation for image-mask pairs.
    hard_case_mode applies more aggressive transforms for difficult patients.
    """
    def __init__(self, crop=None, long_mask=False, train=True, img_size=None, hard_case_mode=False):
        self.crop = crop
        self.long_mask = long_mask
        self.train = train
        self.hard_case_mode = hard_case_mode
        
        if img_size:
            self.img_size = (img_size, img_size)
        else:
            self.img_size = None

        if self.train:
            self.color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            self.gaussian_blur = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0))
            self.color_jitter_hard = T.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.3, hue=0.15)
            self.gaussian_blur_hard = T.GaussianBlur(kernel_size=(7, 11), sigma=(0.5, 6.0))


    def __call__(self, image, mask):
        if self.img_size:
            image = F.resize(image, self.img_size, interpolation=F.InterpolationMode.BILINEAR)
            mask  = F.resize(mask,  self.img_size, interpolation=F.InterpolationMode.NEAREST)

        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)

        if self.train:
            # probabilities scaled up for hard cases
            p = 0.7 if self.hard_case_mode else 0.5
            r = 0.6 if self.hard_case_mode else 0.3
            rot_range = 45.0 if self.hard_case_mode else 20.0
            zoom_range = (0.8, 1.2) if self.hard_case_mode else (0.9, 1.1)
            shear_range = 25 if self.hard_case_mode else 15
            shift_pct = 0.15 if self.hard_case_mode else 0.1

            if random.random() < p:
                image, mask = F.hflip(image), F.hflip(mask)
            if random.random() < p:
                image, mask = F.vflip(image), F.vflip(mask)

            if random.random() < r:
                width, height = image.size
                dx = random.randint(-int(shift_pct * width),  int(shift_pct * width))
                dy = random.randint(-int(shift_pct * height), int(shift_pct * height))
                image = F.affine(image, angle=0, translate=(dx, dy), scale=1.0, shear=0,
                                 interpolation=F.InterpolationMode.BILINEAR, fill=0)
                mask  = F.affine(mask,  angle=0, translate=(dx, dy), scale=1.0, shear=0,
                                 interpolation=F.InterpolationMode.NEAREST,  fill=0)

            if random.random() < r:
                angle = random.uniform(-rot_range, rot_range)
                image = F.affine(image, angle=angle, translate=(0,0), scale=1.0, shear=0,
                                 interpolation=F.InterpolationMode.BILINEAR, fill=0)
                mask  = F.affine(mask,  angle=angle, translate=(0,0), scale=1.0, shear=0,
                                 interpolation=F.InterpolationMode.NEAREST,  fill=0)

            if random.random() < r:
                zoom_factor = random.uniform(*zoom_range)
                image = F.affine(image, angle=0, translate=(0,0), scale=zoom_factor, shear=0,
                                 interpolation=F.InterpolationMode.BILINEAR, fill=0)
                mask  = F.affine(mask,  angle=0, translate=(0,0), scale=zoom_factor, shear=0,
                                 interpolation=F.InterpolationMode.NEAREST,  fill=0)

            if random.random() < r:
                shear_angle = random.uniform(-shear_range, shear_range)
                image = F.affine(image, angle=0, translate=(0,0), scale=1.0, shear=shear_angle,
                                 interpolation=F.InterpolationMode.BILINEAR, fill=0)
                mask  = F.affine(mask,  angle=0, translate=(0,0), scale=1.0, shear=shear_angle,
                                 interpolation=F.InterpolationMode.NEAREST,  fill=0)

            elastic_prob = 0.6 if self.hard_case_mode else 0.2
            if random.random() < elastic_prob:
                image, mask = self._elastic_deform(image, mask)

            jitter = self.color_jitter_hard if self.hard_case_mode else self.color_jitter
            blur   = self.gaussian_blur_hard if self.hard_case_mode else self.gaussian_blur

            if random.random() < r + 0.2:
                image = jitter(image)
            if random.random() < r:
                gamma = random.uniform(0.6, 1.6) if self.hard_case_mode else random.uniform(0.7, 1.5)
                image = F.adjust_gamma(image, gamma)
            if random.random() < r:
                image = blur(image)

        image = F.to_tensor(image)

        if self.train:
            noise_sigma = random.uniform(0.01, 0.15) if self.hard_case_mode else random.uniform(0.01, 0.1)
            if random.random() < (0.5 if self.hard_case_mode else 0.3):
                noise = torch.randn_like(image) * noise_sigma
                image = torch.clamp(image + noise, 0.0, 1.0)

        if self.long_mask:
            mask_np = np.array(mask, dtype=np.uint8)
            mask_np[mask_np > 0] = 1
            mask = torch.from_numpy(mask_np).long()
        else:
            mask = F.to_tensor(mask)

        return image, mask

    def _elastic_deform(self, image, mask):
        img_np  = np.array(image,  dtype=np.float32)
        mask_np = np.array(mask,   dtype=np.float32)

        # grayscale PIL images are (H, W), not (H, W, C)
        if img_np.ndim == 2:
            h, w = img_np.shape
        else:
            h, w = img_np.shape[:2]

        alpha = random.uniform(30, 60)
        sigma = random.uniform(4, 6)

        dx = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha

        x, y  = np.meshgrid(np.arange(w), np.arange(h))
        map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
        map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)

        img_warped  = cv2.remap(img_np,  map_x, map_y, cv2.INTER_LINEAR,  borderMode=cv2.BORDER_REFLECT)
        mask_warped = cv2.remap(mask_np, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        # remap interpolation drifts values away from 0/255 — re-threshold
        mask_warped = (mask_warped >= 0.5).astype(np.uint8) * 255
        img_warped = np.clip(img_warped, 0, 255).astype(np.uint8)

        return Image.fromarray(img_warped), Image.fromarray(mask_warped)

class ImageToImage2D(Dataset):
    def _extract_key(self, filename):
        patient_regex = re.compile(r"^(Pt \d+)")
        slice_regex = re.compile(r"(\d+-\d+-\d+)\.bmp$")
        
        patient_match = patient_regex.match(filename)
        slice_match = slice_regex.search(filename)
        
        if patient_match and slice_match:
            patient_id = patient_match.group(1)
            slice_id = slice_match.group(1)
            return f"{patient_id}_{slice_id}"
        return None

    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'images')
        self.output_path = os.path.join(dataset_path, 'masks')
        
        image_map = {}
        try:
            image_files = [f for f in os.listdir(self.input_path) if f.lower().endswith(".bmp")]
            for f in image_files:
                key = self._extract_key(f)
                if key:
                    image_map[key] = os.path.join(self.input_path, f)
        except FileNotFoundError:
            image_files = []
            print(f"Warning: 'images' directory not found at {self.input_path}")

        mask_map = {}
        try:
            mask_files = [f for f in os.listdir(self.output_path) if f.lower().endswith(".bmp")]
            for f in mask_files:
                key = self._extract_key(f)
                if key:
                    mask_map[key] = os.path.join(self.output_path, f)
        except FileNotFoundError:
            mask_files = []
            print(f"Warning: 'masks' directory not found at {self.output_path}")

        common_keys = sorted(list(set(image_map.keys()).intersection(set(mask_map.keys()))))
        self.file_list = []
        for key in common_keys:
            self.file_list.append((image_map[key], mask_map[key]))

        if len(image_map) != len(mask_map) or len(image_map) != len(self.file_list):
            print(f"Warning: Mismatch in {dataset_path}.")
            print(f"  Found {len(image_map)} valid images and {len(mask_map)} valid masks with matching keys.")
            print(f"  Using {len(self.file_list)} common file pairs.")
        
        self.one_hot_mask = one_hot_mask

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_path, mask_path = self.file_list[idx]
        image_filename = os.path.basename(image_path)
        
        image = cv2.imread(image_path, 0)
        mask = cv2.imread(mask_path, 0) 
        
        if image is None:
            raise FileNotFoundError(f"Could not read image file: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Could not read mask file: {mask_path}")

        image, mask = correct_dims(image, mask)
        
        mask[mask<127] = 0
        mask[mask>=127] = 1


        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        return image, mask, image_filename


class Image2D(Dataset):

    def __init__(self, dataset_path: str, transform: Callable = None):

        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'images')
        
        self.images_list = [f for f in os.listdir(self.input_path) if f.lower().endswith(".bmp")]

        if transform:
            self.transform = transform
        else:
            self.transform = T.ToTensor()

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):

        image_filename = self.images_list[idx]
        image_path = os.path.join(self.input_path, image_filename)

        image = cv2.imread(image_path, 0)
        
        if image is None:
            raise FileNotFoundError(f"Could not read image file: {image_path}")

        image = correct_dims(image)

        image = self.transform(image)

        return image, image_filename

def chk_mkdir(*paths: Container) -> None:
    """
    Creates folders if they do not exist.
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


class Logger:
    def __init__(self, verbose=False):
        self.logs = defaultdict(list)
        self.verbose = verbose

    def log(self, logs):
        for key, value in logs.items():
            self.logs[key].append(value)

        if self.verbose:
            print(logs)

    def get_logs(self):
        return self.logs

    def to_csv(self, path):
        pd.DataFrame(self.logs).to_csv(path, index=None)


class MetricList:
    def __init__(self, metrics):
        assert isinstance(metrics, dict), '\'metrics\' must be a dictionary of callables'
        self.metrics = metrics
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def __call__(self, y_out, y_batch):
        for key, value in self.metrics.items():
            self.results[key] += value(y_out, y_batch)

    def reset(self):
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def get_results(self, normalize=False):
        assert isinstance(normalize, bool) or isinstance(normalize, Number), '\'normalize\' must be boolean or a number'
        if not normalize:
            return self.results
        else:
            return {key: value/normalize for key, value in self.results.items()}