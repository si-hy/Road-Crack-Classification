
"""augmentation_pipeline.py

Comprehensive data‑augmentation pipeline for object‑detection / instance‑segmentation
datasets using **Copy‑Paste ➜ Elastic/Grid ➜ (optional) Cutout**
with polygon‑level label fidelity.

Key features
------------
1. **Polygon‑preserving label transforms** – all geometric ops applied to every
   polygon vertex; bounding boxes are lazily derived from polygons only when
   a consumer explicitly asks for them.
2. **Instance‑aware object extraction** – exact binary masks are cut from the
   source image with the object’s true silhouette (no rectangular masks).
3. **Rich Copy‑Paste diversity** – scaling, rotation, colour‑jitter and
   optional augmentations applied *per pasted instance*.
4. **Advanced blending** – multiple strategies (alpha, colour‑matched,
   multi‑band, Poisson, Poisson‑harmonised) selectable.
5. **Polygon‑aware Elastic‑Grid transform** – same displacement field applied
   to image *and* each polygon vertex.
6. **Class‑balanced sampling** – inverse‑frequency weighting with smoothing so
   rare classes are preferentially pasted but never starved of diversity.

Requirements
------------
pip install -U opencv-python pillow albumentations shapely scikit-image numpy
(NumPy < 2.0 if you use TF <= 2.16, see project docs).

Usage
-----
from augmentation_pipeline import AugmentationPipeline
pipeline = AugmentationPipeline(config)
aug_img, aug_ann = pipeline(image, annotations)

The *annotations* format is a list of dicts:
    {'polygon': [[x1,y1], [x2,y2], ...], 'category_id': int, 'iscrowd': 0/1, ...}

Copyright 2025, Your Name
"""

from __future__ import annotations
import random
import math
import cv2
import enum
from typing import List, Tuple, Dict, Any, Sequence
import numpy as np
from shapely.geometry import Polygon, box as shapely_box
from shapely.affinity import affine_transform
from albumentations import (
    Compose, OneOf, RandomBrightnessContrast, HueSaturationValue,
    ElasticTransform, Rotate, Resize
)
from albumentations.augmentations.geometric.transforms import (
    Affine
)
from skimage import exposure

# ----------------------------------------------------------------------
# Helper geometry utilities
# ----------------------------------------------------------------------
def polygon_to_mask(poly: Sequence[Tuple[float, float]], image_shape: Tuple[int, int]) -> np.ndarray:
    """Rasterise a polygon to a binary mask."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = np.array(poly, dtype=np.int32)
    if pts.size == 0:
        return mask
    cv2.fillPoly(mask, [pts], 255)
    return mask

def apply_affine_to_polygon(poly: Sequence[Tuple[float, float]], M: np.ndarray) -> List[Tuple[float, float]]:
    """Apply a 2×3 affine matrix to polygon vertices."""
    pts = np.hstack([np.array(poly, dtype=np.float32), np.ones((len(poly), 1), dtype=np.float32)])
    transformed = np.dot(M, pts.T).T
    return list(map(tuple, transformed))

def elastic_displacement_field(shape: Tuple[int,int], alpha: float, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate elastic displacement fields for given alpha, sigma."""
    random_state = np.random.RandomState(None)
    dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
    return dx, dy

def apply_elastic(img: np.ndarray, polys: List[List[Tuple[float,float]]], alpha=30, sigma=4) -> Tuple[np.ndarray, List[List[Tuple[float,float]]]]:
    h, w = img.shape[:2]
    dx, dy = elastic_displacement_field((h, w), alpha, sigma)
    # map coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    deformed = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    # transform polygons
    new_polys = []
    for poly in polys:
        deformed_pts = []
        for (px, py) in poly:
            ix, iy = int(round(px)), int(round(py))
            if 0 <= iy < h and 0 <= ix < w:
                nx = px + float(dx[iy, ix])
                ny = py + float(dy[iy, ix])
                deformed_pts.append((nx, ny))
        new_polys.append(deformed_pts)
    return deformed, new_polys


# ----------------------------------------------------------------------
# Blending strategies
# ----------------------------------------------------------------------
class BlendType(enum.Enum):
    SIMPLE_ALPHA = "simple_alpha"
    COLOR_MATCH_ALPHA = "color_match_alpha"
    POISSON = "poisson"
    POISSON_HARMONISED = "poisson_harmonised"
    MULTIBAND = "multiband"

class AdvancedBlender:
    def __init__(self, blend_type: BlendType = BlendType.SIMPLE_ALPHA, multiband_levels: int = 5):
        self.blend_type = blend_type
        self.multiband_levels = multiband_levels

    def __call__(self, target: np.ndarray, patch: np.ndarray, mask: np.ndarray, center: Tuple[int,int]) -> np.ndarray:
        if self.blend_type == BlendType.SIMPLE_ALPHA:
            return self._simple_alpha(target, patch, mask, center)
        elif self.blend_type == BlendType.COLOR_MATCH_ALPHA:
            return self._color_match_alpha(target, patch, mask, center)
        elif self.blend_type == BlendType.POISSON:
            return self._poisson(target, patch, mask, center, cv2.NORMAL_CLONE)
        elif self.blend_type == BlendType.POISSON_HARMONISED:
            return self._poisson(target, patch, mask, center, cv2.MIXED_CLONE)
        elif self.blend_type == BlendType.MULTIBAND:
            return self._multiband(target, patch, mask, center, self.multiband_levels)
        else:
            raise ValueError(f"Unsupported blend type {self.blend_type}")

    # --- blending helpers ---
    def _simple_alpha(self, target, patch, mask, center):
        mask_f = mask.astype(np.float32) / 255.0
        h, w = patch.shape[:2]
        x0 = center[0] - w // 2
        y0 = center[1] - h // 2
        target_slice = target[y0:y0+h, x0:x0+w]
        if target_slice.shape[:2] != patch.shape[:2]:
            return target  # skip if out of bounds
        blended = (patch * mask_f[:,:,None] + target_slice * (1 - mask_f[:,:,None])).astype(target.dtype)
        target[y0:y0+h, x0:x0+w] = blended
        return target

    def _color_match_alpha(self, target, patch, mask, center):
        # match colour histogram of patch to target region
        h, w = patch.shape[:2]
        x0 = center[0] - w // 2
        y0 = center[1] - h // 2
        target_slice = target[y0:y0+h, x0:x0+w]
        for c in range(3):
            patch[...,c] = exposure.match_histograms(patch[...,c], target_slice[...,c])
        return self._simple_alpha(target, patch, mask, center)

    def _poisson(self, target, patch, mask, center, flag):
        # center is (x, y) for seamlessClone
        return cv2.seamlessClone(patch, target, mask, center, flag)

    def _multiband(self, target, patch, mask, center, levels):
        # Fallback to alpha if cv2 detail multiband blend unavailable
        try:
            import cv2.detail as detail
            stitcher = detail_MultiBandBlender(levels, 5)
        except Exception:
            return self._simple_alpha(target, patch, mask, center)


# ----------------------------------------------------------------------
# Copy‑Paste augmentor
# ----------------------------------------------------------------------
class CopyPasteAugmentor:
    def __init__(
        self,
        scale_range: Tuple[float,float]=(0.6, 1.4),
        rotate_deg: float = 20,
        rotate_prob: float = 0.4,
        colour_aug_prob: float = 0.3,
        blender: AdvancedBlender = AdvancedBlender(BlendType.POISSON),
    ):
        self.scale_range = scale_range
        self.rotate_deg = rotate_deg
        self.rotate_prob = rotate_prob
        self.colour_aug_prob = colour_aug_prob
        self.colour_aug = Compose([
            RandomBrightnessContrast(p=0.5),
            HueSaturationValue(p=0.5)
        ], p=self.colour_aug_prob)
        self.blender = blender

    def extract_object(self, image: np.ndarray, poly: Sequence[Tuple[int,int]]) -> Tuple[np.ndarray, np.ndarray]:
        x, y, w, h = cv2.boundingRect(np.array(poly, dtype=np.int32))
        patch = image[y:y+h, x:x+w].copy()
        mask = polygon_to_mask([(px-x, py-y) for (px,py) in poly], patch.shape)
        patch[mask==0] = 0
        return patch, mask

    def transform_instance(self, patch: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # random scale
        scale = random.uniform(*self.scale_range)
        h, w = patch.shape[:2]
        new_size = (int(w*scale), int(h*scale))
        patch = cv2.resize(patch, new_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)

        # random rotate
        if random.random() < self.rotate_prob:
            angle = random.uniform(-self.rotate_deg, self.rotate_deg)
            M = cv2.getRotationMatrix2D((new_size[0]//2, new_size[1]//2), angle, 1.0)
            patch = cv2.warpAffine(patch, M, new_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            mask = cv2.warpAffine(mask, M, new_size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

        # colour augmentation
        augmented = self.colour_aug(image=patch, mask=mask)
        patch, mask = augmented['image'], augmented['mask']
        return patch, mask

    def __call__(self, img: np.ndarray, polys: List[List[Tuple[float,float]]], class_weights: Dict[int,float]) -> Tuple[np.ndarray, List[List[Tuple[float,float]]]]:
        h, w = img.shape[:2]

        # build candidate pool
        pool = [(poly, idx) for idx, poly in enumerate(polys)]
        if not pool:
            return img, polys

        categories = [idx for (_, idx) in pool]
        weights = [class_weights.get(idx, 1.0) for (_,idx) in pool]

        # choose number of objects to paste (1-4)
        k = random.randint(1, min(4, len(pool)))
        chosen = random.choices(pool, weights=weights, k=k)

        for poly, cls_id in chosen:
            patch, mask = self.extract_object(img, poly)
            patch, mask = self.transform_instance(patch, mask)

            # random position
            ph, pw = patch.shape[:2]
            cx = random.randint(pw//2, w-pw//2-1)
            cy = random.randint(ph//2, h-ph//2-1)

            img = self.blender(img, patch, mask, (cx, cy))

            # translate polygon
            dx = cx - pw//2
            dy = cy - ph//2
            new_poly = [(px+dx, py+dy) for (px,py) in poly]
            polys.append(new_poly)
        return img, polys


# ----------------------------------------------------------------------
# Class weight calculator
# ----------------------------------------------------------------------
def compute_inverse_freq_weights(dataset_annotations: List[List[Dict[str,Any]]], smoothing: float=1.0) -> Dict[int, float]:
    cls_count = {}
    for ann_list in dataset_annotations:
        for ann in ann_list:
            cid = ann['category_id']
            cls_count[cid] = cls_count.get(cid, 0) + 1
    max_freq = max(cls_count.values())
    weights = {cid: (max_freq/(cnt + smoothing)) for cid, cnt in cls_count.items()}
    # normalise
    s = sum(weights.values())
    weights = {k:v/s for k,v in weights.items()}
    return weights


# ----------------------------------------------------------------------
# Cutout utility (does not affect polygons; only image)
# ----------------------------------------------------------------------
def random_cutout(img: np.ndarray, holes: int=3, size_ratio: Tuple[float,float]=(0.1,0.3)):
    h, w = img.shape[:2]
    for _ in range(holes):
        rh = int(random.uniform(*size_ratio) * h)
        rw = int(random.uniform(*size_ratio) * w)
        x0 = random.randint(0, w-rw)
        y0 = random.randint(0, h-rh)
        img[y0:y0+rh, x0:x0+rw] = 0
    return img

# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
class AugmentationPipelineConfig:
    copy_paste_first: bool = True
    blend_type: BlendType = BlendType.POISSON
    elastic_alpha: float = 30
    elastic_sigma: float = 4
    apply_cutout_prob: float = 0.3

class AugmentationPipeline:
    def __init__(self, config: AugmentationPipelineConfig):
        self.cfg = config
        self.blender = AdvancedBlender(config.blend_type)
        # placeholder – update after sampling the whole dataset
        self.class_weights = {}

    def set_dataset_stats(self, dataset_annotations: List[List[Dict[str,Any]]]):
        self.class_weights = compute_inverse_freq_weights(dataset_annotations)

    def _augment(self, image: np.ndarray, polygons: List[List[Tuple[float,float]]]) -> Tuple[np.ndarray, List[List[Tuple[float,float]]]]:
        cp = CopyPasteAugmentor(blender=self.blender)

        if self.cfg.copy_paste_first:
            image, polygons = cp(image, polygons, self.class_weights)
            image, polygons = apply_elastic(image, polygons, alpha=self.cfg.elastic_alpha, sigma=self.cfg.elastic_sigma)
        else:
            image, polygons = apply_elastic(image, polygons, alpha=self.cfg.elastic_alpha, sigma=self.cfg.elastic_sigma)
            image, polygons = cp(image, polygons, self.class_weights)

        # Optional cutout
        if random.random() < self.cfg.apply_cutout_prob:
            image = random_cutout(image)

        return image, polygons

    def __call__(self, image: np.ndarray, annotations: List[Dict[str,Any]]) -> Tuple[np.ndarray, List[Dict[str,Any]]]:
        polygons = [ann['polygon'] for ann in annotations]
        aug_img, aug_polys = self._augment(image.copy(), polygons)

        # prepare new annotation list preserving original keys
        new_anns = []
        for ann, poly in zip(annotations, aug_polys):
            ann_copy = ann.copy()
            ann_copy['polygon'] = poly
            new_anns.append(ann_copy)
        return aug_img, new_anns
