import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    return np.array(img)


def save_image(arr: np.ndarray, path: str) -> str:
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return path


def resize_image(
    image: np.ndarray,
    max_dim: int,
    interpolation: int = cv2.INTER_LINEAR
) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    current_max = max(h, w)

    if current_max <= max_dim:
        return image, 1.0

    scale = max_dim / current_max
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    return resized, scale


def create_thumbnail(image: np.ndarray, size: int = 256) -> np.ndarray:
    h, w = image.shape[:2]
    min_dim = min(h, w)

    # Center crop to square
    start_h = (h - min_dim) // 2
    start_w = (w - min_dim) // 2
    cropped = image[start_h:start_h + min_dim, start_w:start_w + min_dim]

    # Resize to thumbnail size
    thumbnail = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)

    return thumbnail


def normalize_image(image: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None) -> np.ndarray:
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    if std is None:
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_float = image.astype(np.float32) / 255.0
    normalized = (img_float - mean) / std

    return normalized


def denormalize_image(image: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None) -> np.ndarray:
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    if std is None:
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    denorm = (image * std + mean) * 255.0
    return denorm.clip(0, 255).astype(np.uint8)


def pad_image(image: np.ndarray, tile_size: int, pad_value: int = 0) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = image.shape[:2]
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size

    if pad_h > 0 or pad_w > 0:
        if len(image.shape) == 3:
            padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)),
                           mode='constant', constant_values=pad_value)
        else:
            padded = np.pad(image, ((0, pad_h), (0, pad_w)),
                           mode='constant', constant_values=pad_value)
        return padded, (pad_h, pad_w)

    return image, (0, 0)


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    colors: dict,
    alpha: float = 0.5,
    skip_background: bool = True
) -> np.ndarray:
    overlay = image.copy().astype(np.float32)
    colored_mask = np.zeros_like(image, dtype=np.float32)

    for class_idx, rgb in colors.items():
        if skip_background and class_idx == 0:
            continue
        class_mask = (mask == class_idx)
        colored_mask[class_mask] = rgb

    mask_present = mask > 0 if skip_background else mask >= 0
    overlay[mask_present] = (
        (1 - alpha) * overlay[mask_present] +
        alpha * colored_mask[mask_present]
    )

    return overlay.astype(np.uint8)
