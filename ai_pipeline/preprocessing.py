"""
Ophthalmic Preprocessing Pipeline.

Implements:
1. Field-Of-View (FOV) circular retina boundary detection and black border cropping.
2. Aspect-ratio preserving letterbox padding.
3. Ben Graham local illumination subtraction (normalizes flash variances across fundus cameras).
4. Circular aperture masking.
"""

import cv2
import numpy as np
import tensorflow as tf


def crop_to_fov(img: np.ndarray, thresh: int = 10) -> np.ndarray:
    """Finds circular retina boundary and crops out empty black margins."""
    if img is None or img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    mask = gray > thresh
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]


def resize_pad(img: np.ndarray, size: int) -> np.ndarray:
    """Aspect-ratio preserving resize with centered letterbox padding."""
    if img is None or img.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def clean_fundus_np(img_uint8: np.ndarray, target_size: int = 384, use_ben_graham: bool = True) -> np.ndarray:
    """
    Complete clinical fundus normalization pipeline:
    FOV Crop -> Letterbox Resize -> Ben Graham Local Normalization -> Circular Aperture Mask.
    """
    img = img_uint8.numpy() if isinstance(img_uint8, tf.Tensor) else img_uint8
    img = crop_to_fov(img)
    img = resize_pad(img, target_size)

    if use_ben_graham:
        blur = cv2.GaussianBlur(img, (0, 0), 15)
        img = cv2.addWeighted(img, 4.0, blur, -4.0, 128)
    else:
        img = cv2.bilateralFilter(img, d=5, sigmaColor=35, sigmaSpace=35)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)

    # Circular mask to remove background edge noise
    mask = np.zeros((target_size, target_size), dtype=np.uint8)
    center = (target_size // 2, target_size // 2)
    cv2.circle(mask, center, int(target_size * 0.485), 255, -1)
    img = cv2.bitwise_and(img, img, mask=mask)
    return img


def preprocess_image_path(image_path: str, target_size: int = 384, use_ben_graham: bool = True) -> np.ndarray:
    """Reads image file from disk and applies clinical normalization."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image from: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return clean_fundus_np(rgb, target_size=target_size, use_ben_graham=use_ben_graham)
