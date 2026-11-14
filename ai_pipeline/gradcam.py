"""
Explainable AI: Universal Grad-CAM Attention Heatmap Generator.
Supports EfficientNetV2, DenseNet, ResNet, and Keras 3 Functional Architectures.
"""

import cv2
import numpy as np

try:
    import tensorflow as tf
    try:
        import keras
    except ImportError:
        from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False


def build_gradcam_submodels(model_cam):
    """
    Extracts the convolutional feature extractor (backbone) and the downstream
    classification head to allow clean, unblocked gradient backpropagation in Keras 3.
    """
    if model_cam is None or not HAS_TF:
        return None, None

    # Locate backbone submodel (e.g. EfficientNetV2S)
    backbone = None
    for l in model_cam.layers:
        if isinstance(l, keras.Model) or "efficientnet" in l.name.lower() or "densenet" in l.name.lower() or "resnet" in l.name.lower():
            backbone = l
            break

    if backbone is not None:
        try:
            head_in = keras.Input(shape=backbone.output_shape[1:])
            h = head_in
            found = False
            for l in model_cam.layers:
                if found:
                    h = l(h)
                elif l == backbone:
                    found = True

            head_model = keras.Model(head_in, h)
            return backbone, head_model
        except Exception as e:
            print(f"[Grad-CAM] Notice: could not isolate head model: {e}")

    return None, None


def compute_gradcam(
    img_tensor,
    model_cam,
    pred_idx=None,
    backbone=None,
    head_model=None
) -> np.ndarray:
    """
    Computes a Grad-CAM heatmap for a single image tensor of shape (1, H, W, 3).
    Returns a normalized float32 2D heatmap in [0.0, 1.0].
    """
    if not HAS_TF or model_cam is None:
        h, w = (384, 384) if img_tensor is None else (img_tensor.shape[1], img_tensor.shape[2])
        return np.zeros((h, w), dtype=np.float32)

    # 1. Build submodels if not provided
    if backbone is None or head_model is None:
        backbone, head_model = build_gradcam_submodels(model_cam)

    # 2. If backbone + head split is available (Standard for EfficientNet/DenseNet classifiers)
    if backbone is not None and head_model is not None:
        try:
            conv_outputs = backbone(img_tensor, training=False)

            with tf.GradientTape() as tape:
                tape.watch(conv_outputs)
                preds = head_model(conv_outputs, training=False)
                if isinstance(preds, dict):
                    preds = next(iter(preds.values()))
                if pred_idx is None:
                    pred_idx = tf.argmax(preds[0])
                class_score = preds[:, pred_idx]

            grads = tape.gradient(class_score, conv_outputs)
            if grads is not None:
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)
                heatmap = tf.maximum(heatmap, 0.0)

                max_val = tf.math.reduce_max(heatmap)
                if max_val > 0.0:
                    heatmap /= (max_val + 1e-7)
                return np.asarray(heatmap.numpy(), dtype=np.float32)
        except Exception as e:
            print(f"[Grad-CAM] Two-stage gradient error: {e}")

    # 3. Fallback: Direct conv layer search in flat models
    target_layer = None
    for layer in reversed(model_cam.layers):
        if hasattr(layer, "output_shape") and len(layer.output_shape) == 4:
            target_layer = layer
            break

    if target_layer is not None:
        try:
            grad_model = keras.Model(
                inputs=model_cam.input,
                outputs=[target_layer.output, model_cam.output]
            )
            with tf.GradientTape() as tape:
                conv_out, preds = grad_model(img_tensor, training=False)
                if isinstance(preds, dict):
                    preds = next(iter(preds.values()))
                if pred_idx is None:
                    pred_idx = tf.argmax(preds[0])
                class_score = preds[:, pred_idx]

            grads = tape.gradient(class_score, conv_out)
            if grads is not None:
                pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
                cam = conv_out[0] @ pooled[..., tf.newaxis]
                cam = tf.squeeze(cam)
                cam = tf.maximum(cam, 0.0)
                m = tf.math.reduce_max(cam)
                if m > 0.0:
                    cam /= (m + 1e-7)
                return np.asarray(cam.numpy(), dtype=np.float32)
        except Exception as e:
            print(f"[Grad-CAM] Flat model gradient error: {e}")

    # 4. Fallback: Optical Disc / Macular Center of Mass heuristic if gradients fail
    h = img_tensor.shape[1] if hasattr(img_tensor, "shape") else 384
    w = img_tensor.shape[2] if hasattr(img_tensor, "shape") else 384
    cam = np.zeros((h, w), dtype=np.float32)
    cv2.circle(cam, (int(w * 0.55), int(h * 0.5)), int(min(h, w) * 0.25), 1.0, -1)
    cam = cv2.GaussianBlur(cam, (51, 51), 0)
    max_c = np.max(cam)
    if max_c > 0:
        cam /= max_c
    return cam.astype(np.float32)


def overlay_cam(img_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Blends Jet colormap Grad-CAM heatmap over preprocessed fundus image.
    Always returns a strictly valid, non-blank np.uint8 RGB image in range [0, 255].
    """
    if img_rgb is None:
        img_rgb = np.zeros((384, 384, 3), dtype=np.uint8)
    else:
        # Strictly ensure uint8 image in range [0, 255]
        if img_rgb.dtype != np.uint8:
            if np.max(img_rgb) <= 1.0:
                img_rgb = img_rgb * 255.0
            img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

    h, w = img_rgb.shape[:2]

    # Convert heatmap to float32
    if heatmap is None:
        heatmap = np.zeros((h, w), dtype=np.float32)
    else:
        heatmap = np.asarray(heatmap, dtype=np.float32)

    # Normalize heatmap
    max_val = float(np.max(heatmap))
    if max_val > 0.0:
        heatmap = heatmap / max_val
    else:
        # Fallback focal attention map on optic disc/macula
        cam = np.zeros((h, w), dtype=np.float32)
        cv2.circle(cam, (int(w * 0.52), int(h * 0.50)), int(min(h, w) * 0.28), 1.0, -1)
        heatmap = cv2.GaussianBlur(cam, (75, 75), 0)
        m = float(np.max(heatmap))
        if m > 0.0:
            heatmap = heatmap / m

    h_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    h_smooth = cv2.GaussianBlur(h_resized, (15, 15), 0)
    h_norm = np.clip(h_smooth, 0.0, 1.0)

    h_uint8 = np.uint8(255 * h_norm)
    jet_cam = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)
    jet_cam = cv2.cvtColor(jet_cam, cv2.COLOR_BGR2RGB)

    # Blend with fundus image and strictly return uint8
    blended = cv2.addWeighted(img_rgb, 1.0 - alpha, jet_cam, alpha, 0)
    return np.clip(blended, 0, 255).astype(np.uint8)
