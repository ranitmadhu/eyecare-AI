"""
Explainable AI: Dual-Eye Grad-CAM Attention Heatmap Generator.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple


def compute_gradcam(
    img_tensor: tf.Tensor,
    model: keras.Model,
    eye_name: str = "left_eye_input"
) -> np.ndarray:
    """
    Computes Grad-CAM attention heatmap for a single eye input tensor of shape (1, H, W, 3).
    Finds the last 4D convolutional layer in the backbone and extracts gradients.
    """
    # Locate shared backbone submodel or convolutional layers
    backbone = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) or "efficientnet" in layer.name or "densenet" in layer.name:
            backbone = layer
            break

    if backbone is None:
        return np.zeros((img_tensor.shape[1], img_tensor.shape[2]), dtype=np.float32)

    # Locate last convolutional layer inside backbone
    target_layer = None
    for l in reversed(backbone.layers):
        if len(l.output_shape) == 4:
            target_layer = l
            break

    if target_layer is None:
        return np.zeros((img_tensor.shape[1], img_tensor.shape[2]), dtype=np.float32)

    grad_model = keras.Model(
        inputs=backbone.input,
        outputs=[target_layer.output, backbone.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_features = grad_model(img_tensor)
        # Pass through pooling layer
        gap = model.get_layer("shared_gap")(base_features)
        bn = model.get_layer("shared_bn")(gap)

        if eye_name == "left_eye_input":
            dense = model.get_layer("left_dense")(bn)
            pred = model.get_layer("left_eye_pred")(dense)
        else:
            dense = model.get_layer("right_dense")(bn)
            pred = model.get_layer("right_eye_pred")(dense)

        top_score = tf.reduce_max(pred)

    grads = tape.gradient(top_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0.0)

    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0.0:
        heatmap /= max_val
    return heatmap.numpy()


def overlay_cam(img_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blends Jet colormap Grad-CAM heatmap over preprocessed fundus image."""
    h, w = img_rgb.shape[:2]
    h_resized = cv2.resize(heatmap, (w, h))
    h_uint8 = np.uint8(255 * h_resized)
    jet_cam = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)
    jet_cam = cv2.cvtColor(jet_cam, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1.0 - alpha, jet_cam, alpha, 0)
