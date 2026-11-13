"""
Siamese Binocular Neural Network Architecture for Bilateral Retinal Screening.

Key architectural features:
1. Weight-shared Siamese backbone (EfficientNetV2-S or DenseNet121).
2. Inter-ocular difference vector |f_OS - f_OD| modeling clinical cup-to-disc & vessel asymmetry.
3. Multi-task output heads:
   - Left Eye diagnosis (4-class softmax)
   - Right Eye diagnosis (4-class softmax)
   - Patient Overall condition (4-class softmax)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple

from config import PipelineConfig
from losses import CategoricalFocalLoss


def build_binocular_model(config: PipelineConfig) -> Tuple[keras.Model, keras.Model]:
    """
    Constructs the dual-input Siamese binocular model.
    Returns: (full_model, shared_backbone)
    """
    input_shape = config.img_size + (3,)

    left_input = keras.Input(shape=input_shape, name="left_eye_input")
    right_input = keras.Input(shape=input_shape, name="right_eye_input")

    # Shared feature extraction backbone
    backbone_choice = config.backbone.lower()
    if "efficientnet" in backbone_choice:
        shared_backbone = keras.applications.EfficientNetV2S(
            include_top=False,
            weights=config.weights,
            input_shape=input_shape
        )
    elif "densenet" in backbone_choice:
        shared_backbone = keras.applications.DenseNet121(
            include_top=False,
            weights=config.weights,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unsupported backbone: {config.backbone}")

    shared_backbone.trainable = False

    gap_layer = layers.GlobalAveragePooling2D(name="shared_gap")
    bn_layer = layers.BatchNormalization(name="shared_bn")

    # Extract features through shared Siamese branch
    feat_left = bn_layer(gap_layer(shared_backbone(left_input, training=False)))
    feat_right = bn_layer(gap_layer(shared_backbone(right_input, training=False)))

    # Inter-ocular asymmetry vector: |f_OS - f_OD|
    feat_diff = layers.Lambda(
        lambda t: tf.abs(t[0] - t[1]),
        name="inter_ocular_difference"
    )([feat_left, feat_right])

    # Bilateral fusion representation
    fused = layers.Concatenate(name="bilateral_fusion")([feat_left, feat_right, feat_diff])

    # Left Eye (OS) Diagnosis Head
    l_dense = layers.Dense(256, activation="swish", name="left_dense")(feat_left)
    l_drop = layers.Dropout(0.30, name="left_drop")(l_dense)
    left_pred = layers.Dense(
        config.num_classes, activation="softmax", dtype="float32", name="left_eye_pred"
    )(l_drop)

    # Right Eye (OD) Diagnosis Head
    r_dense = layers.Dense(256, activation="swish", name="right_dense")(feat_right)
    r_drop = layers.Dropout(0.30, name="right_drop")(r_dense)
    right_pred = layers.Dense(
        config.num_classes, activation="softmax", dtype="float32", name="right_eye_pred"
    )(r_drop)

    # Overall Patient Triage Head
    p_dense = layers.Dense(384, activation="swish", name="patient_dense")(fused)
    p_drop = layers.Dropout(0.40, name="patient_drop")(p_dense)
    patient_pred = layers.Dense(
        config.num_classes, activation="softmax", dtype="float32", name="patient_overall_pred"
    )(p_drop)

    model = keras.Model(
        inputs={"left_eye_input": left_input, "right_eye_input": right_input},
        outputs={
            "left_eye_pred": left_pred,
            "right_eye_pred": right_pred,
            "patient_overall_pred": patient_pred
        },
        name="BilateralRetinalClassifier"
    )

    return model, shared_backbone


def compile_binocular_model(
    model: keras.Model,
    config: PipelineConfig,
    learning_rate: float,
    decay_schedule=None
) -> keras.Model:
    """Compiles the model with Multi-Output Categorical Focal Loss."""
    if config.loss_type == "focal":
        loss_fn = CategoricalFocalLoss(gamma=config.focal_gamma, label_smoothing=config.label_smoothing)
    else:
        loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=config.label_smoothing)

    lr = decay_schedule if decay_schedule is not None else learning_rate

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=config.weight_decay),
        loss={
            "left_eye_pred": loss_fn,
            "right_eye_pred": loss_fn,
            "patient_overall_pred": loss_fn
        },
        loss_weights={
            "left_eye_pred": 1.0,
            "right_eye_pred": 1.0,
            "patient_overall_pred": 1.5
        },
        metrics={
            "left_eye_pred": ["accuracy", keras.metrics.AUC(name="auc")],
            "right_eye_pred": ["accuracy", keras.metrics.AUC(name="auc")],
            "patient_overall_pred": ["accuracy", keras.metrics.AUC(name="auc")]
        }
    )
    return model
