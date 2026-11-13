"""
Bilateral Evaluation Pipeline with Test-Time Augmentation (TTA).
Computes classification report, normalized confusion matrices, and ROC curves.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve

from config import PipelineConfig
from losses import CategoricalFocalLoss


def predict_binocular_tta(model: keras.Model, l_imgs: tf.Tensor, r_imgs: tf.Tensor) -> dict:
    """Averages predictions across original, h-flip, v-flip, and rot90 for both eyes."""
    p0 = model({"left_eye_input": l_imgs, "right_eye_input": r_imgs}, training=False)
    p1 = model({
        "left_eye_input": tf.image.flip_left_right(l_imgs),
        "right_eye_input": tf.image.flip_left_right(r_imgs)
    }, training=False)
    p2 = model({
        "left_eye_input": tf.image.flip_up_down(l_imgs),
        "right_eye_input": tf.image.flip_up_down(r_imgs)
    }, training=False)
    p3 = model({
        "left_eye_input": tf.image.rot90(l_imgs, k=1),
        "right_eye_input": tf.image.rot90(r_imgs, k=1)
    }, training=False)

    avg = {}
    for k in p0.keys():
        avg[k] = (p0[k] + p1[k] + p2[k] + p3[k]) / 4.0
    return avg


def evaluate_binocular_pipeline(
    model: keras.Model,
    test_ds: tf.data.Dataset,
    config: PipelineConfig,
    save_plots: bool = True
) -> dict:
    """Evaluates Siamese binocular model on test set and generates diagnostic plots."""
    print("\n" + "=" * 65)
    print(f"EVALUATING TEST DATASET (TTA={'ENABLED' if config.use_tta else 'DISABLED'})")
    print("=" * 65)

    l_true_list, r_true_list, p_true_list = [], [], []
    l_pred_list, r_pred_list, p_pred_list = [], [], []

    for inputs, targets in test_ds:
        if config.use_tta:
            preds = predict_binocular_tta(model, inputs["left_eye_input"], inputs["right_eye_input"])
        else:
            preds = model(inputs, training=False)

        l_pred_list.append(preds["left_eye_pred"].numpy())
        r_pred_list.append(preds["right_eye_pred"].numpy())
        p_pred_list.append(preds["patient_overall_pred"].numpy())

        l_true_list.append(targets["left_eye_pred"].numpy())
        r_true_list.append(targets["right_eye_pred"].numpy())
        p_true_list.append(targets["patient_overall_pred"].numpy())

    p_true = np.argmax(np.concatenate(p_true_list, axis=0), axis=1)
    p_pred = np.argmax(np.concatenate(p_pred_list, axis=0), axis=1)

    report_str = classification_report(p_true, p_pred, target_names=config.class_names, digits=4)
    print(report_str)

    if save_plots:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        cm = confusion_matrix(p_true, p_pred)
        cm_norm = confusion_matrix(p_true, p_pred, normalize="true")

        ConfusionMatrixDisplay(cm, display_labels=config.class_names).plot(ax=axes[0], cmap="Blues", xticks_rotation=30)
        axes[0].set_title("Patient Overall Confusion Matrix (Counts)")

        ConfusionMatrixDisplay(cm_norm, display_labels=config.class_names).plot(ax=axes[1], cmap="Purples", xticks_rotation=30, values_format=".2f")
        axes[1].set_title("Patient Overall Confusion Matrix (Normalized Recall)")
        plt.tight_layout()

        out_cm = os.path.join(config.output_dir, "confusion_matrix.png")
        plt.savefig(out_cm, dpi=200)
        plt.close()
        print(f"[Evaluation] Confusion matrix saved to: {out_cm}")

    return {
        "classification_report": report_str,
        "y_true": p_true,
        "y_pred": p_pred
    }
