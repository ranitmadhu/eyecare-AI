"""
Two-Stage Training Pipeline for Siamese Binocular Retinal Classifier.
Stage 1: Head warmup (frozen backbone)
Stage 2: Deep fine-tuning (top 150+ layers unfrozen under Cosine Annealing decay)
"""

import os
import tensorflow as tf
from tensorflow import keras
from typing import Tuple

from config import PipelineConfig
from models import build_binocular_model, compile_binocular_model


def train_binocular_pipeline(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    steps_per_epoch: int,
    config: PipelineConfig
) -> Tuple[keras.Model, dict]:
    """Runs end-to-end two-stage training pipeline."""
    print("=" * 65)
    print("STAGE 1: BINOCULAR CLASSIFICATION HEADS WARMUP")
    print("=" * 65)

    model, backbone = build_binocular_model(config)
    model = compile_binocular_model(model, config, learning_rate=config.head_lr)

    head_ckpt = os.path.join(config.checkpoint_dir, "best_binocular_head.keras")
    callbacks_head = [
        keras.callbacks.ModelCheckpoint(
            head_ckpt, monitor="val_patient_overall_pred_auc", mode="max", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_patient_overall_pred_auc", mode="max", patience=5, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        )
    ]

    hist_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs_head,
        callbacks=callbacks_head
    )

    print("\n" + "=" * 65)
    print("STAGE 2: DEEP SIAMESE FINE-TUNING (COSINE LR DECAY)")
    print("=" * 65)

    backbone.trainable = True
    total_layers = len(backbone.layers)
    cutoff = max(0, total_layers - config.unfreeze_layers)
    for layer in backbone.layers[:cutoff]:
        layer.trainable = False

    print(f"[Training] Unfrozen top {total_layers - cutoff} of {total_layers} backbone layers.")

    total_steps = steps_per_epoch * config.epochs_finetune
    cosine_lr = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config.finetune_lr,
        decay_steps=total_steps,
        alpha=0.02
    )

    model = compile_binocular_model(model, config, learning_rate=config.finetune_lr, decay_schedule=cosine_lr)

    ft_ckpt = os.path.join(config.checkpoint_dir, "best_binocular_finetuned.keras")
    callbacks_ft = [
        keras.callbacks.ModelCheckpoint(
            ft_ckpt, monitor="val_patient_overall_pred_auc", mode="max", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_patient_overall_pred_auc", mode="max", patience=6, restore_best_weights=True, verbose=1
        )
    ]

    hist_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs_finetune,
        callbacks=callbacks_ft
    )

    # Save final pipeline model
    final_path = os.path.join(config.checkpoint_dir, config.final_model_name)
    model.save(final_path)
    print(f"\n[Training] Final model saved successfully at: {final_path}")

    return model, {"stage1": hist_head.history, "stage2": hist_ft.history}
