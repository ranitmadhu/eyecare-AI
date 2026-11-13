"""
Dataset Ingestion, Bilateral Pairing, and Augmentation Pipeline.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any

from config import PipelineConfig
from preprocessing import clean_fundus_np


def get_dataset_dataframe(config: PipelineConfig) -> pd.DataFrame:
    """Ingests image paths, labels them, and constructs bilateral (OS + OD) patient pairs."""
    data_dir = config.data_dir

    # Fallback to kagglehub download if data_dir not found
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        try:
            import kagglehub
            print(f"[Dataset] Downloading '{config.kaggle_dataset}' via kagglehub...")
            dl_path = kagglehub.dataset_download(config.kaggle_dataset)
            nested = os.path.join(dl_path, "datasets")
            nested_alt = os.path.join(dl_path, "dataset")
            if os.path.exists(nested):
                data_dir = nested
            elif os.path.exists(nested_alt):
                data_dir = nested_alt
            else:
                data_dir = dl_path
            print("[Dataset] Dataset downloaded to:", data_dir)
        except Exception as e:
            print("[Dataset] Warning: kagglehub download skipped/failed:", e)

    patterns = ["**/*.[jJ][pP][gG]", "**/*.[jJ][pP][eE][gG]", "**/*.[pP][nN][gG]"]
    all_files = []
    for pat in patterns:
        all_files.extend(glob.glob(os.path.join(data_dir, pat), recursive=True))

    if len(all_files) == 0:
        raise FileNotFoundError(f"No retinal images found in {data_dir}!")

    records = []
    for fpath in all_files:
        pname = os.path.basename(os.path.dirname(fpath)).lower()
        fname = os.path.basename(fpath).lower()
        label = None
        for cls in config.class_names:
            if cls == "diabetic_retinopathy" and ("diabetes" in pname or "diabetes" in fname):
                label = cls
                break
            elif cls in pname or cls in fname:
                label = cls
                break
        if label:
            is_left = ("_left" in fname or "_os" in fname or "left" in fname)
            is_right = ("_right" in fname or "_od" in fname or "right" in fname)
            m = re.match(r"^(\d+)[_\-\.]", fname)
            pid = m.group(1) if m else None

            records.append({
                "filepath": fpath,
                "filename": fname,
                "label": label,
                "pid": pid,
                "is_left": is_left,
                "is_right": is_right
            })

    df_raw = pd.DataFrame(records)
    print(f"[Dataset] Loaded {len(df_raw)} images matching target classes.")

    # Pair into Left and Right Eye (Bilateral) patient sets
    paired_rows = []
    patients_with_id = df_raw[df_raw["pid"].notnull()]["pid"].unique()
    paired_pids = set()

    for pid in patients_with_id:
        sub = df_raw[df_raw["pid"] == pid]
        left_row = sub[sub["is_left"]]
        right_row = sub[sub["is_right"]]
        if len(left_row) > 0 and len(right_row) > 0:
            l = left_row.iloc[0]
            r = right_row.iloc[0]
            paired_rows.append({
                "left_path": l["filepath"],
                "right_path": r["filepath"],
                "left_label": l["label"],
                "right_label": r["label"],
                "patient_label": l["label"] if l["label"] == r["label"] else (l["label"] if l["label"] != "normal" else r["label"])
            })
            paired_pids.add(pid)

    # For remaining images, pair within the same class
    df_rem = df_raw[~df_raw["pid"].isin(paired_pids)].copy()
    for cls in config.class_names:
        cls_items = df_rem[df_rem["label"] == cls]["filepath"].values
        np.random.shuffle(cls_items)
        n_pairs = len(cls_items) // 2
        for i in range(n_pairs):
            paired_rows.append({
                "left_path": cls_items[2 * i],
                "right_path": cls_items[2 * i + 1],
                "left_label": cls,
                "right_label": cls,
                "patient_label": cls
            })

    pair_df = pd.DataFrame(paired_rows)
    label_map = {name: i for i, name in enumerate(config.class_names)}
    pair_df["left_idx"] = pair_df["left_label"].map(label_map)
    pair_df["right_idx"] = pair_df["right_label"].map(label_map)
    pair_df["patient_idx"] = pair_df["patient_label"].map(label_map)

    print(f"[Dataset] Constructed {len(pair_df)} Bilateral (Left OS + Right OD) patient pairs.")
    return pair_df


def build_ocular_augmentation() -> keras.Sequential:
    """Ocular rotation, flips, zoom, brightness and contrast jitter."""
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.5, fill_mode="constant"),
        layers.RandomZoom(height_factor=(-0.08, 0.10), fill_mode="constant"),
        layers.RandomContrast(0.12),
        layers.RandomBrightness(0.08),
    ], name="ocular_augmentations")


def make_bilateral_tf_dataset(
    df_pairs: pd.DataFrame,
    config: PipelineConfig,
    training: bool = False
) -> tf.data.Dataset:
    """Builds an optimized tf.data.Dataset yielding dual-eye tensors and targets."""
    l_paths = df_pairs["left_path"].values
    r_paths = df_pairs["right_path"].values
    l_labels = df_pairs["left_idx"].values
    r_labels = df_pairs["right_idx"].values
    p_labels = df_pairs["patient_idx"].values

    target_size = config.img_size[0]
    use_bg = config.use_ben_graham
    num_classes = config.num_classes
    augment_op = build_ocular_augmentation()

    def _py_pair_loader(lp, rp):
        import cv2
        def _clean(p_str):
            raw = cv2.imread(p_str)
            if raw is None:
                return np.zeros((target_size, target_size, 3), dtype=np.uint8)
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            return clean_fundus_np(raw, target_size=target_size, use_ben_graham=use_bg)

        l_c = _clean(lp.numpy().decode("utf-8"))
        r_c = _clean(rp.numpy().decode("utf-8"))
        return l_c, r_c

    def _parse_fn(lp, rp, l_lbl, r_lbl, p_lbl):
        l_img, r_img = tf.py_function(func=_py_pair_loader, inp=[lp, rp], Tout=[tf.uint8, tf.uint8])
        l_img.set_shape([target_size, target_size, 3])
        r_img.set_shape([target_size, target_size, 3])
        l_img = tf.cast(l_img, tf.float32)
        r_img = tf.cast(r_img, tf.float32)

        inputs = {"left_eye_input": l_img, "right_eye_input": r_img}
        targets = {
            "left_eye_pred": tf.one_hot(l_lbl, num_classes),
            "right_eye_pred": tf.one_hot(r_lbl, num_classes),
            "patient_overall_pred": tf.one_hot(p_lbl, num_classes)
        }
        return inputs, targets

    ds = tf.data.Dataset.from_tensor_slices((l_paths, r_paths, l_labels, r_labels, p_labels))
    if training:
        ds = ds.shuffle(buffer_size=len(df_pairs), seed=config.seed, reshuffle_each_iteration=True)
    ds = ds.map(_parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(
            lambda inp, tgt: ({
                "left_eye_input": augment_op(inp["left_eye_input"], training=True),
                "right_eye_input": augment_op(inp["right_eye_input"], training=True)
            }, tgt),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    return ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)


def split_bilateral_data(
    pair_df: pd.DataFrame,
    config: PipelineConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train, val, test split based on patient condition."""
    total_holdout = config.val_split + config.test_split
    train_df, temp_df = train_test_split(
        pair_df, test_size=total_holdout, random_state=config.seed, stratify=pair_df["patient_idx"]
    )
    val_ratio = config.test_split / total_holdout
    val_df, test_df = train_test_split(
        temp_df, test_size=val_ratio, random_state=config.seed, stratify=temp_df["patient_idx"]
    )
    print(f"[Dataset] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)} pairs")
    return train_df, val_df, test_df
