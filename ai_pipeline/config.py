"""
Global Configuration for Bilateral Retinal Disease Classifier Pipeline.
"""

from dataclasses import dataclass
from typing import Tuple, List
import os


@dataclass
class PipelineConfig:
    # Dataset & Classes
    data_dir: str = r"C:\Users\Ranit Madhu\.cache\kagglehub\datasets\tanjemahamed\odir5k-classification\versions\1\datasets"
    kaggle_dataset: str = "tanjemahamed/odir5k-classification"
    class_names: Tuple[str, ...] = ("normal", "cataract", "glaucoma", "diabetic_retinopathy")
    num_classes: int = 4

    # Resolution & Batching
    img_size: Tuple[int, int] = (384, 384)
    batch_size: int = 16

    # Model Backbone
    backbone: str = "efficientnet_v2_s"  # "efficientnet_v2_s", "densenet121"
    weights: str = "imagenet"

    # Loss Function (Categorical Focal Loss)
    loss_type: str = "focal"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05

    # Preprocessing & TTA
    use_ben_graham: bool = True
    use_tta: bool = True

    # Splits
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42

    # Training Epochs
    epochs_head: int = 10
    epochs_finetune: int = 20
    unfreeze_layers: int = 150
    head_lr: float = 1e-3
    finetune_lr: float = 3e-5
    weight_decay: float = 1e-4

    # Output Directories
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    final_model_name: str = "retinal_binocular_pipeline.keras"

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


config = PipelineConfig()
