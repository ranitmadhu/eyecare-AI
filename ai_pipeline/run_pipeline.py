"""
CLI Runner for the Bilateral Retinal Classifier Pipeline.

Usage Examples:
--------------
1. Run full training pipeline:
   python run_pipeline.py --mode train --data-dir "path/to/datasets"

2. Run evaluation pipeline:
   python run_pipeline.py --mode eval --model checkpoints/best_binocular_finetuned.keras

3. Run bilateral screening on a patient:
   python run_pipeline.py --mode predict --left sample_left.jpg --right sample_right.jpg --output dashboard.png
"""

import argparse
import sys
import os
import json

from config import PipelineConfig
from pipeline import BilateralRetinalPipeline


def main():
    parser = argparse.ArgumentParser(description="Bilateral Retinal Disease Screening Pipeline")
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "predict"],
        default="predict",
        help="Pipeline execution mode: 'train', 'eval', or 'predict'"
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Path to ODIR-5K dataset directory")
    parser.add_argument("--model", type=str, default=None, help="Path to trained .keras model checkpoint")
    parser.add_argument("--left", type=str, default=None, help="Path to Left Eye (OS) fundus image")
    parser.add_argument("--right", type=str, default=None, help="Path to Right Eye (OD) fundus image")
    parser.add_argument("--output", type=str, default="bilateral_report.png", help="Path to save output 8-panel dashboard")
    parser.add_argument("--no-tta", action="store_true", help="Disable Test-Time Augmentation")

    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.no_tta:
        cfg.use_tta = False

    pipeline = BilateralRetinalPipeline(config=cfg, model_path=args.model)

    if args.mode == "train":
        print("[Runner] Starting Full Siamese Training Pipeline...")
        model, results = pipeline.run_training_pipeline()
        print("\n[Runner] Training Pipeline Completed Successfully!")

    elif args.mode == "eval":
        if not args.model or not os.path.exists(args.model):
            print(f"[Runner] Error: Must provide valid --model path for evaluation.")
            sys.exit(1)

        from dataset import get_dataset_dataframe, split_bilateral_data, make_bilateral_tf_dataset
        from evaluate import evaluate_binocular_pipeline

        pair_df = get_dataset_dataframe(cfg)
        _, _, test_df = split_bilateral_data(pair_df, cfg)
        test_ds = make_bilateral_tf_dataset(test_df, cfg, training=False)

        metrics = evaluate_binocular_pipeline(pipeline.model, test_ds, cfg, save_plots=True)
        print("\n[Runner] Evaluation Finished.")

    elif args.mode == "predict":
        if not args.left or not args.right:
            print("[Runner] Please provide both --left and --right image paths for bilateral screening.")
            print("Example: python run_pipeline.py --mode predict --left left_eye.jpg --right right_eye.jpg")
            sys.exit(1)

        if not os.path.exists(args.left):
            print(f"[Runner] Error: Left image not found at {args.left}")
            sys.exit(1)
        if not os.path.exists(args.right):
            print(f"[Runner] Error: Right image not found at {args.right}")
            sys.exit(1)

        print(f"[Runner] Running Bilateral Screening Pipeline...")
        print(f"• Left Eye (OS):  {args.left}")
        print(f"• Right Eye (OD): {args.right}")

        result = pipeline.predict_patient(args.left, args.right)

        print("\n" + "=" * 65)
        print("PATIENT CLINICAL ACTION PLAN:")
        print(f"• Overall Diagnosis: {result.diagnosis}")
        print(f"• Triage Urgency:    {result.urgency}")
        print(f"• Clinical Findings: {result.findings}")
        print(f"• Next Steps:        {result.recommendation}")
        print("=" * 65)

        print(f"\n[Left Eye OS]  -> {result.left_label.title()} ({result.left_confidence:.1f}%)")
        print(f"[Right Eye OD] -> {result.right_label.title()} ({result.right_confidence:.1f}%)")

        result.save_dashboard(args.output)
        print(f"\n[Runner] Results successfully saved to: {args.output}")


if __name__ == "__main__":
    main()
