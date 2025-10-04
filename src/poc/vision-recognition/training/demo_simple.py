#!/usr/bin/env python3
"""
Simple Training Demo without Unicode for Windows compatibility
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def show_training_overview():
    """Show overview of the training process"""
    print("=" * 80)
    print("LFM2-VL VISION RECOGNITION - TRAINING OVERVIEW")
    print("=" * 80)
    print()

    print("TRAINING PIPELINE:")
    print("   1. Data Preparation & Augmentation")
    print("   2. Model Architecture Setup (Vision Backbone + Classification Head)")
    print("   3. Training Loop with Early Stopping")
    print("   4. Model Evaluation & Performance Benchmarking")
    print("   5. Fine-Tuning for Specific Tasks")
    print()

    print("KEY FEATURES:")
    print("   * Vision-Language Model (LFM2-VL) architecture")
    print("   * Face and object detection capabilities")
    print("   * Automated data augmentation")
    print("   * Progressive learning rate scheduling")
    print("   * Hardware acceleration (GPU/CPU)")
    print("   * Model optimization for edge deployment")
    print("   * Transfer learning and fine-tuning")
    print()

    print("SUPPORTED TASKS:")
    print("   * Image Classification")
    print("   * Object Detection")
    print("   * Face Recognition")
    print("   * Multi-class Recognition")
    print()


def demo_training():
    """Demonstrate model training"""
    print("=" * 60)
    print("Demo: LFM2-VL Model Training")
    print("=" * 60)
    print()

    try:
        from train_model import main as train_main
        print("Starting model training...")
        model, history = train_main()
        print("SUCCESS: Training completed!")
        return True
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        return False


def demo_evaluation():
    """Demonstrate model evaluation"""
    print("\n" + "=" * 60)
    print("Demo: Model Evaluation")
    print("=" * 60)
    print()

    try:
        from evaluate_model import main as eval_main
        print("Starting model evaluation...")
        metrics, performance = eval_main()
        print("SUCCESS: Evaluation completed!")
        return True
    except Exception as e:
        print(f"ERROR: Evaluation failed: {e}")
        return False


def demo_fine_tuning():
    """Demonstrate model fine-tuning"""
    print("\n" + "=" * 60)
    print("Demo: Model Fine-Tuning")
    print("=" * 60)
    print()

    try:
        from fine_tune import main as finetune_main
        print("Starting model fine-tuning...")
        fine_tuner, history = finetune_main()
        print("SUCCESS: Fine-tuning completed!")
        return True
    except Exception as e:
        print(f"ERROR: Fine-tuning failed: {e}")
        return False


def show_usage_examples():
    """Show usage examples"""
    print("=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    print()

    print("1. Basic Training:")
    print("   python training/train_model.py")
    print()

    print("2. Model Evaluation:")
    print("   python training/evaluate_model.py")
    print()

    print("3. Fine-Tuning:")
    print("   python training/fine_tune.py")
    print()

    print("4. Interactive Demo:")
    print("   python training/demo_simple.py")
    print()


def main():
    """Main function"""
    print("LFM2-VL Vision Recognition Training Demo")
    print("=" * 50)
    print()

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "train":
            demo_training()
        elif command == "eval":
            demo_evaluation()
        elif command == "finetune":
            demo_fine_tuning()
        elif command == "overview":
            show_training_overview()
        elif command == "examples":
            show_usage_examples()
        else:
            print(f"Unknown command: {command}")
            print("Available: train, eval, finetune, overview, examples")
    else:
        # Interactive menu
        while True:
            print("\nTraining Demo Menu:")
            print("1. Show Overview")
            print("2. Show Examples")
            print("3. Run Training Demo")
            print("4. Run Evaluation Demo")
            print("5. Run Fine-Tuning Demo")
            print("0. Exit")
            print()

            try:
                choice = input("Select option (0-5): ").strip()
                if choice == "0":
                    break
                elif choice == "1":
                    show_training_overview()
                elif choice == "2":
                    show_usage_examples()
                elif choice == "3":
                    demo_training()
                elif choice == "4":
                    demo_evaluation()
                elif choice == "5":
                    demo_fine_tuning()
                else:
                    print("Invalid option.")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

            input("Press Enter to continue...")

    print("\nDemo completed!")


if __name__ == "__main__":
    main()