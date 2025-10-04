#!/usr/bin/env python3
"""
Demo Script for LFM2-VL Training Pipeline
Shows how to train, evaluate, and fine-tune the vision recognition model
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


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
        print("SUCCESS: Training completed successfully!")
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
        print("✅ Evaluation completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False


def demo_fine_tuning():
    """Demonstrate model fine-tuning"""
    print("\n" + "=" * 60)
    print("Demo: Model Fine-Tuning")
    print("=" * 60)
    print()

    try:
        from fine_tune import main as finetune_main
        print("🎯 Starting model fine-tuning...")
        fine_tuner, history = finetune_main()
        print("✅ Fine-tuning completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        return False


def show_training_overview():
    """Show overview of the training process"""
    print("=" * 80)
    print("LFM2-VL VISION RECOGNITION - TRAINING OVERVIEW")
    print("=" * 80)
    print()

    print("📋 TRAINING PIPELINE:")
    print("   1. Data Preparation & Augmentation")
    print("   2. Model Architecture Setup (Vision Backbone + Classification Head)")
    print("   3. Training Loop with Early Stopping")
    print("   4. Model Evaluation & Performance Benchmarking")
    print("   5. Fine-Tuning for Specific Tasks")
    print()

    print("🎯 KEY FEATURES:")
    print("   • Vision-Language Model (LFM2-VL) architecture")
    print("   • Face and object detection capabilities")
    print("   • Automated data augmentation")
    print("   • Progressive learning rate scheduling")
    print("   • Hardware acceleration (GPU/CPU)")
    print("   • Model optimization for edge deployment")
    print("   • Transfer learning and fine-tuning")
    print()

    print("📊 SUPPORTED TASKS:")
    print("   • Image Classification")
    print("   • Object Detection")
    print("   • Face Recognition")
    print("   • Multi-class Recognition")
    print()

    print("⚙️ CONFIGURATION OPTIONS:")
    print("   • Model size and complexity")
    print("   • Training parameters (LR, epochs, batch size)")
    print("   • Data augmentation strategies")
    print("   • Optimization techniques")
    print("   • Hardware utilization")
    print()


def show_usage_examples():
    """Show usage examples for training"""
    print("=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    print()

    examples = [
        ("Basic Training", """
# 1. Prepare your dataset
data/
├── train/
│   ├── face/
│   ├── person/
│   └── car/
└── val/
    ├── face/
    ├── person/
    └── car/

# 2. Start training
python training/train_model.py
"""),

        ("Custom Configuration", """
# Modify TrainingConfig in train_model.py
config = TrainingConfig(
    num_classes=5,           # Your number of classes
    batch_size=32,           # Adjust based on GPU memory
    learning_rate=0.001,     # Learning rate
    num_epochs=50,           # Training epochs
    device="cuda"            # Use GPU if available
)
"""),

        ("Model Evaluation", """
# Evaluate trained model
python training/evaluate_model.py

# This will generate:
# - Accuracy, Precision, Recall, F1-Score
# - Per-class performance metrics
# - Confusion matrix
# - Inference time benchmarks
"""),

        ("Fine-Tuning for New Task", """
# Fine-tune for specific domain
python training/fine_tune.py

# Configuration for fine-tuning:
config = FineTuningConfig(
    pretrained_model_path="checkpoints/best_model.pth",
    new_num_classes=3,       # New task classes
    learning_rate=0.0001,    # Lower LR for fine-tuning
    freeze_backbone=True     # Freeze pre-trained layers
)
"""),

        ("Real Dataset Integration", """
# Replace mock dataset in VisionDataset._load_samples()
def _load_samples(self):
    samples = []
    for class_dir in os.listdir(self.data_dir):
        class_path = self.data_dir / class_dir
        class_id = self.class_to_id[class_dir]

        for img_file in class_path.glob('*.jpg'):
            samples.append({
                'image_path': str(img_file),
                'label': class_id,
                'category': class_dir
            })
    return samples
""")
    ]

    for title, code in examples:
        print(f"📖 {title}:")
        print(code)
        print()


def show_best_practices():
    """Show training best practices"""
    print("=" * 80)
    print("TRAINING BEST PRACTICES")
    print("=" * 80)
    print()

    practices = [
        ("Data Preparation", [
            "• Balance your dataset across classes",
            "• Use high-quality, diverse images",
            "• Split: 70% train, 20% validation, 10% test",
            "• Apply appropriate data augmentation"
        ]),

        ("Model Configuration", [
            "• Start with pre-trained weights when available",
            "• Use appropriate input resolution (224x224 standard)",
            "• Adjust model complexity based on dataset size",
            "• Monitor GPU memory usage"
        ]),

        ("Training Strategy", [
            "• Use learning rate scheduling",
            "• Implement early stopping",
            "• Monitor both training and validation metrics",
            "• Save checkpoints regularly"
        ]),

        ("Fine-Tuning Tips", [
            "• Use lower learning rates (10x smaller)",
            "• Freeze backbone initially, unfreeze gradually",
            "• Use smaller batch sizes",
            "• Apply stronger regularization"
        ]),

        ("Performance Optimization", [
            "• Use mixed precision training (FP16)",
            "• Enable data loading parallelization",
            "• Optimize data augmentation pipeline",
            "• Profile GPU utilization"
        ])
    ]

    for category, tips in practices:
        print(f"🎯 {category}:")
        for tip in tips:
            print(f"   {tip}")
        print()


def interactive_demo():
    """Interactive demo menu"""
    while True:
        print("\n" + "=" * 60)
        print("LFM2-VL TRAINING DEMO MENU")
        print("=" * 60)
        print()
        print("1. Show Training Overview")
        print("2. Show Usage Examples")
        print("3. Show Best Practices")
        print("4. Run Training Demo")
        print("5. Run Evaluation Demo")
        print("6. Run Fine-Tuning Demo")
        print("7. Run All Demos")
        print("0. Exit")
        print()

        try:
            choice = input("Select option (0-7): ").strip()

            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                show_training_overview()
            elif choice == "2":
                show_usage_examples()
            elif choice == "3":
                show_best_practices()
            elif choice == "4":
                demo_training()
            elif choice == "5":
                demo_evaluation()
            elif choice == "6":
                demo_fine_tuning()
            elif choice == "7":
                print("🚀 Running all demos...")
                demo_training()
                demo_evaluation()
                demo_fine_tuning()
            else:
                print("❌ Invalid option. Please try again.")

        except KeyboardInterrupt:
            print("\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

        input("\nPress Enter to continue...")


def main():
    """Main demo function"""
    print("LFM2-VL Vision Recognition Training Demo")
    print()
    print("This demo shows how to train the LFM2-VL model to recognize")
    print("faces and objects in images using PyTorch.")
    print()

    # Check if in interactive mode
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
        elif command == "practices":
            show_best_practices()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: train, eval, finetune, overview, examples, practices")
    else:
        # Interactive mode
        interactive_demo()


if __name__ == "__main__":
    main()