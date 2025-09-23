#!/usr/bin/env python3
"""
Quick setup script for LFM2-VL Vision Recognition POC
Simple version without Unicode for Windows compatibility
"""

import os
import sys
import json
from pathlib import Path


def create_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "data/test_images",
        "output",
        "logs",
        "tests/fixtures"
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print("Directory structure created")


def create_mock_models():
    """Create mock model files"""
    models_dir = Path("models")

    mock_files = {
        "lfm2_vl_base.pth": "# PyTorch base model (mock)\n# This is a placeholder file",
        "lfm2_vl_mobile.tflite": "# TensorFlow Lite mobile model (mock)",
        "lfm2_vl_mobile.mlmodel": "# CoreML iOS model (mock)",
        "labels.txt": "person\nface\ncar\nbicycle\ndog\ncat\nbird\nchair\ntable\nbottle",
        "config.json": json.dumps({
            "model_version": "1.0.0",
            "input_size": [416, 416],
            "num_classes": 10,
            "confidence_threshold": 0.5
        }, indent=2)
    }

    for filename, content in mock_files.items():
        filepath = models_dir / filename
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Created: {filepath}")


def create_test_images():
    """Create simple test images using basic Python"""
    try:
        import numpy as np

        test_images_dir = Path("data/test_images")

        # Create simple text file descriptions instead of actual images
        test_descriptions = {
            "test_objects.txt": "Synthetic image with green rectangle and red circle",
            "test_face.txt": "Synthetic face-like pattern with eyes and mouth",
            "test_mixed.txt": "Mixed scene with various shapes and objects"
        }

        for filename, description in test_descriptions.items():
            filepath = test_images_dir / filename
            with open(filepath, 'w') as f:
                f.write(description)

        print("Test image descriptions created")

    except ImportError:
        print("NumPy not available, skipping test image creation")


def create_quick_demo():
    """Create simple demo script"""
    demo_script = '''#!/usr/bin/env python3
"""Quick start demo for LFM2-VL Vision Recognition"""

import sys
import os
sys.path.insert(0, 'core')

def main():
    print("LFM2-VL Quick Demo")
    print("=" * 30)

    try:
        from vision_pipeline import VisionPipeline, ProcessingMode
        print("SUCCESS: Core modules imported")

        # Initialize pipeline
        pipeline = VisionPipeline(processing_mode=ProcessingMode.REALTIME)
        print("SUCCESS: Vision pipeline initialized")

        # Create test data
        import numpy as np
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Process image
        results = pipeline.process_image(test_image)

        # Show results
        print(f"Faces detected: {len(results['faces'])}")
        print(f"Objects detected: {len(results['objects'])}")
        print(f"Processing time: {results['metadata']['processing_time']:.3f}s")

        # Performance stats
        stats = pipeline.get_performance_stats()
        print(f"Performance: {stats['average_inference_time']:.3f}s avg")

        print("\\nDemo completed successfully!")

    except Exception as e:
        print(f"ERROR: {e}")
        return False

    return True

if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)
'''

    with open("quick_demo.py", 'w') as f:
        f.write(demo_script)

    print("Quick demo script created")


def run_basic_test():
    """Run basic test"""
    print("Running basic test...")

    try:
        sys.path.insert(0, str(Path("core")))
        from vision_pipeline import VisionPipeline, ProcessingMode

        pipeline = VisionPipeline(processing_mode=ProcessingMode.REALTIME)
        print("Basic test passed")
        return True

    except Exception as e:
        print(f"Test failed: {e}")
        return False


def main():
    print("LFM2-VL Vision Recognition POC Setup")
    print("=" * 50)

    # Create structure
    create_directories()
    create_mock_models()
    create_test_images()
    create_quick_demo()

    # Test basic functionality
    if run_basic_test():
        print("\nSetup completed successfully!")
        print("\nTo run demo:")
        print("  python quick_demo.py")
        print("  python examples/demo.py")
    else:
        print("\nSetup completed with warnings")


if __name__ == '__main__':
    main()