#!/usr/bin/env python3
"""
Setup script for LFM2-VL Vision Recognition POC
Handles installation, model download, and environment setup
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
import urllib.request
import shutil


def run_command(cmd, check=True):
    """Run shell command with error handling"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if check and result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {result.stderr}")
        sys.exit(1)

    return result


def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)

    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")


def create_virtual_environment():
    """Create Python virtual environment"""
    venv_path = Path("venv")

    if venv_path.exists():
        print("📁 Virtual environment already exists")
        return

    print("🔧 Creating virtual environment...")
    run_command("python -m venv venv")

    # Activate script paths for different platforms
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate.bat"
        pip_path = "venv\\Scripts\\pip"
    else:  # Unix-like
        activate_script = "venv/bin/activate"
        pip_path = "venv/bin/pip"

    print(f"✅ Virtual environment created")
    print(f"   To activate: {activate_script}")

    return pip_path


def install_dependencies(pip_path=None):
    """Install Python dependencies"""
    pip_cmd = pip_path or "pip"
    requirements_file = Path("config/requirements.txt")

    if not requirements_file.exists():
        print("❌ Requirements file not found")
        sys.exit(1)

    print("📦 Installing dependencies...")
    run_command(f"{pip_cmd} install --upgrade pip")
    run_command(f"{pip_cmd} install -r {requirements_file}")

    print("✅ Dependencies installed")


def create_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "data/test_images",
        "output",
        "logs",
        "cache",
        "tests/fixtures"
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print("✅ Directory structure created")


def download_test_assets():
    """Download test images and create mock models"""
    print("🖼️ Setting up test assets...")

    # Create mock model files (placeholders)
    models_dir = Path("models")

    mock_models = {
        "lfm2_vl_base.pth": "PyTorch base model (mock)",
        "lfm2_vl_mobile.tflite": "TensorFlow Lite mobile model (mock)",
        "lfm2_vl_mobile.mlmodel": "CoreML iOS model (mock)",
        "labels.txt": "person\\nface\\ncar\\nbicycle\\ndog\\ncat\\nbird\\nchair\\ntable\\nbottle",
        "config.json": json.dumps({
            "model_version": "1.0.0",
            "input_size": [416, 416],
            "num_classes": 10,
            "confidence_threshold": 0.5
        }, indent=2)
    }

    for filename, content in mock_models.items():
        filepath = models_dir / filename
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"   Created: {filepath}")

    # Create test images using OpenCV (simple colored rectangles)
    create_test_images()

    print("✅ Test assets created")


def create_test_images():
    """Create synthetic test images for demo"""
    try:
        import cv2
        import numpy as np

        test_images_dir = Path("data/test_images")

        # Test image 1: Simple objects
        img1 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img1, (100, 100), (200, 200), (0, 255, 0), -1)  # Green rectangle
        cv2.circle(img1, (400, 200), 50, (0, 0, 255), -1)  # Red circle
        cv2.imwrite(str(test_images_dir / "test_objects.jpg"), img1)

        # Test image 2: Face-like pattern
        img2 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(img2, (320, 240), 80, (255, 255, 200), -1)  # Face outline
        cv2.circle(img2, (300, 220), 10, (0, 0, 0), -1)  # Left eye
        cv2.circle(img2, (340, 220), 10, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(img2, (320, 260), (20, 10), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        cv2.imwrite(str(test_images_dir / "test_face.jpg"), img2)

        # Test image 3: Mixed scene
        img3 = np.random.randint(0, 100, (480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img3, (50, 50), (150, 150), (255, 0, 0), 3)
        cv2.rectangle(img3, (200, 200), (350, 350), (0, 255, 0), 3)
        cv2.circle(img3, (500, 100), 40, (255, 255, 0), 3)
        cv2.imwrite(str(test_images_dir / "test_mixed.jpg"), img3)

        print("   Created synthetic test images")

    except ImportError:
        print("   ⚠️ OpenCV not available, skipping test image creation")


def run_basic_test():
    """Run basic functionality test"""
    print("🧪 Running basic functionality test...")

    try:
        # Test imports
        sys.path.insert(0, str(Path("core")))

        from vision_pipeline import VisionPipeline, ProcessingMode
        from model_optimizer import ModelOptimizer

        print("   ✅ Core modules import successfully")

        # Test pipeline initialization
        pipeline = VisionPipeline(
            processing_mode=ProcessingMode.REALTIME,
            config={'confidence_threshold': 0.5}
        )
        print("   ✅ Vision pipeline initialized")

        # Test optimizer initialization
        optimizer = ModelOptimizer(target_device="mobile")
        print("   ✅ Model optimizer initialized")

        # Test with synthetic image
        import numpy as np
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        results = pipeline.process_image(test_image)
        print(f"   ✅ Image processing test: {len(results['faces'])} faces, {len(results['objects'])} objects")

        print("✅ Basic functionality test passed")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    return True


def create_demo_script():
    """Create quick start demo script"""
    demo_script = '''#!/usr/bin/env python3
"""Quick start demo for LFM2-VL Vision Recognition"""

import sys
import os
sys.path.insert(0, 'core')

from vision_pipeline import VisionPipeline, ProcessingMode
import numpy as np

def main():
    print("🚀 LFM2-VL Quick Demo")
    print("=" * 30)

    # Initialize pipeline
    pipeline = VisionPipeline(processing_mode=ProcessingMode.REALTIME)

    # Create synthetic test image
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

    print("\\n✅ Demo completed successfully!")

if __name__ == '__main__':
    main()
'''

    with open("quick_demo.py", 'w') as f:
        f.write(demo_script)

    print("✅ Quick demo script created: quick_demo.py")


def print_usage_instructions():
    """Print usage instructions after setup"""
    print("\n" + "="*60)
    print("🎉 LEAP-PSW Vision Recognition POC Setup Complete!")
    print("="*60)

    print("\n🚀 Quick Start:")
    if os.name == 'nt':  # Windows
        print("   1. Activate environment: venv\\Scripts\\activate.bat")
    else:  # Unix-like
        print("   1. Activate environment: source venv/bin/activate")

    print("   2. Run quick demo: python quick_demo.py")
    print("   3. Run full demo: python examples/demo.py")

    print("\n📋 Available Demo Commands:")
    print("   # Image processing")
    print("   python examples/demo.py --demo-type image --image-path data/test_images/test_face.jpg")
    print("   ")
    print("   # Batch processing")
    print("   python examples/demo.py --demo-type batch")
    print("   ")
    print("   # Model optimization")
    print("   python examples/demo.py --demo-type optimize --target-device mobile")
    print("   ")
    print("   # Performance benchmark")
    print("   python examples/demo.py --demo-type benchmark")

    print("\n📁 Project Structure:")
    print("   models/           - Mock model files")
    print("   data/test_images/ - Synthetic test images")
    print("   core/            - Core processing modules")
    print("   examples/        - Demo scripts")
    print("   mobile/          - iOS/Android integration")

    print("\n📚 Next Steps:")
    print("   • Review README.md for detailed documentation")
    print("   • Customize config files for your use case")
    print("   • Integrate real LFM2-VL models when available")
    print("   • Deploy to mobile devices using templates")

    print("\n🔧 Development:")
    print("   • Run tests: pytest tests/")
    print("   • Code formatting: black core/ examples/")
    print("   • Type checking: mypy core/")


def main():
    parser = argparse.ArgumentParser(description='Setup LFM2-VL Vision Recognition POC')
    parser.add_argument('--skip-venv', action='store_true',
                       help='Skip virtual environment creation')
    parser.add_argument('--skip-deps', action='store_true',
                       help='Skip dependency installation')
    parser.add_argument('--skip-test', action='store_true',
                       help='Skip functionality test')
    parser.add_argument('--quick', action='store_true',
                       help='Quick setup (minimal dependencies)')

    args = parser.parse_args()

    print("🔧 LFM2-VL Vision Recognition POC Setup")
    print("=" * 50)

    # Check Python version
    check_python_version()

    # Create virtual environment
    pip_path = None
    if not args.skip_venv:
        pip_path = create_virtual_environment()

    # Create directory structure
    create_directories()

    # Install dependencies
    if not args.skip_deps:
        install_dependencies(pip_path)

    # Setup test assets
    download_test_assets()

    # Create demo script
    create_demo_script()

    # Run basic test
    if not args.skip_test:
        if not run_basic_test():
            print("⚠️ Setup completed with test failures")
            return

    # Print usage instructions
    print_usage_instructions()


if __name__ == '__main__':
    main()