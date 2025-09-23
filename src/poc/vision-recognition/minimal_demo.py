#!/usr/bin/env python3
"""
Minimal demo for LFM2-VL Vision Recognition POC
Works without external dependencies for basic testing
"""

import sys
import os
import time
import json
from pathlib import Path

# Add core module to path
sys.path.insert(0, 'core')


class MockNumPy:
    """Mock numpy for basic demo"""
    @staticmethod
    def random_randint(low, high, shape, dtype=None):
        import random
        total_elements = 1
        for dim in shape:
            total_elements *= dim

        # Create flat list of random integers
        data = [random.randint(low, high-1) for _ in range(total_elements)]

        # For demo purposes, just return the data as nested lists
        if len(shape) == 3:  # Image-like (H, W, C)
            h, w, c = shape
            result = []
            idx = 0
            for y in range(h):
                row = []
                for x in range(w):
                    pixel = []
                    for channel in range(c):
                        pixel.append(data[idx])
                        idx += 1
                    row.append(pixel)
                result.append(row)
            return MockArray(result, shape)
        else:
            return MockArray(data, shape)

    @staticmethod
    def random_randn(*shape):
        import random
        total_elements = 1
        for dim in shape:
            total_elements *= dim

        data = [random.gauss(0, 1) for _ in range(total_elements)]
        return MockArray(data, shape)


class MockArray:
    """Mock numpy array"""
    def __init__(self, data, shape):
        self.data = data
        self.shape = shape

    def copy(self):
        return MockArray(self.data.copy(), self.shape)


# Monkey patch numpy for demo
class MockNP:
    random = MockNumPy
    uint8 = int
    float16 = float

    @staticmethod
    def array(data):
        if isinstance(data, list):
            return MockArray(data, (len(data),))
        return data

    @staticmethod
    def dot(a, b):
        # Simple dot product mock
        return sum(x * y for x, y in zip(a.data[:len(b.data)], b.data))

    @staticmethod
    def linalg_norm(arr):
        # Simple norm calculation
        return sum(x * x for x in arr.data) ** 0.5

# Patch the missing modules
sys.modules['numpy'] = MockNP
sys.modules['cv2'] = type('MockCV2', (), {})()


def print_header():
    """Print demo header"""
    print("=" * 60)
    print("LFM2-VL Vision Recognition POC - Minimal Demo")
    print("=" * 60)
    print()


def print_system_info():
    """Print system information"""
    print("System Information:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Platform: {sys.platform}")
    print(f"  Working directory: {os.getcwd()}")
    print()


def test_core_imports():
    """Test importing core modules"""
    print("Testing core module imports...")

    try:
        from vision_pipeline import VisionPipeline, ProcessingMode
        print("  ✓ vision_pipeline imported successfully")

        from model_optimizer import ModelOptimizer
        print("  ✓ model_optimizer imported successfully")

        return True

    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_pipeline_initialization():
    """Test pipeline initialization"""
    print("\nTesting pipeline initialization...")

    try:
        from vision_pipeline import VisionPipeline, ProcessingMode

        # Test different modes
        modes = [ProcessingMode.REALTIME, ProcessingMode.BATCH, ProcessingMode.EDGE]

        for mode in modes:
            pipeline = VisionPipeline(processing_mode=mode)
            print(f"  ✓ {mode.value} mode initialized")

        return True

    except Exception as e:
        print(f"  ✗ Initialization failed: {e}")
        return False


def test_image_processing():
    """Test basic image processing"""
    print("\nTesting image processing...")

    try:
        from vision_pipeline import VisionPipeline, ProcessingMode

        pipeline = VisionPipeline(processing_mode=ProcessingMode.REALTIME)

        # Create mock image data
        mock_image = MockNP.random.random_randint(0, 255, (480, 640, 3))

        start_time = time.time()
        results = pipeline.process_image(mock_image)
        processing_time = time.time() - start_time

        print(f"  ✓ Image processed in {processing_time:.3f}s")
        print(f"  ✓ Found {len(results['faces'])} faces")
        print(f"  ✓ Found {len(results['objects'])} objects")

        # Test performance stats
        stats = pipeline.get_performance_stats()
        print(f"  ✓ Performance tracking: {stats['total_processed']} images processed")

        return True

    except Exception as e:
        print(f"  ✗ Processing failed: {e}")
        return False


def test_model_optimization():
    """Test model optimization"""
    print("\nTesting model optimization...")

    try:
        from model_optimizer import ModelOptimizer, OptimizationTarget

        optimizer = ModelOptimizer(target_device="mobile")
        print("  ✓ Optimizer initialized for mobile")

        # Mock optimization
        mock_model = "mock_model_data"
        optimized_model, result = optimizer.optimize(mock_model)

        print(f"  ✓ Model optimized: {result.compression_ratio:.2f}x compression")
        print(f"  ✓ Estimated speedup: {result.estimated_speedup:.2f}x")

        # Test report generation
        report = optimizer.generate_optimization_report()
        print(f"  ✓ Optimization report generated")

        return True

    except Exception as e:
        print(f"  ✗ Optimization failed: {e}")
        return False


def test_batch_processing():
    """Test batch processing"""
    print("\nTesting batch processing...")

    try:
        from vision_pipeline import VisionPipeline, ProcessingMode

        pipeline = VisionPipeline(processing_mode=ProcessingMode.BATCH)

        # Create multiple mock images
        mock_images = [
            MockNP.random.random_randint(0, 255, (320, 320, 3))
            for _ in range(3)
        ]

        start_time = time.time()
        results = pipeline.process_batch(mock_images)
        batch_time = time.time() - start_time

        print(f"  ✓ Batch of {len(mock_images)} images processed in {batch_time:.3f}s")
        print(f"  ✓ Average per image: {batch_time/len(mock_images):.3f}s")

        total_detections = sum(
            len(r['faces']) + len(r['objects'])
            for r in results
        )
        print(f"  ✓ Total detections across batch: {total_detections}")

        return True

    except Exception as e:
        print(f"  ✗ Batch processing failed: {e}")
        return False


def show_project_structure():
    """Show project structure"""
    print("\nProject Structure:")

    structure = {
        "core/": "Core processing modules",
        "examples/": "Demo scripts and examples",
        "mobile/": "iOS and Android integration",
        "tests/": "Unit tests and test fixtures",
        "models/": "Model files (mock)",
        "data/": "Test data and images",
        "config/": "Configuration files"
    }

    for path, description in structure.items():
        exists = "✓" if Path(path).exists() else "✗"
        print(f"  {exists} {path:<15} - {description}")


def show_usage_examples():
    """Show usage examples"""
    print("\nUsage Examples:")
    print()

    examples = [
        ("Basic image processing", "python examples/demo.py --demo-type image"),
        ("Batch processing", "python examples/demo.py --demo-type batch"),
        ("Model optimization", "python examples/demo.py --demo-type optimize"),
        ("Performance benchmark", "python examples/demo.py --demo-type benchmark"),
        ("Interactive demo (Windows)", "run_demo.bat"),
        ("Interactive demo (Linux/Mac)", "./run_demo.sh")
    ]

    for description, command in examples:
        print(f"  {description}:")
        print(f"    {command}")
        print()


def main():
    """Run minimal demo"""
    print_header()
    print_system_info()

    # Run tests
    tests = [
        ("Core Imports", test_core_imports),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Image Processing", test_image_processing),
        ("Model Optimization", test_model_optimization),
        ("Batch Processing", test_batch_processing)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ {test_name} crashed: {e}")
            failed += 1

    # Show results
    print("\n" + "=" * 60)
    print("Demo Results:")
    print(f"  Tests passed: {passed}")
    print(f"  Tests failed: {failed}")
    print(f"  Success rate: {passed/(passed+failed)*100:.1f}%")

    if failed == 0:
        print("\n🎉 All tests passed! POC is working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check error messages above.")

    # Show additional info
    show_project_structure()
    show_usage_examples()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)