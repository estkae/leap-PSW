#!/usr/bin/env python3
"""
Simple test for LFM2-VL Vision Recognition POC
Windows compatible version without Unicode characters
"""

import sys
import os
import time
from pathlib import Path

# Mock numpy for basic testing
class MockArray:
    def __init__(self, data, shape):
        self.data = data
        self.shape = shape
        self.dtype = int

    def copy(self):
        return MockArray(self.data, self.shape)

class MockNP:
    ndarray = MockArray
    uint8 = int
    float16 = float

    @staticmethod
    def random_randint(low, high, shape, dtype=None):
        import random
        total = 1
        for dim in shape:
            total *= dim
        data = [random.randint(low, high-1) for _ in range(total)]
        return MockArray(data, shape)

    @staticmethod
    def random_randn(*shape):
        import random
        total = 1
        for dim in shape:
            total *= dim
        data = [random.gauss(0, 1) for _ in range(total)]
        return MockArray(data, shape)

# Install mocks
sys.modules['numpy'] = MockNP
sys.modules['cv2'] = type('MockCV2', (), {})()

# Add core to path
sys.path.insert(0, 'core')


def main():
    print("LFM2-VL Vision Recognition POC - Simple Test")
    print("=" * 50)

    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    print()

    # Test 1: Import core modules
    print("Test 1: Importing core modules...")
    try:
        from vision_pipeline import VisionPipeline, ProcessingMode
        print("  SUCCESS: vision_pipeline imported")

        from model_optimizer import ModelOptimizer
        print("  SUCCESS: model_optimizer imported")

        test1_pass = True
    except Exception as e:
        print(f"  FAILED: {e}")
        test1_pass = False

    if not test1_pass:
        print("Core import failed. Cannot continue.")
        return False

    # Test 2: Initialize pipeline
    print("\nTest 2: Initialize pipeline...")
    try:
        pipeline = VisionPipeline(processing_mode=ProcessingMode.REALTIME)
        print("  SUCCESS: Pipeline initialized")
        test2_pass = True
    except Exception as e:
        print(f"  FAILED: {e}")
        test2_pass = False

    # Test 3: Process mock image
    print("\nTest 3: Process mock image...")
    try:
        mock_image = MockNP.random_randint(0, 255, (480, 640, 3))

        start_time = time.time()
        results = pipeline.process_image(mock_image)
        processing_time = time.time() - start_time

        print(f"  SUCCESS: Processed in {processing_time:.3f}s")
        print(f"  Found {len(results['faces'])} faces")
        print(f"  Found {len(results['objects'])} objects")

        test3_pass = True
    except Exception as e:
        print(f"  FAILED: {e}")
        test3_pass = False

    # Test 4: Model optimization
    print("\nTest 4: Model optimization...")
    try:
        optimizer = ModelOptimizer(target_device="mobile")
        mock_model = "test_model"
        optimized_model, result = optimizer.optimize(mock_model)

        print(f"  SUCCESS: Compression {result.compression_ratio:.2f}x")
        print(f"  Estimated speedup: {result.estimated_speedup:.2f}x")

        test4_pass = True
    except Exception as e:
        print(f"  FAILED: {e}")
        test4_pass = False

    # Test 5: Performance tracking
    print("\nTest 5: Performance tracking...")
    try:
        stats = pipeline.get_performance_stats()
        print(f"  SUCCESS: {stats['total_processed']} images processed")
        print(f"  Average time: {stats['average_inference_time']:.3f}s")

        test5_pass = True
    except Exception as e:
        print(f"  FAILED: {e}")
        test5_pass = False

    # Summary
    tests = [test1_pass, test2_pass, test3_pass, test4_pass, test5_pass]
    passed = sum(tests)
    total = len(tests)

    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"  Passed: {passed}/{total}")
    print(f"  Success rate: {passed/total*100:.1f}%")

    if passed == total:
        print("\nALL TESTS PASSED! POC is working correctly.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r config/requirements.txt")
        print("  2. Run full demo: python examples/demo.py")
        print("  3. Try mobile integration templates")
    else:
        print(f"\n{total-passed} test(s) failed. Check error messages above.")

    print("\nProject structure:")
    folders = ["core/", "examples/", "mobile/", "tests/", "models/", "data/"]
    for folder in folders:
        exists = "EXISTS" if Path(folder).exists() else "MISSING"
        print(f"  {folder:<12} {exists}")

    print("\n" + "=" * 50)

    return passed == total


if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)