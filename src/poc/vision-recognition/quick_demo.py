#!/usr/bin/env python3
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

        print("\nDemo completed successfully!")

    except Exception as e:
        print(f"ERROR: {e}")
        return False

    return True

if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)
