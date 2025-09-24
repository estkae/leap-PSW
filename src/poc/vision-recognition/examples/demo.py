#!/usr/bin/env python3
"""
LFM2-VL Vision Recognition Demo
Demonstrates face and object detection capabilities
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
import cv2
import argparse
import json
from datetime import datetime
from pathlib import Path

from vision_pipeline import VisionPipeline, ProcessingMode
from model_optimizer import ModelOptimizer, OptimizationTarget, OptimizationLevel


def create_demo_image():
    """Create a synthetic demo image for testing"""
    # Create a simple test image with basic shapes
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add background gradient
    for y in range(480):
        for x in range(640):
            image[y, x] = [
                int(50 + (y / 480) * 100),  # Blue gradient
                int(30 + (x / 640) * 80),   # Green gradient
                int(40 + ((x + y) / 1120) * 120)  # Red gradient
            ]

    # Add some geometric shapes to simulate objects
    # Rectangle (simulating object)
    cv2.rectangle(image, (100, 100), (200, 200), (0, 255, 0), 2)
    cv2.putText(image, "Object 1", (105, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Circle (simulating face)
    cv2.circle(image, (400, 150), 50, (0, 0, 255), 2)
    cv2.putText(image, "Face", (375, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Another rectangle
    cv2.rectangle(image, (300, 300), (450, 400), (255, 0, 0), 2)
    cv2.putText(image, "Object 2", (305, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return image


def draw_detections(image, detections):
    """Draw detection results on image"""
    output_image = image.copy()

    for detection in detections['faces']:
        bbox = detection.bbox
        confidence = detection.confidence

        # Draw bounding box for faces (green)
        cv2.rectangle(
            output_image,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
            (0, 255, 0),  # Green for faces
            2
        )

        # Draw label
        label = f"Face {confidence:.2f}"
        cv2.putText(
            output_image,
            label,
            (int(bbox[0]), int(bbox[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1
        )

        # Draw landmarks if available
        if hasattr(detection, 'landmarks') and detection.landmarks:
            for landmark_name, (x, y) in detection.landmarks.items():
                cv2.circle(output_image, (int(x), int(y)), 2, (0, 255, 0), -1)

    for detection in detections['objects']:
        bbox = detection.bbox
        confidence = detection.confidence
        label_text = detection.label

        # Draw bounding box for objects (blue)
        cv2.rectangle(
            output_image,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
            (255, 0, 0),  # Blue for objects
            2
        )

        # Draw label
        label = f"{label_text} {confidence:.2f}"
        cv2.putText(
            output_image,
            label,
            (int(bbox[0]), int(bbox[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            1
        )

    return output_image


def demo_image_processing(args):
    """Demonstrate image processing capabilities"""
    print("🖼️  Starting Image Processing Demo")
    print("=" * 50)

    # Initialize vision pipeline
    config = {
        'max_detections': args.max_detections,
        'confidence_threshold': args.confidence,
        'enable_tracking': False
    }

    processing_mode = ProcessingMode(args.mode)
    pipeline = VisionPipeline(
        model_path=args.model_path,
        processing_mode=processing_mode,
        config=config
    )

    # Load or create image
    if args.image_path:
        if os.path.exists(args.image_path):
            image = cv2.imread(args.image_path)

            # If image loading failed, try BMP alternative
            if image is None:
                bmp_path = args.image_path.replace('.jpg', '.bmp').replace('.png', '.bmp')
                if os.path.exists(bmp_path):
                    print(f"📁 Trying BMP alternative: {bmp_path}")
                    image = cv2.imread(bmp_path)

            if image is not None:
                print(f"📁 Loaded image: {args.image_path}")
            else:
                print(f"⚠️ Could not load image: {args.image_path}")
                print("   Creating synthetic demo image instead...")
                image = create_demo_image()
        else:
            print(f"❌ Image not found: {args.image_path}")
            return
    else:
        image = create_demo_image()
        print("🎨 Created synthetic demo image")

    if image is None:
        print("❌ Failed to load or create image")
        return

    print(f"Image shape: {image.shape}")

    # Process image
    print("\n🔍 Processing image...")
    results = pipeline.process_image(
        image,
        detect_faces=args.detect_faces,
        detect_objects=args.detect_objects,
        return_embeddings=args.embeddings
    )

    # Print results
    print(f"\n📊 Detection Results:")
    print(f"   Faces found: {len(results['faces'])}")
    print(f"   Objects found: {len(results['objects'])}")
    print(f"   Processing time: {results['metadata']['processing_time']:.3f}s")

    # Detailed results
    if args.verbose:
        print("\n📝 Detailed Results:")
        for i, face in enumerate(results['faces']):
            print(f"   Face {i+1}:")
            print(f"     Confidence: {face.confidence:.3f}")
            print(f"     Bounding box: {face.bbox}")
            if hasattr(face, 'age_estimate') and face.age_estimate:
                print(f"     Estimated age: {face.age_estimate}")
            if hasattr(face, 'emotion') and face.emotion:
                print(f"     Emotion: {face.emotion}")

        for i, obj in enumerate(results['objects']):
            print(f"   Object {i+1}:")
            print(f"     Label: {obj.label}")
            print(f"     Confidence: {obj.confidence:.3f}")
            print(f"     Bounding box: {obj.bbox}")

    # Draw results and save
    if args.output_path:
        output_image = draw_detections(image, results)
        cv2.imwrite(args.output_path, output_image)
        print(f"\n💾 Results saved to: {args.output_path}")

    # Performance metrics
    stats = pipeline.get_performance_stats()
    print(f"\n📈 Performance Statistics:")
    print(f"   Total processed: {stats['total_processed']}")
    print(f"   Average time: {stats['average_inference_time']:.3f}s")
    print(f"   Last inference: {stats['last_inference_time']:.3f}s")


def demo_batch_processing(args):
    """Demonstrate batch processing capabilities"""
    print("📦 Starting Batch Processing Demo")
    print("=" * 50)

    # Create multiple demo images
    images = []
    for i in range(5):
        img = create_demo_image()
        # Add some variation
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        images.append(img)

    print(f"Created {len(images)} test images")

    # Initialize pipeline for batch processing
    pipeline = VisionPipeline(
        processing_mode=ProcessingMode.BATCH,
        config={'confidence_threshold': args.confidence}
    )

    # Process batch
    print("\n🔄 Processing batch...")
    results = pipeline.process_batch(
        images,
        detect_faces=args.detect_faces,
        detect_objects=args.detect_objects
    )

    # Summary results
    total_faces = sum(len(r['faces']) for r in results)
    total_objects = sum(len(r['objects']) for r in results)
    total_time = sum(r['metadata']['processing_time'] for r in results)

    print(f"\n📊 Batch Results:")
    print(f"   Total faces: {total_faces}")
    print(f"   Total objects: {total_objects}")
    print(f"   Total processing time: {total_time:.3f}s")
    print(f"   Average per image: {total_time/len(images):.3f}s")


def demo_optimization(args):
    """Demonstrate model optimization for edge deployment"""
    print("⚡ Starting Optimization Demo")
    print("=" * 50)

    # Initialize optimizer
    target_device = args.target_device or "mobile"
    optimizer = ModelOptimizer(target_device=target_device)

    # Mock model for demonstration
    mock_model = "mock_lfm2_vl_model"
    print(f"🤖 Optimizing model for: {target_device}")

    # Optimize model
    optimized_model, result = optimizer.optimize(mock_model)

    # Print results
    print(f"\n📊 Optimization Results:")
    print(f"   Original size: {result.original_size_mb:.1f} MB")
    print(f"   Optimized size: {result.optimized_size_mb:.1f} MB")
    print(f"   Compression ratio: {result.compression_ratio:.2f}x")
    print(f"   Estimated speedup: {result.estimated_speedup:.2f}x")
    print(f"   Accuracy impact: {result.accuracy_impact*100:.2f}%")

    print(f"\n🔧 Applied optimizations:")
    for step in result.optimization_steps:
        print(f"   ✓ {step}")

    if result.warnings:
        print(f"\n⚠️  Warnings:")
        for warning in result.warnings:
            print(f"   ⚠️  {warning}")

    # Generate detailed report
    report = optimizer.generate_optimization_report()

    if args.output_path:
        report_path = args.output_path.replace('.jpg', '_optimization_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Detailed report saved to: {report_path}")


def demo_benchmark():
    """Run performance benchmarks"""
    print("🏁 Starting Performance Benchmark")
    print("=" * 50)

    # Create test data
    test_images = [create_demo_image() for _ in range(10)]

    # Test different modes
    modes = [ProcessingMode.REALTIME, ProcessingMode.BATCH, ProcessingMode.EDGE]

    results = {}

    for mode in modes:
        print(f"\n⏱️  Benchmarking {mode.value} mode...")

        pipeline = VisionPipeline(
            processing_mode=mode,
            config={'max_detections': 5}
        )

        # Warm-up
        pipeline.process_image(test_images[0])

        # Benchmark
        start_time = datetime.now()
        for img in test_images:
            pipeline.process_image(img)
        end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()
        fps = len(test_images) / duration

        stats = pipeline.get_performance_stats()

        results[mode.value] = {
            'total_time': duration,
            'fps': fps,
            'avg_inference': stats['average_inference_time'],
            'images_processed': len(test_images)
        }

        print(f"   Total time: {duration:.3f}s")
        print(f"   FPS: {fps:.1f}")
        print(f"   Avg inference: {stats['average_inference_time']:.3f}s")

    # Compare results
    print(f"\n📈 Benchmark Comparison:")
    print(f"{'Mode':<12} {'FPS':<8} {'Avg Time':<12} {'Efficiency':<12}")
    print("-" * 48)

    best_fps = max(results[mode]['fps'] for mode in results)

    for mode, stats in results.items():
        efficiency = (stats['fps'] / best_fps) * 100
        print(f"{mode:<12} {stats['fps']:<8.1f} {stats['avg_inference']:<12.3f} {efficiency:<12.1f}%")


def main():
    parser = argparse.ArgumentParser(description='LFM2-VL Vision Recognition Demo')
    parser.add_argument('--mode', choices=['realtime', 'batch', 'edge'],
                       default='realtime', help='Processing mode')
    parser.add_argument('--image-path', type=str, help='Path to input image')
    parser.add_argument('--model-path', type=str, help='Path to LFM2-VL model')
    parser.add_argument('--output-path', type=str, help='Path to save output')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--max-detections', type=int, default=10,
                       help='Maximum detections per image')
    parser.add_argument('--detect-faces', action='store_true', default=True,
                       help='Enable face detection')
    parser.add_argument('--detect-objects', action='store_true', default=True,
                       help='Enable object detection')
    parser.add_argument('--embeddings', action='store_true',
                       help='Generate face embeddings')
    parser.add_argument('--target-device', choices=['mobile', 'ios', 'android', 'edge', 'tpu'],
                       help='Target device for optimization')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--demo-type', choices=['image', 'batch', 'optimize', 'benchmark'],
                       default='image', help='Demo type to run')

    args = parser.parse_args()

    print("🚀 LFM2-VL Vision Recognition Demo")
    print("=" * 50)
    print(f"Demo type: {args.demo_type}")
    print(f"Processing mode: {args.mode}")
    print(f"Confidence threshold: {args.confidence}")
    print()

    try:
        if args.demo_type == 'image':
            demo_image_processing(args)
        elif args.demo_type == 'batch':
            demo_batch_processing(args)
        elif args.demo_type == 'optimize':
            demo_optimization(args)
        elif args.demo_type == 'benchmark':
            demo_benchmark()

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

    print("\n✅ Demo completed!")


if __name__ == '__main__':
    main()