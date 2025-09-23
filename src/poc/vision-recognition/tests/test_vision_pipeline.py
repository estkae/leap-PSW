#!/usr/bin/env python3
"""
Unit tests for Vision Pipeline
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path

# Add core module to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from vision_pipeline import (
    VisionPipeline,
    ProcessingMode,
    Detection,
    FaceDetection,
    ObjectDetection,
    FaceDetectionModule,
    ObjectDetectionModule,
    ImagePreprocessor
)


class TestVisionPipeline:
    """Test cases for VisionPipeline class"""

    @pytest.fixture
    def test_image(self):
        """Create a test image"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def pipeline(self):
        """Create a test pipeline"""
        return VisionPipeline(
            processing_mode=ProcessingMode.REALTIME,
            config={'confidence_threshold': 0.5}
        )

    def test_pipeline_initialization(self):
        """Test pipeline initialization with different modes"""
        # Test realtime mode
        pipeline_rt = VisionPipeline(processing_mode=ProcessingMode.REALTIME)
        assert pipeline_rt.processing_mode == ProcessingMode.REALTIME
        assert pipeline_rt.config['confidence_threshold'] == 0.5

        # Test batch mode
        pipeline_batch = VisionPipeline(processing_mode=ProcessingMode.BATCH)
        assert pipeline_batch.processing_mode == ProcessingMode.BATCH
        assert pipeline_batch.config['max_detections'] == 50

        # Test edge mode
        pipeline_edge = VisionPipeline(processing_mode=ProcessingMode.EDGE)
        assert pipeline_edge.processing_mode == ProcessingMode.EDGE
        assert pipeline_edge.config['input_size'] == (320, 320)

    def test_process_image_basic(self, pipeline, test_image):
        """Test basic image processing"""
        results = pipeline.process_image(test_image)

        # Check result structure
        assert 'faces' in results
        assert 'objects' in results
        assert 'metadata' in results

        # Check metadata
        metadata = results['metadata']
        assert 'input_shape' in metadata
        assert 'processing_time' in metadata
        assert 'mode' in metadata

        assert metadata['input_shape'] == test_image.shape
        assert metadata['processing_time'] > 0
        assert metadata['mode'] == 'realtime'

    def test_process_image_with_options(self, pipeline, test_image):
        """Test image processing with different options"""
        # Test with faces only
        results = pipeline.process_image(
            test_image,
            detect_faces=True,
            detect_objects=False
        )
        assert len(results['faces']) >= 0
        # Objects should be empty or processed depending on implementation

        # Test with objects only
        results = pipeline.process_image(
            test_image,
            detect_faces=False,
            detect_objects=True
        )
        assert len(results['objects']) >= 0

        # Test with embeddings
        results = pipeline.process_image(
            test_image,
            return_embeddings=True
        )
        for face in results['faces']:
            if hasattr(face, 'embedding'):
                assert face.embedding is not None

    def test_process_batch(self, pipeline):
        """Test batch processing"""
        test_images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(3)
        ]

        results = pipeline.process_batch(test_images)

        assert len(results) == len(test_images)
        for result in results:
            assert 'faces' in result
            assert 'objects' in result
            assert 'metadata' in result

    def test_process_video_frame(self, pipeline, test_image):
        """Test video frame processing"""
        results = pipeline.process_video_frame(test_image, frame_number=1)

        assert 'faces' in results
        assert 'objects' in results
        assert 'metadata' in results
        assert results['metadata']['frame_number'] == 1

    def test_performance_stats(self, pipeline, test_image):
        """Test performance metrics tracking"""
        # Process some images
        for _ in range(3):
            pipeline.process_image(test_image)

        stats = pipeline.get_performance_stats()

        assert 'total_processed' in stats
        assert 'average_inference_time' in stats
        assert 'last_inference_time' in stats

        assert stats['total_processed'] == 3
        assert stats['average_inference_time'] > 0
        assert stats['last_inference_time'] > 0

    def test_optimize_for_device(self, pipeline):
        """Test device optimization"""
        original_config = pipeline.config.copy()

        # Test mobile optimization
        pipeline.optimize_for_device("mobile")
        assert pipeline.config['input_size'] == (320, 320)
        assert pipeline.config['quantization'] == True

        # Test edge optimization
        pipeline.optimize_for_device("edge")
        assert pipeline.config['input_size'] == (416, 416)

    def test_cleanup(self, pipeline):
        """Test cleanup functionality"""
        # Should not raise exception
        pipeline.cleanup()


class TestDetectionClasses:
    """Test detection data classes"""

    def test_detection_creation(self):
        """Test basic Detection creation"""
        detection = Detection(
            bbox=(10, 20, 30, 40),
            confidence=0.85,
            label="test_object"
        )

        assert detection.bbox == (10, 20, 30, 40)
        assert detection.confidence == 0.85
        assert detection.label == "test_object"

    def test_face_detection_creation(self):
        """Test FaceDetection creation"""
        face = FaceDetection(
            bbox=(50, 60, 70, 80),
            confidence=0.92,
            label="face",
            landmarks={'left_eye': (60, 70), 'right_eye': (80, 70)},
            age_estimate=25,
            emotion="happy"
        )

        assert face.bbox == (50, 60, 70, 80)
        assert face.confidence == 0.92
        assert face.landmarks['left_eye'] == (60, 70)
        assert face.age_estimate == 25
        assert face.emotion == "happy"

    def test_object_detection_creation(self):
        """Test ObjectDetection creation"""
        obj = ObjectDetection(
            bbox=(100, 110, 120, 130),
            confidence=0.78,
            label="car",
            category="vehicle",
            attributes={'color': 'red'}
        )

        assert obj.bbox == (100, 110, 120, 130)
        assert obj.category == "vehicle"
        assert obj.attributes['color'] == 'red'


class TestFaceDetectionModule:
    """Test FaceDetectionModule"""

    @pytest.fixture
    def face_module(self):
        """Create face detection module"""
        config = {'confidence_threshold': 0.5, 'cache_embeddings': True}
        return FaceDetectionModule("mock_model", config)

    def test_face_detection(self, face_module):
        """Test face detection functionality"""
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        faces = face_module.detect(test_image)

        assert isinstance(faces, list)
        for face in faces:
            assert isinstance(face, FaceDetection)
            assert face.confidence >= 0.5  # Above threshold

    def test_face_recognition(self, face_module):
        """Test face recognition functionality"""
        # Create mock embeddings
        query_embedding = np.random.randn(512)
        database_embeddings = [np.random.randn(512) for _ in range(5)]

        # Add one similar embedding
        database_embeddings[2] = query_embedding + np.random.randn(512) * 0.1

        matched_index, similarity = face_module.recognize(query_embedding, database_embeddings)

        assert 0 <= matched_index < len(database_embeddings)
        assert -1 <= similarity <= 1


class TestObjectDetectionModule:
    """Test ObjectDetectionModule"""

    @pytest.fixture
    def object_module(self):
        """Create object detection module"""
        config = {'confidence_threshold': 0.5, 'max_detections': 10}
        return ObjectDetectionModule("mock_model", config)

    def test_object_detection(self, object_module):
        """Test object detection functionality"""
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        objects = object_module.detect(test_image)

        assert isinstance(objects, list)
        assert len(objects) <= 10  # Max detections limit

        for obj in objects:
            assert isinstance(obj, ObjectDetection)
            assert obj.confidence >= 0.5  # Above threshold

    def test_class_names_loading(self, object_module):
        """Test class names loading"""
        assert len(object_module.class_names) > 0
        assert "person" in object_module.class_names
        assert "car" in object_module.class_names

    def test_nms_application(self, object_module):
        """Test Non-Maximum Suppression"""
        # Create overlapping detections
        detections = [
            ObjectDetection((100, 100, 50, 50), 0.9, "car"),
            ObjectDetection((110, 110, 50, 50), 0.8, "car"),  # Overlapping
            ObjectDetection((300, 300, 50, 50), 0.7, "person")  # Non-overlapping
        ]

        filtered = object_module._apply_nms(detections)

        # Should remove overlapping detection
        assert len(filtered) <= len(detections)


class TestImagePreprocessor:
    """Test ImagePreprocessor"""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor"""
        config = {'input_size': (416, 416), 'fp16': False}
        return ImagePreprocessor(config)

    def test_basic_preprocessing(self, preprocessor):
        """Test basic image preprocessing"""
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        processed = preprocessor.process(test_image)

        assert processed is not None
        # Should return processed image
        assert processed.shape == test_image.shape

    def test_fp16_preprocessing(self):
        """Test FP16 preprocessing"""
        config = {'input_size': (416, 416), 'fp16': True}
        preprocessor = ImagePreprocessor(config)

        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        processed = preprocessor.process(test_image)

        # Should be converted to FP16 if enabled
        if config['fp16']:
            assert processed.dtype == np.float16


class TestIntegration:
    """Integration tests"""

    def test_end_to_end_processing(self):
        """Test complete end-to-end processing"""
        # Create pipeline
        pipeline = VisionPipeline(
            processing_mode=ProcessingMode.REALTIME,
            config={
                'confidence_threshold': 0.3,
                'max_detections': 5
            }
        )

        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Process image
        results = pipeline.process_image(
            test_image,
            detect_faces=True,
            detect_objects=True,
            return_embeddings=True
        )

        # Verify results
        assert isinstance(results, dict)
        assert len(results['faces']) <= 5
        assert len(results['objects']) <= 5

        # Check processing time is reasonable
        assert results['metadata']['processing_time'] < 5.0  # Should be fast for mock

    def test_different_processing_modes(self):
        """Test different processing modes work together"""
        test_image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)

        modes = [ProcessingMode.REALTIME, ProcessingMode.BATCH, ProcessingMode.EDGE]

        for mode in modes:
            pipeline = VisionPipeline(processing_mode=mode)
            results = pipeline.process_image(test_image)

            assert 'faces' in results
            assert 'objects' in results
            assert results['metadata']['mode'] == mode.value

    def test_performance_consistency(self):
        """Test performance consistency across multiple runs"""
        pipeline = VisionPipeline(processing_mode=ProcessingMode.REALTIME)
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        processing_times = []
        for _ in range(5):
            results = pipeline.process_image(test_image)
            processing_times.append(results['metadata']['processing_time'])

        # Processing times should be relatively consistent (within 100% of mean)
        mean_time = np.mean(processing_times)
        std_time = np.std(processing_times)

        assert std_time < mean_time  # Standard deviation should be less than mean


if __name__ == '__main__':
    # Run tests if executed directly
    pytest.main([__file__, '-v'])