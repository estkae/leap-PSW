import CoreML
import Vision
import UIKit
import AVFoundation

/// LFM2-VL Vision Processor for iOS
/// Handles face and object detection using optimized CoreML models
class LFM2VLProcessor: NSObject {

    // MARK: - Properties

    private var visionModel: VNCoreMLModel?
    private var faceDetectionRequest: VNDetectFaceRectanglesRequest?
    private var objectDetectionRequest: VNCoreMLRequest?
    private let processingQueue = DispatchQueue(label: "com.leap.vision.processing", qos: .userInitiated)

    /// Performance metrics
    private var performanceMetrics = PerformanceMetrics()

    /// Processing configuration
    var configuration: ProcessingConfiguration

    // MARK: - Types

    struct ProcessingConfiguration {
        var maxDetections: Int = 10
        var confidenceThreshold: Float = 0.5
        var enableFaceDetection: Bool = true
        var enableObjectDetection: Bool = true
        var processingMode: ProcessingMode = .realtime
        var useNeuralEngine: Bool = true
    }

    enum ProcessingMode {
        case realtime
        case batch
        case lowPower
    }

    struct Detection {
        let boundingBox: CGRect
        let confidence: Float
        let label: String
        let type: DetectionType
        var metadata: [String: Any]?
    }

    enum DetectionType {
        case face
        case object
    }

    struct PerformanceMetrics {
        var totalProcessed: Int = 0
        var averageProcessingTime: Double = 0
        var lastProcessingTime: Double = 0

        mutating func update(processingTime: Double) {
            totalProcessed += 1
            let n = Double(totalProcessed)
            averageProcessingTime = ((averageProcessingTime * (n - 1)) + processingTime) / n
            lastProcessingTime = processingTime
        }
    }

    // MARK: - Initialization

    override init() {
        self.configuration = ProcessingConfiguration()
        super.init()
    }

    /// Initialize with custom configuration
    init(configuration: ProcessingConfiguration) {
        self.configuration = configuration
        super.init()
    }

    /// Load and prepare the LFM2-VL model
    func loadModel(from url: URL) throws {
        // Configure CoreML model options
        let config = MLModelConfiguration()
        if configuration.useNeuralEngine {
            config.computeUnits = .all  // Use Neural Engine when available
        } else {
            config.computeUnits = .cpuOnly
        }

        // Load the model
        let mlModel = try MLModel(contentsOf: url, configuration: config)
        visionModel = try VNCoreMLModel(for: mlModel)

        // Setup requests
        setupVisionRequests()

        print("✅ Model loaded successfully")
        print("   Model URL: \(url.lastPathComponent)")
        print("   Compute Units: \(config.computeUnits == .all ? "Neural Engine + CPU" : "CPU Only")")
    }

    // MARK: - Vision Request Setup

    private func setupVisionRequests() {
        // Face detection request
        if configuration.enableFaceDetection {
            faceDetectionRequest = VNDetectFaceRectanglesRequest { [weak self] request, error in
                self?.handleFaceDetection(request: request, error: error)
            }
            faceDetectionRequest?.revision = VNDetectFaceRectanglesRequestRevision3
        }

        // Object detection request using CoreML model
        if configuration.enableObjectDetection, let model = visionModel {
            objectDetectionRequest = VNCoreMLRequest(model: model) { [weak self] request, error in
                self?.handleObjectDetection(request: request, error: error)
            }
            objectDetectionRequest?.imageCropAndScaleOption = .scaleFit
        }
    }

    // MARK: - Image Processing

    /// Process a single image for face and object detection
    func process(image: UIImage, completion: @escaping ([Detection]) -> Void) {
        guard let cgImage = image.cgImage else {
            completion([])
            return
        }

        let startTime = CFAbsoluteTimeGetCurrent()

        processingQueue.async { [weak self] in
            guard let self = self else { return }

            var detections: [Detection] = []
            let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])

            // Build request array
            var requests: [VNRequest] = []
            if let faceRequest = self.faceDetectionRequest {
                requests.append(faceRequest)
            }
            if let objectRequest = self.objectDetectionRequest {
                requests.append(objectRequest)
            }

            // Perform requests
            do {
                try requestHandler.perform(requests)

                // Collect results
                if let faceResults = self.faceDetectionRequest?.results as? [VNFaceObservation] {
                    detections.append(contentsOf: self.processFaceResults(faceResults))
                }

                if let objectResults = self.objectDetectionRequest?.results as? [VNRecognizedObjectObservation] {
                    detections.append(contentsOf: self.processObjectResults(objectResults))
                }

                // Update metrics
                let processingTime = CFAbsoluteTimeGetCurrent() - startTime
                self.performanceMetrics.update(processingTime: processingTime)

                // Return results on main queue
                DispatchQueue.main.async {
                    completion(detections)
                }

            } catch {
                print("❌ Vision request failed: \(error)")
                DispatchQueue.main.async {
                    completion([])
                }
            }
        }
    }

    /// Process video frame for real-time detection
    func processVideoFrame(_ sampleBuffer: CMSampleBuffer, completion: @escaping ([Detection]) -> Void) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            completion([])
            return
        }

        let startTime = CFAbsoluteTimeGetCurrent()

        processingQueue.async { [weak self] in
            guard let self = self else { return }

            var detections: [Detection] = []
            let requestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])

            // For real-time processing, we might want to skip frames if processing is slow
            if self.configuration.processingMode == .realtime &&
               self.performanceMetrics.lastProcessingTime > 0.033 { // More than 30fps
                // Skip alternate frames for better performance
                if self.performanceMetrics.totalProcessed % 2 == 0 {
                    completion([])
                    return
                }
            }

            // Build and perform requests (similar to image processing)
            var requests: [VNRequest] = []
            if let faceRequest = self.faceDetectionRequest {
                requests.append(faceRequest)
            }
            if let objectRequest = self.objectDetectionRequest {
                requests.append(objectRequest)
            }

            do {
                try requestHandler.perform(requests)

                // Process results
                if let faceResults = self.faceDetectionRequest?.results as? [VNFaceObservation] {
                    detections.append(contentsOf: self.processFaceResults(faceResults))
                }

                if let objectResults = self.objectDetectionRequest?.results as? [VNRecognizedObjectObservation] {
                    detections.append(contentsOf: self.processObjectResults(objectResults))
                }

                // Update metrics
                let processingTime = CFAbsoluteTimeGetCurrent() - startTime
                self.performanceMetrics.update(processingTime: processingTime)

                // Return results
                DispatchQueue.main.async {
                    completion(detections)
                }

            } catch {
                print("❌ Video frame processing failed: \(error)")
                DispatchQueue.main.async {
                    completion([])
                }
            }
        }
    }

    // MARK: - Result Processing

    private func processFaceResults(_ observations: [VNFaceObservation]) -> [Detection] {
        return observations.compactMap { observation in
            guard observation.confidence >= configuration.confidenceThreshold else {
                return nil
            }

            var metadata: [String: Any] = [:]

            // Add face landmarks if available
            if let landmarks = observation.landmarks {
                metadata["hasLandmarks"] = true
                metadata["landmarkCount"] = landmarks.allPoints?.pointCount ?? 0
            }

            // Add additional face attributes
            metadata["roll"] = observation.roll?.floatValue ?? 0
            metadata["yaw"] = observation.yaw?.floatValue ?? 0

            return Detection(
                boundingBox: observation.boundingBox,
                confidence: observation.confidence,
                label: "face",
                type: .face,
                metadata: metadata
            )
        }
    }

    private func processObjectResults(_ observations: [VNRecognizedObjectObservation]) -> [Detection] {
        return observations.prefix(configuration.maxDetections).compactMap { observation in
            guard let topLabel = observation.labels.first,
                  topLabel.confidence >= configuration.confidenceThreshold else {
                return nil
            }

            var metadata: [String: Any] = [
                "allLabels": observation.labels.map { ["label": $0.identifier, "confidence": $0.confidence] }
            ]

            return Detection(
                boundingBox: observation.boundingBox,
                confidence: topLabel.confidence,
                label: topLabel.identifier,
                type: .object,
                metadata: metadata
            )
        }
    }

    // MARK: - Request Handlers

    private func handleFaceDetection(request: VNRequest, error: Error?) {
        if let error = error {
            print("❌ Face detection error: \(error)")
        }
        // Results are processed in the main processing methods
    }

    private func handleObjectDetection(request: VNRequest, error: Error?) {
        if let error = error {
            print("❌ Object detection error: \(error)")
        }
        // Results are processed in the main processing methods
    }

    // MARK: - Performance & Optimization

    /// Get current performance metrics
    func getPerformanceMetrics() -> PerformanceMetrics {
        return performanceMetrics
    }

    /// Reset performance metrics
    func resetMetrics() {
        performanceMetrics = PerformanceMetrics()
    }

    /// Optimize for specific use case
    func optimizeFor(_ mode: ProcessingMode) {
        configuration.processingMode = mode

        switch mode {
        case .realtime:
            configuration.maxDetections = 5
            configuration.confidenceThreshold = 0.6
        case .batch:
            configuration.maxDetections = 20
            configuration.confidenceThreshold = 0.4
        case .lowPower:
            configuration.maxDetections = 3
            configuration.confidenceThreshold = 0.7
            configuration.useNeuralEngine = false  // Save battery
        }
    }
}

// MARK: - UIImage Extension for Vision

extension UIImage {
    /// Convert UIImage coordinates to Vision coordinates
    func visionToUIKit(boundingBox: CGRect) -> CGRect {
        let width = self.size.width
        let height = self.size.height

        let x = boundingBox.origin.x * width
        let y = (1 - boundingBox.origin.y - boundingBox.height) * height
        let rectWidth = boundingBox.width * width
        let rectHeight = boundingBox.height * height

        return CGRect(x: x, y: y, width: rectWidth, height: rectHeight)
    }
}

// MARK: - Demo Usage

class VisionDemoViewController: UIViewController {

    private let processor = LFM2VLProcessor()
    private var imageView: UIImageView!

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        loadModelAndProcess()
    }

    private func setupUI() {
        imageView = UIImageView(frame: view.bounds)
        imageView.contentMode = .scaleAspectFit
        view.addSubview(imageView)
    }

    private func loadModelAndProcess() {
        // Load model (path would be to actual .mlmodel file)
        guard let modelURL = Bundle.main.url(forResource: "LFM2VL_Mobile", withExtension: "mlmodelc") else {
            print("Model not found")
            return
        }

        do {
            try processor.loadModel(from: modelURL)

            // Process sample image
            if let sampleImage = UIImage(named: "sample") {
                processImage(sampleImage)
            }
        } catch {
            print("Failed to load model: \(error)")
        }
    }

    private func processImage(_ image: UIImage) {
        imageView.image = image

        processor.process(image: image) { [weak self] detections in
            self?.drawDetections(detections, on: image)

            // Print results
            print("Found \(detections.count) detections:")
            for detection in detections {
                print("  - \(detection.label): \(String(format: "%.2f", detection.confidence * 100))%")
            }

            // Print performance metrics
            let metrics = self?.processor.getPerformanceMetrics()
            print("Processing time: \(String(format: "%.3f", metrics?.lastProcessingTime ?? 0))s")
        }
    }

    private func drawDetections(_ detections: [LFM2VLProcessor.Detection], on image: UIImage) {
        UIGraphicsBeginImageContextWithOptions(image.size, false, 0)
        image.draw(at: .zero)

        let context = UIGraphicsGetCurrentContext()

        for detection in detections {
            // Convert Vision coordinates to UIKit coordinates
            let rect = image.visionToUIKit(boundingBox: detection.boundingBox)

            // Set color based on type
            let color: UIColor = detection.type == .face ? .green : .blue
            context?.setStrokeColor(color.cgColor)
            context?.setLineWidth(3)
            context?.stroke(rect)

            // Draw label
            let label = "\(detection.label) \(String(format: "%.0f%%", detection.confidence * 100))"
            let attributes: [NSAttributedString.Key: Any] = [
                .foregroundColor: color,
                .font: UIFont.boldSystemFont(ofSize: 14),
                .backgroundColor: UIColor.white.withAlphaComponent(0.7)
            ]
            label.draw(at: CGPoint(x: rect.origin.x, y: rect.origin.y - 20), withAttributes: attributes)
        }

        let annotatedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        imageView.image = annotatedImage
    }
}