package com.leap.vision

import android.content.Context
import android.graphics.*
import android.media.Image
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import kotlin.math.min

/**
 * LFM2-VL Vision Processor for Android
 * Handles face and object detection using optimized TensorFlow Lite models
 */
class LFM2VLProcessor(private val context: Context) {

    companion object {
        private const val TAG = "LFM2VLProcessor"
        private const val MODEL_FILE = "lfm2_vl_mobile.tflite"
        private const val LABEL_FILE = "labels.txt"
        private const val NUM_THREADS = 4
        private const val INPUT_SIZE = 416
        private const val IS_QUANTIZED = true
        private const val NUM_DETECTIONS = 10
    }

    // Model and interpreter
    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()

    // Delegates for acceleration
    private var gpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: NnApiDelegate? = null

    // Processing configuration
    var configuration = ProcessingConfiguration()

    // Performance metrics
    private val performanceMetrics = PerformanceMetrics()

    // Thread pool for parallel processing
    private val executorService = Executors.newFixedThreadPool(2)

    /**
     * Processing configuration
     */
    data class ProcessingConfiguration(
        var maxDetections: Int = 10,
        var confidenceThreshold: Float = 0.5f,
        var nmsThreshold: Float = 0.4f,
        var enableFaceDetection: Boolean = true,
        var enableObjectDetection: Boolean = true,
        var processingMode: ProcessingMode = ProcessingMode.REALTIME,
        var useGpu: Boolean = false,
        var useNnapi: Boolean = true
    )

    /**
     * Processing modes
     */
    enum class ProcessingMode {
        REALTIME,    // For camera preview
        BATCH,       // For multiple images
        LOW_POWER    // Battery saving mode
    }

    /**
     * Detection result
     */
    data class Detection(
        val boundingBox: RectF,
        val confidence: Float,
        val label: String,
        val type: DetectionType,
        val metadata: Map<String, Any>? = null
    )

    /**
     * Detection types
     */
    enum class DetectionType {
        FACE,
        OBJECT
    }

    /**
     * Performance metrics
     */
    data class PerformanceMetrics(
        var totalProcessed: Int = 0,
        var averageInferenceTime: Long = 0,
        var lastInferenceTime: Long = 0
    ) {
        fun update(inferenceTime: Long) {
            totalProcessed++
            averageInferenceTime = ((averageInferenceTime * (totalProcessed - 1)) + inferenceTime) / totalProcessed
            lastInferenceTime = inferenceTime
        }
    }

    /**
     * Initialize the processor and load model
     */
    fun initialize() {
        try {
            // Load labels
            labels = FileUtil.loadLabels(context, LABEL_FILE)

            // Load model
            val modelBuffer = FileUtil.loadMappedFile(context, MODEL_FILE)

            // Create interpreter options
            val options = Interpreter.Options().apply {
                setNumThreads(NUM_THREADS)

                // Add GPU delegate if requested
                if (configuration.useGpu) {
                    gpuDelegate = GpuDelegate()
                    addDelegate(gpuDelegate)
                    Log.d(TAG, "GPU acceleration enabled")
                }

                // Add NNAPI delegate if requested
                if (configuration.useNnapi) {
                    nnApiDelegate = NnApiDelegate()
                    addDelegate(nnApiDelegate)
                    Log.d(TAG, "NNAPI acceleration enabled")
                }
            }

            // Create interpreter
            interpreter = Interpreter(modelBuffer, options)

            Log.i(TAG, "✅ Model loaded successfully")
            Log.i(TAG, "   Input tensor: ${interpreter?.getInputTensor(0)?.shape()?.contentToString()}")
            Log.i(TAG, "   Output tensors: ${interpreter?.outputTensorCount}")

        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to initialize model", e)
            throw RuntimeException("Model initialization failed", e)
        }
    }

    /**
     * Process a single image
     */
    fun processImage(bitmap: Bitmap, callback: (List<Detection>) -> Unit) {
        executorService.execute {
            val startTime = SystemClock.elapsedRealtime()

            try {
                // Preprocess image
                val inputTensor = preprocessImage(bitmap)

                // Run inference
                val outputs = runInference(inputTensor)

                // Process results
                val detections = processResults(outputs, bitmap.width, bitmap.height)

                // Update metrics
                val inferenceTime = SystemClock.elapsedRealtime() - startTime
                performanceMetrics.update(inferenceTime)

                // Return results
                callback(detections)

                Log.d(TAG, "Processed image in ${inferenceTime}ms, found ${detections.size} detections")

            } catch (e: Exception) {
                Log.e(TAG, "Error processing image", e)
                callback(emptyList())
            }
        }
    }

    /**
     * Process camera frame
     */
    fun processCameraFrame(image: Image, rotation: Int, callback: (List<Detection>) -> Unit) {
        executorService.execute {
            val startTime = SystemClock.elapsedRealtime()

            try {
                // Convert Image to Bitmap
                val bitmap = imageToBitmap(image)

                // Apply rotation if needed
                val rotatedBitmap = rotateBitmap(bitmap, rotation)

                // Preprocess
                val inputTensor = preprocessImage(rotatedBitmap)

                // Run inference
                val outputs = runInference(inputTensor)

                // Process results
                val detections = processResults(outputs, rotatedBitmap.width, rotatedBitmap.height)

                // For realtime mode, skip frames if processing is slow
                if (configuration.processingMode == ProcessingMode.REALTIME &&
                    performanceMetrics.lastInferenceTime > 33) { // Less than 30fps
                    // Could implement frame skipping logic here
                }

                // Update metrics
                val inferenceTime = SystemClock.elapsedRealtime() - startTime
                performanceMetrics.update(inferenceTime)

                // Return results
                callback(detections)

            } catch (e: Exception) {
                Log.e(TAG, "Error processing camera frame", e)
                callback(emptyList())
            }
        }
    }

    /**
     * Preprocess image for model input
     */
    private fun preprocessImage(bitmap: Bitmap): TensorImage {
        // Create TensorImage
        var tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)

        // Create image processor
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(INPUT_SIZE, INPUT_SIZE))
            .add(ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(127.5f, 127.5f)) // Normalize to [-1, 1]
            .build()

        // Process image
        tensorImage = imageProcessor.process(tensorImage)

        return tensorImage
    }

    /**
     * Run model inference
     */
    private fun runInference(input: TensorImage): Map<Int, Any> {
        val interpreter = this.interpreter ?: throw IllegalStateException("Model not initialized")

        // Prepare output buffers
        val outputs = HashMap<Int, Any>()

        // Output tensor 0: Detection boxes [1, NUM_DETECTIONS, 4]
        val boxes = Array(1) { Array(NUM_DETECTIONS) { FloatArray(4) } }
        outputs[0] = boxes

        // Output tensor 1: Detection classes [1, NUM_DETECTIONS]
        val classes = Array(1) { FloatArray(NUM_DETECTIONS) }
        outputs[1] = classes

        // Output tensor 2: Detection scores [1, NUM_DETECTIONS]
        val scores = Array(1) { FloatArray(NUM_DETECTIONS) }
        outputs[2] = scores

        // Output tensor 3: Number of detections [1]
        val numDetections = FloatArray(1)
        outputs[3] = numDetections

        // Run inference
        interpreter.runForMultipleInputsOutputs(arrayOf(input.buffer), outputs)

        return outputs
    }

    /**
     * Process model outputs to detections
     */
    private fun processResults(
        outputs: Map<Int, Any>,
        imageWidth: Int,
        imageHeight: Int
    ): List<Detection> {
        val detections = mutableListOf<Detection>()

        // Extract outputs
        val boxes = outputs[0] as Array<Array<FloatArray>>
        val classes = outputs[1] as Array<FloatArray>
        val scores = outputs[2] as Array<FloatArray>
        val numDetections = (outputs[3] as FloatArray)[0].toInt()

        // Process each detection
        for (i in 0 until min(numDetections, NUM_DETECTIONS)) {
            val score = scores[0][i]

            // Filter by confidence threshold
            if (score < configuration.confidenceThreshold) continue

            // Get bounding box
            val box = boxes[0][i]
            val boundingBox = RectF(
                box[1] * imageWidth,  // left
                box[0] * imageHeight, // top
                box[3] * imageWidth,  // right
                box[2] * imageHeight  // bottom
            )

            // Get class label
            val classIndex = classes[0][i].toInt()
            val label = if (classIndex < labels.size) labels[classIndex] else "unknown"

            // Determine detection type
            val type = if (label == "face" || label == "person") {
                DetectionType.FACE
            } else {
                DetectionType.OBJECT
            }

            // Create detection
            val detection = Detection(
                boundingBox = boundingBox,
                confidence = score,
                label = label,
                type = type,
                metadata = mapOf(
                    "classIndex" to classIndex,
                    "processingMode" to configuration.processingMode.name
                )
            )

            detections.add(detection)
        }

        // Apply NMS if needed
        return applyNMS(detections)
    }

    /**
     * Apply Non-Maximum Suppression
     */
    private fun applyNMS(detections: List<Detection>): List<Detection> {
        if (detections.size <= 1) return detections

        val sorted = detections.sortedByDescending { it.confidence }
        val selected = mutableListOf<Detection>()

        for (detection in sorted) {
            var shouldAdd = true

            for (selected_detection in selected) {
                val iou = calculateIoU(detection.boundingBox, selected_detection.boundingBox)
                if (iou > configuration.nmsThreshold) {
                    shouldAdd = false
                    break
                }
            }

            if (shouldAdd) {
                selected.add(detection)
                if (selected.size >= configuration.maxDetections) break
            }
        }

        return selected
    }

    /**
     * Calculate Intersection over Union
     */
    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val intersection = RectF()
        val intersects = intersection.setIntersect(box1, box2)

        if (!intersects) return 0f

        val intersectionArea = intersection.width() * intersection.height()
        val box1Area = box1.width() * box1.height()
        val box2Area = box2.width() * box2.height()
        val unionArea = box1Area + box2Area - intersectionArea

        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }

    /**
     * Convert Image to Bitmap
     */
    private fun imageToBitmap(image: Image): Bitmap {
        val planes = image.planes
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]

        val ySize = yPlane.buffer.remaining()
        val uSize = uPlane.buffer.remaining()
        val vSize = vPlane.buffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yPlane.buffer.get(nv21, 0, ySize)
        vPlane.buffer.get(nv21, ySize, vSize)
        uPlane.buffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
        val imageBytes = out.toByteArray()

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    /**
     * Rotate bitmap
     */
    private fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap

        val matrix = Matrix()
        matrix.postRotate(degrees.toFloat())

        return Bitmap.createBitmap(
            bitmap,
            0,
            0,
            bitmap.width,
            bitmap.height,
            matrix,
            true
        )
    }

    /**
     * Draw detections on bitmap
     */
    fun drawDetections(bitmap: Bitmap, detections: List<Detection>): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val paint = Paint()

        for (detection in detections) {
            // Set color based on type
            paint.color = when (detection.type) {
                DetectionType.FACE -> Color.GREEN
                DetectionType.OBJECT -> Color.BLUE
            }
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 3f

            // Draw bounding box
            canvas.drawRect(detection.boundingBox, paint)

            // Draw label
            paint.style = Paint.Style.FILL
            paint.textSize = 40f
            val label = "${detection.label} ${(detection.confidence * 100).toInt()}%"
            canvas.drawText(
                label,
                detection.boundingBox.left,
                detection.boundingBox.top - 10,
                paint
            )
        }

        return mutableBitmap
    }

    /**
     * Get performance metrics
     */
    fun getPerformanceMetrics(): PerformanceMetrics = performanceMetrics

    /**
     * Optimize for specific mode
     */
    fun optimizeFor(mode: ProcessingMode) {
        configuration.processingMode = mode

        when (mode) {
            ProcessingMode.REALTIME -> {
                configuration.maxDetections = 5
                configuration.confidenceThreshold = 0.6f
                configuration.useGpu = true
            }
            ProcessingMode.BATCH -> {
                configuration.maxDetections = 20
                configuration.confidenceThreshold = 0.4f
                configuration.useNnapi = true
            }
            ProcessingMode.LOW_POWER -> {
                configuration.maxDetections = 3
                configuration.confidenceThreshold = 0.7f
                configuration.useGpu = false
                configuration.useNnapi = false
            }
        }

        // Reinitialize with new configuration
        close()
        initialize()
    }

    /**
     * Clean up resources
     */
    fun close() {
        interpreter?.close()
        interpreter = null

        gpuDelegate?.close()
        gpuDelegate = null

        nnApiDelegate?.close()
        nnApiDelegate = null

        executorService.shutdown()
    }
}