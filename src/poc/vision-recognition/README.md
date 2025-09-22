# LFM2-VL Vision Recognition POC

## Übersicht

Dieser Proof of Concept demonstriert die Integration von LFM2-VL (Vision-Language Model) für Gesichts- und Objekterkennung im LEAP-PSW System. Die Implementierung fokussiert sich auf Edge-optimierte Deployment-Strategien für mobile Geräte und IoT-Hardware.

## 🚀 Features

### Core Funktionalitäten
- **Gesichtserkennung** mit Landmark-Detection und Emotionsanalyse
- **Objekterkennung** mit Multi-Class-Classification
- **Edge-Optimierung** für mobile und eingebettete Systeme
- **Real-time Processing** mit Video-Stream-Support
- **Multi-Platform Deployment** (iOS, Android, Edge-Geräte)

### Technische Highlights
- **LFM2-VL Integration** mit LEAP-Framework
- **Modell-Optimierung** (Quantization, Pruning, Operator Fusion)
- **Hardware-Beschleunigung** (Neural Engine, GPU, NNAPI, TPU)
- **Adaptive Performance** basierend auf Geräteleistung
- **Privacy-by-Design** mit On-Device Processing

## 📁 Projektstruktur

```
src/poc/vision-recognition/
├── core/                          # Kern-Module
│   ├── vision_pipeline.py        # Haupt-Processing-Pipeline
│   └── model_optimizer.py        # Edge-Optimierung
├── mobile/                        # Mobile Integrationen
│   ├── ios/
│   │   └── LFM2VLProcessor.swift  # iOS CoreML Integration
│   └── android/
│       └── LFM2VLProcessor.kt     # Android TFLite Integration
├── examples/
│   └── demo.py                    # Demo-Anwendung
├── config/
│   └── requirements.txt           # Python Dependencies
└── README.md
```

## 🛠️ Installation & Setup

### Voraussetzungen

```bash
# Python 3.8+ erforderlich
python --version

# Virtual Environment erstellen
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate     # Windows
```

### Dependencies installieren

```bash
# Core Python Dependencies
pip install -r config/requirements.txt

# Zusätzliche Downloads (je nach Platform)
# iOS: Xcode mit CoreML Tools
# Android: Android Studio mit TensorFlow Lite
```

### LFM2-VL Modell Setup

```bash
# Modell herunterladen (Platzhalter URLs)
wget https://models.leap.ai/lfm2-vl/base.pth -P models/
wget https://models.leap.ai/lfm2-vl/mobile.tflite -P models/
wget https://models.leap.ai/lfm2-vl/mobile.mlmodel -P models/
```

## 🔧 Verwendung

### 1. Basic Image Processing

```python
from core.vision_pipeline import VisionPipeline, ProcessingMode

# Pipeline initialisieren
pipeline = VisionPipeline(
    model_path="models/lfm2_vl_mobile.tflite",
    processing_mode=ProcessingMode.REALTIME
)

# Bild verarbeiten
import cv2
image = cv2.imread("test_image.jpg")
results = pipeline.process_image(image)

print(f"Faces: {len(results['faces'])}")
print(f"Objects: {len(results['objects'])}")
```

### 2. Model Optimization

```python
from core.model_optimizer import ModelOptimizer, OptimizationTarget

# Optimizer für iOS initialisieren
optimizer = ModelOptimizer(target_device="ios")

# Model optimieren
optimized_model, result = optimizer.optimize(
    original_model,
    target=OptimizationTarget.MOBILE_IOS
)

print(f"Compression: {result.compression_ratio:.2f}x")
print(f"Speedup: {result.estimated_speedup:.2f}x")
```

### 3. Demo Script ausführen

```bash
# Einfache Bild-Erkennung
python examples/demo.py --demo-type image --image-path test.jpg

# Batch-Processing
python examples/demo.py --demo-type batch --verbose

# Modell-Optimierung testen
python examples/demo.py --demo-type optimize --target-device mobile

# Performance Benchmark
python examples/demo.py --demo-type benchmark
```

### 4. Mobile Integration

#### iOS (Swift)
```swift
import UIKit

class ViewController: UIViewController {
    let processor = LFM2VLProcessor()

    override func viewDidLoad() {
        super.viewDidLoad()

        // Model laden
        let modelURL = Bundle.main.url(forResource: "LFM2VL_Mobile", withExtension: "mlmodelc")!
        try! processor.loadModel(from: modelURL)

        // Bild verarbeiten
        let image = UIImage(named: "test")!
        processor.process(image: image) { detections in
            print("Found \(detections.count) detections")
        }
    }
}
```

#### Android (Kotlin)
```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var processor: LFM2VLProcessor

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Processor initialisieren
        processor = LFM2VLProcessor(this)
        processor.initialize()

        // Bild verarbeiten
        val bitmap = BitmapFactory.decodeResource(resources, R.drawable.test)
        processor.processImage(bitmap) { detections ->
            Log.d("Vision", "Found ${detections.size} detections")
        }
    }
}
```

## 📊 Performance Benchmarks

### Desktop (Mock Results)
| Modus     | FPS | Avg Latency | Memory Usage |
|-----------|-----|-------------|--------------|
| Realtime  | 25  | 40ms        | 150MB        |
| Batch     | 15  | 67ms        | 200MB        |
| Edge      | 35  | 29ms        | 80MB         |

### Mobile Devices (Geschätzt)
| Device          | FPS | Model Size | Inference |
|-----------------|-----|------------|-----------|
| iPhone 13 Pro   | 30  | 45MB       | 33ms      |
| iPhone 12       | 22  | 45MB       | 45ms      |
| Samsung S22     | 18  | 60MB       | 55ms      |
| Pixel 6         | 20  | 60MB       | 50ms      |

## 🎯 Anwendungsfälle

### 1. Security & Access Control
```python
# Gesichtsbasierte Authentifizierung
pipeline = VisionPipeline(processing_mode=ProcessingMode.REALTIME)
pipeline.optimize_for_device("mobile")

# Real-time Erkennung
results = pipeline.process_video_frame(frame)
faces = results['faces']

for face in faces:
    if face.confidence > 0.85:
        embedding = face.embedding
        match = facial_recognition_db.match(embedding)
        if match:
            grant_access(match.user_id)
```

### 2. Retail & Shopping
```python
# Produkt-Erkennung
config = {'enable_object_tracking': True, 'max_detections': 20}
pipeline = VisionPipeline(config=config)

results = pipeline.process_image(shopping_image)
products = [obj for obj in results['objects'] if obj.category == 'product']

# Visual Search
similar_products = product_search_engine.find_similar(products[0].embedding)
```

### 3. Healthcare & Monitoring
```python
# Emotion & Stress Detection
pipeline = VisionPipeline(processing_mode=ProcessingMode.EDGE)

results = pipeline.process_video_frame(patient_frame)
faces = results['faces']

for face in faces:
    emotion = face.emotion
    stress_level = analyze_stress(face.landmarks, face.embedding)

    if stress_level > threshold:
        alert_medical_staff(patient_id, stress_level)
```

## 🔧 Konfiguration

### Environment Variables
```bash
# Model Paths
export LFM2VL_MODEL_PATH="/path/to/models/"
export LFM2VL_CACHE_DIR="/path/to/cache/"

# Performance Tuning
export LFM2VL_MAX_THREADS=4
export LFM2VL_USE_GPU=true
export LFM2VL_BATCH_SIZE=1

# Logging
export LFM2VL_LOG_LEVEL=INFO
export LFM2VL_LOG_FILE="/var/log/leap-psw/vision.log"
```

### Konfigurationsdatei (config.yaml)
```yaml
vision:
  model:
    path: "./models/lfm2_vl_mobile.tflite"
    optimization_level: "moderate"
    quantization: true

  processing:
    mode: "realtime"
    max_detections: 10
    confidence_threshold: 0.5
    nms_threshold: 0.4

  performance:
    use_gpu: true
    use_neural_engine: true  # iOS only
    max_threads: 4

  features:
    face_detection: true
    object_detection: true
    emotion_analysis: true
    landmark_detection: true
```

## 🧪 Testing

### Unit Tests ausführen
```bash
# Alle Tests
pytest tests/ -v

# Nur Vision Pipeline Tests
pytest tests/test_vision_pipeline.py -v

# Mit Coverage
pytest tests/ --cov=core --cov-report=html
```

### Performance Tests
```bash
# Benchmark verschiedene Modi
python examples/demo.py --demo-type benchmark

# Memory Profiling
python -m memory_profiler examples/demo.py

# GPU Utilization (NVIDIA)
nvidia-smi -l 1  # Monitor während Processing
```

## 📈 Monitoring & Logging

### Metriken sammeln
```python
from core.vision_pipeline import VisionPipeline

pipeline = VisionPipeline()

# Processing
results = pipeline.process_image(image)

# Metriken abrufen
stats = pipeline.get_performance_stats()
print(f"Avg inference time: {stats['average_inference_time']:.3f}s")
print(f"Total processed: {stats['total_processed']}")
```

### Logging konfigurieren
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vision_processing.log'),
        logging.StreamHandler()
    ]
)
```

## 🚀 Deployment

### Docker Container
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "examples/demo.py", "--demo-type", "image"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vision-recognition
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: vision-api
        image: leap-psw/vision:latest
        resources:
          limits:
            memory: "512Mi"
            cpu: "1000m"
          requests:
            memory: "256Mi"
            cpu: "500m"
```

## 🔒 Security & Privacy

### Datenverarbeitung
- **On-Device Processing**: Keine Daten verlassen das Gerät
- **Encryption**: Alle gespeicherten Modelle sind verschlüsselt
- **Anonymization**: Gesichtsembeddings werden anonymisiert gespeichert

### Compliance
- **DSGVO-konform**: Keine Speicherung biometrischer Rohdaten
- **SOC 2 Type II**: Audit-Trails für alle Verarbeitungsschritte
- **ISO 27001**: Sicherheitsstandards für Modell-Deployment

## 🤝 Beitragen

### Development Setup
```bash
# Repository klonen
git clone https://github.com/estkae/leap-PSW.git
cd leap-PSW/src/poc/vision-recognition

# Pre-commit hooks installieren
pip install pre-commit
pre-commit install

# Tests vor Commit ausführen
pytest tests/
black core/ examples/
flake8 core/ examples/
```

### Code Style
- **Black** für Python Formatting
- **Type Hints** für alle öffentlichen Funktionen
- **Docstrings** im Google Style
- **Unit Tests** mit >90% Coverage

## 📚 Dokumentation

- [Vision Pipeline API](docs/api/vision_pipeline.md)
- [Model Optimizer Guide](docs/guides/optimization.md)
- [Mobile Integration](docs/mobile/README.md)
- [Performance Tuning](docs/performance/README.md)

## 🐛 Troubleshooting

### Häufige Probleme

#### Model Loading Errors
```bash
# Problem: Model nicht gefunden
# Lösung: Pfad überprüfen
export LFM2VL_MODEL_PATH="/correct/path/to/models/"
```

#### Memory Issues
```python
# Problem: Out of Memory bei großen Bildern
# Lösung: Bilder vorher verkleinern
image = cv2.resize(image, (640, 640))
```

#### Slow Inference
```python
# Problem: Langsame Inferenz
# Lösung: Optimierungen aktivieren
config = {
    'use_gpu': True,
    'quantization': True,
    'max_detections': 5  # Weniger Detektionen
}
```

### Debug Mode
```python
# Detailliertes Logging aktivieren
import logging
logging.basicConfig(level=logging.DEBUG)

# Pipeline mit Debug-Modus
pipeline = VisionPipeline(debug=True)
```

## 📝 Changelog

### v0.1.0 (2024-01-15)
- Initial POC implementation
- Basic face and object detection
- iOS and Android integration templates
- Model optimization framework
- Demo scripts and examples

## 📄 Lizenz

Proprietär - AALS Software AG. Alle Rechte vorbehalten.

## 📧 Support

Bei Fragen oder Problemen wenden Sie sich an:
- **Email**: leap-support@aals-software.com
- **Slack**: #leap-psw-vision
- **Issues**: GitHub Issues für Bug Reports

---

© 2024 AALS Software AG. Confidential and Proprietary.