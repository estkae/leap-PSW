# Fine-Tuning Guide: Modell für spezielle Aufgaben anpassen

## Was ist Fine-Tuning?

Fine-Tuning bedeutet, ein bereits trainiertes Modell für eine neue, spezielle Aufgabe anzupassen. Das ist viel schneller und effizienter als ein neues Training von Grund auf.

## Wann Fine-Tuning verwenden?

- **Neue Objektklassen**: Sie möchten andere Objekte erkennen
- **Spezielle Domäne**: Medizinische Bilder, Satellitenbilder, etc.
- **Wenig Trainingsdaten**: Sie haben nur wenige Beispiele
- **Zeitersparnis**: Schneller als komplettes Neutraining

## Schritt-für-Schritt Anleitung

### 1. Basis-Modell vorbereiten

Zuerst brauchen Sie ein vortrainiertes Modell:

```bash
# Zuerst ein Basis-Modell trainieren
python training/train_model.py

# Das erzeugt: checkpoints/best_model.pth
```

### 2. Neue Daten vorbereiten

Organisieren Sie Ihre neuen Klassen:

```
data_finetune/
├── train/
│   ├── hund/
│   │   ├── hund001.jpg
│   │   └── ...
│   ├── katze/
│   │   ├── katze001.jpg
│   │   └── ...
│   └── vogel/
│       ├── vogel001.jpg
│       └── ...
└── val/
    ├── hund/
    ├── katze/
    └── vogel/
```

### 3. Fine-Tuning konfigurieren

In `fine_tune.py` anpassen:

```python
config = FineTuningConfig(
    # Basis-Modell laden
    pretrained_model_path="checkpoints/best_model.pth",

    # Neue Aufgabe
    new_num_classes=3,       # hund, katze, vogel

    # Fine-Tuning Parameter
    learning_rate=0.0001,    # Kleinere Lernrate!
    num_epochs=20,           # Weniger Epochen
    batch_size=16,           # Kleinere Batches

    # Strategien
    freeze_backbone=True,          # Basis-Features einfrieren
    progressive_unfreezing=True,   # Schrittweise entsperren

    # Daten
    data_dir="data_finetune"
)
```

### 4. Fine-Tuning starten

```bash
python training/fine_tune.py
```

## Fine-Tuning Strategien

### A) Frozen Backbone (Empfohlen für wenig Daten)

```python
freeze_backbone=True
```

- **Was passiert**: Nur der Klassifikator wird trainiert
- **Vorteil**: Schnell, braucht wenig Daten
- **Nachteil**: Begrenzte Anpassung

### B) Progressive Unfreezing

```python
progressive_unfreezing=True
```

- **Phase 1**: Nur Klassifikator (5 Epochen)
- **Phase 2**: Letzte Schicht entsperren (10 Epochen)
- **Phase 3**: Alle Schichten entsperren (5 Epochen)

### C) Full Fine-Tuning

```python
freeze_backbone=False
```

- **Was passiert**: Alle Schichten werden angepasst
- **Vorteil**: Beste Anpassung
- **Nachteil**: Braucht viele Daten und Zeit

## Praktische Beispiele

### Beispiel 1: Tierarten erkennen

```python
# Von allgemeinen Objekten zu spezifischen Tieren
config = FineTuningConfig(
    pretrained_model_path="checkpoints/objects_model.pth",
    new_num_classes=5,  # hund, katze, vogel, fisch, hamster
    learning_rate=0.0001,
    freeze_backbone=True
)
```

### Beispiel 2: Gesichter einer Familie

```python
# Von allgemeiner Gesichtserkennung zu Familie
config = FineTuningConfig(
    pretrained_model_path="checkpoints/face_model.pth",
    new_num_classes=4,  # mama, papa, kind1, kind2
    learning_rate=0.00005,  # Sehr kleine Lernrate
    progressive_unfreezing=True
)
```

### Beispiel 3: Medizinische Bilder

```python
# Von normalen Fotos zu Röntgenbildern
config = FineTuningConfig(
    pretrained_model_path="checkpoints/general_model.pth",
    new_num_classes=3,  # normal, pneumonia, covid
    learning_rate=0.0001,
    num_epochs=30,
    data_augmentation=False  # Bei medizinischen Bildern vorsichtig
)
```

## Tipps für erfolgreiches Fine-Tuning

### 1. Lernrate wählen
- **Zu hoch**: Zerstört vortrainierte Features
- **Zu niedrig**: Lernt zu langsam
- **Empfehlung**: 10x-100x kleiner als beim ersten Training

### 2. Datenqualität
- **Mindestens 50 Bilder pro Klasse**
- **Ähnlicher Stil wie Originaldaten**
- **Gute Bildqualität**

### 3. Epochen
- **Weniger ist oft mehr**
- **Early Stopping verwenden**
- **Validation Loss beobachten**

### 4. Batch Size
- **Kleinere Batches bei Fine-Tuning**
- **Bessere Gradientenqualität**
- **Weniger Speicherverbrauch**

## Häufige Probleme

### Problem: Overfitting
```python
# Lösung: Mehr Regularisierung
config.weight_decay = 0.01
config.dropout = 0.5
# Weniger Epochen
config.num_epochs = 15
```

### Problem: Katastrophic Forgetting
```python
# Lösung: Kleinere Lernrate
config.learning_rate = 0.00005
config.freeze_backbone = True
```

### Problem: Schlechte Performance
```python
# Lösung: Progressive Unfreezing
config.progressive_unfreezing = True
# Mehr Trainingsdaten sammeln
# Datenaugmentation aktivieren
```

## Fine-Tuning für Ihr Projekt

Das trainierte Modell können Sie dann in der Vision-Pipeline verwenden:

```python
from core.vision_pipeline import VisionPipeline

# Fine-getuntes Modell laden
pipeline = VisionPipeline(
    model_path="checkpoints/finetuned_model.pth",
    processing_mode=ProcessingMode.REALTIME
)

# Für Echtzeit-Erkennung verwenden
result = pipeline.process_frame(camera_frame)
```

So können Sie das LFM2-VL Modell für Ihre spezifischen Erkennungsaufgaben anpassen!