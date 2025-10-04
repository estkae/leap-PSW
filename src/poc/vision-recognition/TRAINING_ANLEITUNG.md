# Anleitung: Modell-Training für Bilderkennung

## So bringen Sie dem Modell bei, Bilder zu erkennen

### 1. Daten vorbereiten

Organisieren Sie Ihre Bilder in dieser Struktur:

```
data/
├── train/          (70% Ihrer Bilder)
│   ├── gesicht/
│   │   ├── gesicht001.jpg
│   │   ├── gesicht002.jpg
│   │   └── ...
│   ├── person/
│   │   ├── person001.jpg
│   │   └── ...
│   └── auto/
│       ├── auto001.jpg
│       └── ...
├── val/            (20% Ihrer Bilder)
│   ├── gesicht/
│   ├── person/
│   └── auto/
└── test/           (10% Ihrer Bilder)
    ├── gesicht/
    ├── person/
    └── auto/
```

### 2. Bildanforderungen

- **Format**: JPG, PNG, BMP
- **Größe**: Mindestens 224x224 Pixel
- **Qualität**: Hochauflösend und scharf
- **Anzahl**: Mindestens 100 Bilder pro Klasse
- **Vielfalt**: Verschiedene Winkel, Beleuchtung, Hintergründe

### 3. Training starten

#### A) Basis-Training
```bash
python training/train_model.py
```

#### B) Mit angepasster Konfiguration
Ändern Sie in `train_model.py`:
```python
config = TrainingConfig(
    num_classes=3,           # Anzahl Ihrer Klassen (gesicht, person, auto)
    batch_size=16,           # Reduzieren wenn GPU-Speicher knapp
    learning_rate=0.001,     # Lernrate
    num_epochs=50,           # Anzahl Durchgänge
    device="cuda"            # GPU verwenden wenn verfügbar
)
```

### 4. Training überwachen

Das Training zeigt Ihnen:
- **Loss**: Wird kleiner = Modell lernt
- **Accuracy**: Wird größer = Modell wird besser
- **Validation**: Prüft ob Modell auch neue Bilder erkennt

### 5. Modell testen

Nach dem Training:
```bash
python training/evaluate_model.py
```

### 6. Einzelbild testen

```python
from evaluate_model import ModelEvaluator
evaluator = ModelEvaluator("checkpoints/best_model.pth", config)
result = evaluator.evaluate_single_image("mein_testbild.jpg")
print(f"Erkannt: {result['predicted_class']}")
print(f"Sicherheit: {result['confidence']:.2f}")
```

### 7. Fine-Tuning für spezielle Aufgaben

Wenn Sie bereits ein trainiertes Modell haben und es für neue Objekte anpassen möchten:

```bash
python training/fine_tune.py
```

## Praktische Tipps

### Training optimieren
- **Mehr Daten**: Bessere Ergebnisse
- **Ausgewogene Klassen**: Gleich viele Bilder pro Typ
- **Datenaugmentation**: Automatische Variationen
- **Early Stopping**: Stoppt wenn keine Verbesserung

### Häufige Probleme
- **GPU-Speicher voll**: Batch-Size reduzieren
- **Training zu langsam**: Weniger Epochen oder CPU verwenden
- **Schlechte Ergebnisse**: Mehr/bessere Trainingsdaten

## Beispiel für echte Anwendung

1. Sammeln Sie 300 Gesichtsbilder → `data/train/gesicht/`
2. Sammeln Sie 300 Personenbilder → `data/train/person/`
3. Sammeln Sie 300 Auto-Bilder → `data/train/auto/`
4. Teilen Sie 20% für Validierung ab
5. Starten Sie Training: `python training/train_model.py`
6. Nach 30-50 Epochen haben Sie ein funktionsfähiges Modell!

Das Modell kann dann in der Vision-Pipeline verwendet werden für Echtzeit-Erkennung.