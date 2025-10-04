# COCO Dataset Download & Organisation Guide

## 🎯 Schnellstart - Komplett-Pipeline

```bash
# 1. COCO Dataset herunterladen (Person & Auto)
python download_coco_dataset.py --splits train val --categories person car

# 2. In LFM2-VL Struktur organisieren
python organize_training_data.py --source coco_download/organized --target .

# 3. Fertig! Training starten
python training/train_model.py
```

## 📥 Schritt 1: COCO Dataset Download

### Basis-Download (Person & Auto)
```bash
python download_coco_dataset.py
```

**Was passiert:**
- ✅ Lädt COCO 2017 Train-Set (~18 GB)
- ✅ Lädt COCO 2017 Val-Set (~1 GB)
- ✅ Filtert Person-Bilder
- ✅ Filtert Auto-Bilder
- ✅ Organisiert in `coco_download/organized/`

### Erweiterte Optionen

#### Nur Training-Daten
```bash
python download_coco_dataset.py --splits train
```

#### Zusätzliche Kategorien
```bash
python download_coco_dataset.py --categories person car bicycle motorcycle
```

#### Custom Download-Verzeichnis
```bash
python download_coco_dataset.py --download-dir /path/to/download
```

#### Direktes Output-Verzeichnis
```bash
python download_coco_dataset.py --output-dir /path/to/organized
```

### Parameter-Übersicht

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `--splits` | `train val` | Welche Splits laden |
| `--categories` | `person car` | Welche Kategorien filtern |
| `--download-dir` | `./coco_download` | Wo Downloads speichern |
| `--output-dir` | `coco_download/organized` | Wo organisierte Daten speichern |

## 📁 Schritt 2: Training-Daten organisieren

### Basis-Organisation
```bash
python organize_training_data.py --source coco_download/organized
```

**Was passiert:**
- ✅ Erstellt `datatrainperson/`, `datatrainauto/`
- ✅ Erstellt `datavalperson/`, `datavalauto/`
- ✅ Erstellt `datatestperson/`, `datatestauto/`
- ✅ Split: 70% Train, 20% Val, 10% Test

### Custom Split-Verhältnisse
```bash
python organize_training_data.py \
  --source coco_download/organized \
  --train-ratio 0.8 \
  --val-ratio 0.15 \
  --test-ratio 0.05
```

### Nur bestimmte Kategorien
```bash
python organize_training_data.py \
  --source coco_download/organized \
  --categories person
```

### Reproduzierbare Splits
```bash
python organize_training_data.py \
  --source coco_download/organized \
  --seed 42
```

### Parameter-Übersicht

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `--source` | (erforderlich) | Source-Verzeichnis mit COCO Daten |
| `--target` | `.` | Ziel-Verzeichnis |
| `--categories` | `person car` | Welche Kategorien organisieren |
| `--train-ratio` | `0.7` | Anteil Training (70%) |
| `--val-ratio` | `0.2` | Anteil Validation (20%) |
| `--test-ratio` | `0.1` | Anteil Test (10%) |
| `--seed` | `42` | Random Seed |

## 📊 Erwartete Datenmengen

### COCO 2017 Dataset

| Split | Gesamt | Person | Auto (ca.) |
|-------|--------|--------|-----------|
| Train | 118K Bilder | ~65K | ~12K |
| Val | 5K Bilder | ~3K | ~600 |

### Nach Organisation (70/20/10 Split)

#### Person-Kategorie
```
datatrainperson/    ~65,000 Bilder
datavalperson/      ~600 Bilder
datatestperson/     ~300 Bilder
```

#### Auto-Kategorie
```
datatrainauto/      ~12,000 Bilder
datavalauto/        ~120 Bilder
datatestauto/       ~60 Bilder
```

## 🗂️ Resultierende Ordnerstruktur

```
vision-recognition/
├── coco_download/              # Download-Verzeichnis
│   ├── downloads/              # ZIP-Dateien
│   │   ├── train2017.zip      (~18 GB)
│   │   ├── val2017.zip        (~1 GB)
│   │   └── annotations_trainval2017.zip
│   ├── extracted/              # Extrahierte Daten
│   │   ├── train2017/
│   │   ├── val2017/
│   │   └── annotations/
│   └── organized/              # Gefilterte Daten
│       ├── train/
│       │   ├── person/
│       │   └── car/
│       └── val/
│           ├── person/
│           └── car/
│
├── datatrainperson/           # Training Personen
├── datatraingesicht/          # Training Gesichter (TODO)
├── datatrainauto/             # Training Autos
├── datavalperson/             # Validation Personen
├── datavalgesicht/            # Validation Gesichter (TODO)
├── datavalauto/               # Validation Autos
├── datatestperson/            # Test Personen
├── datatestgesicht/           # Test Gesichter (TODO)
└── datatestauto/              # Test Autos
```

## ⚙️ Requirements

### Python-Pakete installieren
```bash
pip install requests tqdm pillow
```

Oder nutze bestehende `requirements.txt`:
```bash
pip install -r requirements.txt
```

## 💾 Speicherplatz-Anforderungen

| Phase | Speicherbedarf | Beschreibung |
|-------|----------------|--------------|
| Download | ~20 GB | ZIP-Dateien |
| Extraction | ~20 GB | Extrahierte Bilder |
| Organization | ~5-10 GB | Gefilterte Kategorien |
| **GESAMT** | **~45-50 GB** | Inkl. temporäre Dateien |

### Speicherplatz reduzieren

```bash
# Nach erfolgreicher Organisation ZIP-Dateien löschen
rm -rf coco_download/downloads/*.zip

# Nach Organisation extrahierte Originale löschen
rm -rf coco_download/extracted/

# Nur organisierte Daten behalten (~5-10 GB)
```

## 🔄 Workflow-Beispiele

### Minimaler Download (nur Val-Set für Tests)
```bash
# Schneller Test mit kleinerem Datensatz
python download_coco_dataset.py --splits val --categories person

python organize_training_data.py \
  --source coco_download/organized \
  --categories person \
  --train-ratio 0.6 \
  --val-ratio 0.2 \
  --test-ratio 0.2
```

### Vollständiger Download (alle Kategorien)
```bash
# Maximaler Datensatz
python download_coco_dataset.py \
  --splits train val \
  --categories person car bicycle motorcycle bus truck

python organize_training_data.py \
  --source coco_download/organized \
  --categories person car
```

### Zwei-Phasen Download
```bash
# Phase 1: Val-Set für schnelle Tests
python download_coco_dataset.py --splits val

# Phase 2: Später Train-Set für volles Training
python download_coco_dataset.py --splits train
```

## 🚀 Next Steps

Nach erfolgreicher Daten-Organisation:

### 1. Daten überprüfen
```bash
# Anzahl Bilder pro Kategorie checken
ls datatrainperson/*.jpg | wc -l
ls datatrainauto/*.jpg | wc -l
```

### 2. Training starten
```bash
cd training/
python train_model.py
```

### 3. Quick Validation Test
```python
from training.evaluate_model import ModelEvaluator
from training.train_model import TrainingConfig

config = TrainingConfig()
# Test mit einem Validierungsbild
# ...
```

## 🔧 Troubleshooting

### Problem: Download zu langsam
```bash
# Lösung: Nutze Mirrors oder parallele Downloads
# Alternativ: Download manuell und extrahiere in coco_download/extracted/
```

### Problem: Nicht genug Speicherplatz
```bash
# Lösung 1: Nur Val-Set verwenden
python download_coco_dataset.py --splits val

# Lösung 2: Nach jeder Phase aufräumen
rm -rf coco_download/downloads/
```

### Problem: Script bricht ab
```bash
# Lösung: Re-run ist sicher (überspringt existierende Dateien)
python download_coco_dataset.py
# "✓ ... bereits vorhanden, überspringe Download"
```

### Problem: Falsche Ordnerstruktur
```bash
# Lösung: Reorganisiere mit anderen Parametern
python organize_training_data.py \
  --source coco_download/organized \
  --target . \
  --train-ratio 0.8
```

## 📚 Weitere Datasets (Optional)

### Gesichter-spezifisch

#### WIDER FACE
```bash
# Download von: http://shuoyang1213.me/WIDERFACE/
# Manuell in datatraingesicht/, datavalgesicht/, datatestgesicht/ organisieren
```

#### CelebA
```bash
# Download von: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# Registrierung erforderlich
```

### Script für manuelle Integration
```python
# organize_custom_dataset.py (TODO)
# Für Integration eigener Datasets
```

## 📖 Weitere Dokumentation

- [Training Guide](training/README.md)
- [Model Evaluation](training/evaluate_model.py)
- [Vision Pipeline](core/README.md)

---

© 2024 AALS Software AG - LEAP-PSW Project