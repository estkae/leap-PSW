# Dataset Download & Organisation Guide - LFM2-VL

## 🚀 QUICK START - Test mit kleinem Val-Set (EMPFOHLEN)

```bash
# Schneller Test-Download (~2-3 GB, 10-15 Min)
python quick_test_download.py

# Fertig! Training starten
cd training/
python demo_training.py
```

**Was wird heruntergeladen:**
- ✅ LFW Faces (~170 MB, ~13K Gesichter)
- ✅ COCO Val-Set (~1 GB, Person & Car)
- ✅ Automatische Organisation in train/val/test
- ⏱️ Nur 10-15 Minuten statt mehrere Stunden!

---

## 📦 Vollständiger Download (für Production Training)

### Option 1: Alle Datasets auf einmal

```bash
# 1. Gesichter herunterladen (LFW - klein & schnell)
python download_face_dataset.py --dataset lfw

# 2. COCO Dataset (Person & Auto)
python download_coco_dataset.py --splits train val --categories person car

# 3. Alles organisieren
python organize_training_data.py --source face_download/organized --categories face
python organize_training_data.py --source coco_download/organized --categories person car
```

---

## 📁 Einzelne Datasets herunterladen

### 🎭 Gesichter-Dataset (Face Detection)

#### LFW Dataset (Empfohlen für Start)
```bash
python download_face_dataset.py --dataset lfw
```

**Specs:**
- 📦 Größe: ~170 MB
- 📊 Bilder: ~13,000 Gesichter
- ⏱️ Download: ~2-5 Minuten
- ✅ Keine zusätzlichen Dependencies

#### WIDER FACE Dataset (Für Production)
```bash
# Erst gdown installieren
pip install gdown

# Download
python download_face_dataset.py --dataset widerface --splits train val
```

**Specs:**
- 📦 Größe: ~3.5 GB
- 📊 Bilder: 32,203 Bilder, 393,703 Gesichter
- ⏱️ Download: ~30-60 Minuten

### 👤 Person & 🚗 Auto (COCO Dataset)

```bash
python download_coco_dataset.py --splits train val --categories person car
```

**Specs:**
- 📦 Train: ~18 GB
- 📦 Val: ~1 GB
- 📊 Person: ~65K train, ~3K val
- 📊 Auto: ~12K train, ~600 val
- ⏱️ Download: ~2-4 Stunden

---

## 📂 Daten organisieren

```bash
# Gesichter organisieren
python organize_training_data.py \
  --source face_download/organized \
  --target . \
  --categories face

# Person & Auto organisieren
python organize_training_data.py \
  --source coco_download/organized \
  --target . \
  --categories person car
```

**Erstellt automatisch:**
```
datatraingesicht/   # Training Gesichter
datavalgesicht/     # Validation Gesichter
datatestgesicht/    # Test Gesichter
datatrainperson/    # Training Personen
datavalperson/      # Validation Personen
datatestperson/     # Test Personen
datatrainauto/      # Training Autos
datavalauto/        # Validation Autos
datatestauto/       # Test Autos
```

---

## 📊 Dataset-Übersicht

| Dataset | Kategorie | Bilder | Größe | Download-Zeit | Empfehlung |
|---------|-----------|--------|-------|---------------|------------|
| **LFW** | Gesichter | 13K | 170 MB | 2-5 Min | ✅ Für Start |
| **WIDER FACE** | Gesichter | 393K | 3.5 GB | 30-60 Min | Production |
| **COCO Val** | Person/Auto | 5K | 1 GB | 10-20 Min | ✅ Für Tests |
| **COCO Train** | Person/Auto | 118K | 18 GB | 2-4 Std | Production |

---

## 💾 Speicherplatz-Anforderungen

### Quick Test (empfohlen für Start)
```
LFW Face:           170 MB
COCO Val:         1,000 MB
Organisiert:      ~500 MB
---------------------------
GESAMT:          ~2-3 GB
```

### Production (volles Training)
```
COCO Train:      18,000 MB
COCO Val:         1,000 MB
WIDER FACE:       3,500 MB
Downloads:       ~5,000 MB (temp)
Organisiert:    ~10,000 MB
---------------------------
GESAMT:         ~40-50 GB
```

---

## 🔧 Requirements

```bash
# Basis-Requirements
pip install requests tqdm pillow

# Für WIDER FACE Dataset
pip install gdown

# Komplett (aus project requirements.txt)
pip install -r requirements.txt
```

---

## 🎯 Workflow-Empfehlungen

### 1️⃣ Anfänger / Quick Test
```bash
python quick_test_download.py
```

### 2️⃣ Entwicklung / Prototyping
```bash
python download_face_dataset.py --dataset lfw
python download_coco_dataset.py --splits val
python organize_training_data.py --source face_download/organized --categories face
python organize_training_data.py --source coco_download/organized --categories person car
```

### 3️⃣ Production / Full Training
```bash
pip install gdown
python download_face_dataset.py --dataset widerface --splits train val
python download_coco_dataset.py --splits train val --categories person car
python organize_training_data.py --source face_download/organized --categories face
python organize_training_data.py --source coco_download/organized --categories person car
```

---

## 🚨 Troubleshooting

### Problem: Download zu langsam
```bash
# Lösung: Quick Test Script verwenden
python quick_test_download.py
```

### Problem: WIDER FACE Download schlägt fehl
```bash
# Lösung: Nutze LFW stattdessen
python download_face_dataset.py --dataset lfw
```

### Problem: Nicht genug Speicher
```bash
# Lösung: Quick Test mit Val-Set
python quick_test_download.py  # Nur ~3 GB
```

---

## 🎓 Nächste Schritte

Nach erfolgreichem Download:

```bash
# 1. Quick Demo
python quick_demo.py

# 2. Training Demo
cd training/
python demo_training.py

# 3. Full Training
cd training/
python train_model.py
```

---

## 📖 Dataset-Quellen

- **COCO**: https://cocodataset.org/
- **LFW**: http://vis-www.cs.umass.edu/lfw/
- **WIDER FACE**: http://shuoyang1213.me/WIDERFACE/

---

© 2024 AALS Software AG - LEAP-PSW Project