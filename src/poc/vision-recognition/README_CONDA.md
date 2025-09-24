# LFM2-VL Vision Recognition POC - Conda Setup Guide

## 🚀 Schnellstart mit Conda

Diese Anleitung beschreibt die Verwendung des LEAP-PSW Vision Recognition POC mit Conda-Umgebung.

## 📋 Voraussetzungen

- **Anaconda** oder **Miniconda** installiert
  - Download: [Anaconda](https://www.anaconda.com/) oder [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- **Windows 10/11** (für andere OS siehe Unix-Version)
- **Git** für Repository-Verwaltung

## 🛠️ Installation

### Option 1: Automatisches Setup (Empfohlen)

```batch
# 1. In das POC-Verzeichnis wechseln
cd src\poc\vision-recognition

# 2. Setup-Script ausführen
setup_conda.bat
```

Das Script:
- ✅ Erstellt automatisch die `leap-PSW` Umgebung
- ✅ Installiert alle Dependencies
- ✅ Erstellt Test-Images und Mock-Modelle
- ✅ Führt Funktionstests durch

### Option 2: Manuelle Installation

```batch
# 1. Environment aus YAML erstellen
conda env create -f environment.yml

# 2. Environment aktivieren
conda activate leap-PSW

# 3. Test-Setup ausführen
python quick_setup.py
python create_test_images.py
```

### Option 3: Minimale Installation

```batch
# 1. Basis-Environment erstellen
conda create -n leap-PSW python=3.10

# 2. Aktivieren
conda activate leap-PSW

# 3. Core-Packages installieren
conda install numpy opencv scikit-image pillow
conda install pytorch torchvision cpuonly -c pytorch

# 4. Setup ausführen
python quick_setup.py
```

## 🎮 Demo ausführen

### Mit Conda-Launcher (Empfohlen)

```batch
# Startet automatisch mit leap-PSW Environment
run_demo_conda.bat
```

Verfügbare Optionen:
1. **Quick Demo** - Basis-Funktionalitätstest
2. **Image Processing** - Bildverarbeitung Demo
3. **Batch Processing** - Stapelverarbeitung
4. **Model Optimization** - Modell-Optimierung
5. **Performance Benchmark** - Leistungstests
6. **Interactive Menu** - Interaktives Menü
7. **Install/Update** - Dependencies verwalten

### Manuell in Conda

```batch
# 1. Environment aktivieren
conda activate leap-PSW

# 2. Simple Test
python simple_test.py

# 3. Vollständige Demo
python examples/demo.py --demo-type image

# 4. Benchmark
python examples/demo.py --demo-type benchmark
```

## 📦 Environment-Verwaltung

### Environment-Info anzeigen

```batch
# Alle Environments auflisten
conda env list

# Packages in leap-PSW anzeigen
conda activate leap-PSW
conda list
```

### Environment aktualisieren

```batch
# Aus environment.yml aktualisieren
conda env update -f environment.yml

# Einzelne Packages
conda activate leap-PSW
conda update numpy opencv pytorch
```

### Environment exportieren

```batch
# Aktuelle Konfiguration speichern
conda activate leap-PSW
conda env export > environment_backup.yml
```

### Environment löschen und neu erstellen

```batch
# Löschen
conda deactivate
conda env remove -n leap-PSW

# Neu erstellen
conda env create -f environment.yml
```

## 🔧 Troubleshooting

### Problem: "Conda nicht gefunden"

```batch
# Conda zum PATH hinzufügen
set PATH=%PATH%;C:\Users\%USERNAME%\Anaconda3\Scripts
```

### Problem: "Environment existiert bereits"

```batch
# Option 1: Aktualisieren
conda env update -f environment.yml

# Option 2: Löschen und neu erstellen
conda env remove -n leap-PSW
conda env create -f environment.yml
```

### Problem: "Package-Konflikte"

```batch
# Konflikte auflösen
conda activate leap-PSW
conda update --all
pip install --upgrade pip
```

### Problem: "CUDA/GPU nicht verfügbar"

```batch
# CPU-only Version verwenden
conda install pytorch torchvision cpuonly -c pytorch

# Oder GPU-Version (wenn CUDA installiert)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

## 📊 Package-Versionen

### Core Requirements

| Package | Version | Zweck |
|---------|---------|-------|
| Python | 3.10 | Basis |
| NumPy | ≥1.21.0 | Array-Operationen |
| OpenCV | ≥4.5.0 | Bildverarbeitung |
| PyTorch | ≥1.12.0 | Deep Learning |
| Pillow | ≥8.3.0 | Bildbearbeitung |

### Optional (für vollständige Features)

| Package | Version | Zweck |
|---------|---------|-------|
| TensorFlow | ≥2.9.0 | TFLite Export |
| CoreMLTools | ≥6.0 | iOS Deployment |
| ONNX | ≥1.12.0 | Model Export |
| FastAPI | ≥0.78.0 | REST API |

## 🚢 Deployment mit Conda

### Environment für Production

```yaml
# environment_prod.yml
name: leap-PSW-prod
dependencies:
  - python=3.10
  - numpy=1.21.*
  - opencv=4.5.*
  - pytorch=1.12.*
  # Fixierte Versionen für Stabilität
```

### Docker mit Conda

```dockerfile
FROM continuumio/miniconda3

COPY environment.yml .
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "leap-PSW", "/bin/bash", "-c"]

WORKDIR /app
COPY . .

CMD ["conda", "run", "-n", "leap-PSW", "python", "examples/demo.py"]
```

## 🔍 Testen der Installation

### Automatischer Test

```batch
# Führt alle Tests durch
run_demo_conda.bat
# Wähle Option 5 (Run unit tests)
```

### Manueller Test

```python
# test_installation.py
import sys

def test_imports():
    """Test ob alle wichtigen Packages verfügbar sind"""
    packages = {
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'torch': 'PyTorch',
        'PIL': 'Pillow'
    }

    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✅ {name} - OK")
        except ImportError:
            print(f"❌ {name} - FEHLT")

if __name__ == '__main__':
    test_imports()
```

## 📝 Nützliche Conda-Befehle

```batch
# Environment-Management
conda create -n NAME python=3.10     # Erstellen
conda activate NAME                   # Aktivieren
conda deactivate                      # Deaktivieren
conda env remove -n NAME              # Löschen

# Package-Management
conda install PACKAGE                 # Installieren
conda update PACKAGE                  # Aktualisieren
conda remove PACKAGE                  # Entfernen
conda list                           # Alle Packages

# Channel-Management
conda config --add channels conda-forge
conda config --show channels
```

## 🤝 Support

Bei Problemen:

1. **Logs prüfen**: `logs/` Verzeichnis
2. **Environment neu erstellen**: `setup_conda.bat`
3. **GitHub Issues**: [Repository Issues](https://github.com/estkae/leap-PSW/issues)

## 📄 Lizenz

Proprietär - AALS Software AG. Alle Rechte vorbehalten.

---

© 2024 AALS Software AG