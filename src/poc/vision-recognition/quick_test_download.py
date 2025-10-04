#!/usr/bin/env python3
"""
Quick Test Download - Kleine Val-Sets für schnellen Test
=========================================================

Lädt nur die Validation-Sets herunter für schnelle Tests:
- LFW Faces (~170 MB, ~13K Bilder)
- COCO Val Person & Car (~1 GB, ~5K Bilder)

Komplett-Download dauert nur ~10-15 Minuten statt mehrere Stunden!

Usage:
    python quick_test_download.py
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> int:
    """Führe Command aus mit Fehlerbehandlung"""
    print("\n" + "=" * 60)
    print(f"🚀 {description}")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n[ERROR] Fehler bei: {description}")
        return result.returncode

    print(f"\n[OK] Erfolgreich: {description}")
    return 0


def main():
    """Quick Test Download Pipeline"""

    print("""
================================================================
|                                                              |
|     LFM2-VL QUICK TEST DOWNLOAD                             |
|                                                              |
|     Lädt kleine Val-Sets für schnelle Tests                 |
|                                                              |
================================================================

📊 Was wird heruntergeladen:
   - LFW Faces Dataset (~170 MB, ~13K Gesichter)
   - COCO Val-Set (~1 GB, Person & Car)

[TIME]  Geschätzte Dauer: 10-15 Minuten
💾 Speicherbedarf: ~2-3 GB

Starte in 3 Sekunden...
""")

    import time
    time.sleep(3)

    # Check if scripts exist
    face_script = Path("download_face_dataset.py")
    coco_script = Path("download_coco_dataset.py")
    organize_script = Path("organize_training_data.py")

    if not all([face_script.exists(), coco_script.exists(), organize_script.exists()]):
        print("[ERROR] Scripts nicht gefunden. Bitte im vision-recognition/ Ordner ausführen.")
        return 1

    steps = [
        # Step 1: Download Face Dataset (LFW)
        {
            'cmd': [
                sys.executable,
                'download_face_dataset.py',
                '--dataset', 'lfw',
                '--download-dir', './face_download',
                '--output-dir', './dataset_download_temp/face_organized'
            ],
            'desc': 'Step 1/5: Lade LFW Face Dataset'
        },

        # Step 2: Download COCO Val-Set
        {
            'cmd': [
                sys.executable,
                'download_coco_dataset.py',
                '--splits', 'val',
                '--categories', 'person', 'car',
                '--download-dir', './coco_download',
                '--output-dir', './dataset_download_temp/coco_organized'
            ],
            'desc': 'Step 2/5: Lade COCO Validation Set'
        },

        # Step 3: Organisiere Face-Daten
        {
            'cmd': [
                sys.executable,
                'organize_training_data.py',
                '--source', './dataset_download_temp/face_organized',
                '--target', '.',
                '--categories', 'face',
                '--train-ratio', '0.6',
                '--val-ratio', '0.2',
                '--test-ratio', '0.2'
            ],
            'desc': 'Step 3/5: Organisiere Face-Daten'
        },

        # Step 4: Organisiere COCO-Daten
        {
            'cmd': [
                sys.executable,
                'organize_training_data.py',
                '--source', './dataset_download_temp/coco_organized',
                '--target', '.',
                '--categories', 'person', 'car',
                '--train-ratio', '0.0',  # Val-Set nicht nochmal splitten
                '--val-ratio', '0.7',
                '--test-ratio', '0.3'
            ],
            'desc': 'Step 4/5: Organisiere COCO-Daten'
        },
    ]

    # Execute all steps
    for i, step in enumerate(steps, 1):
        result = run_command(step['cmd'], step['desc'])
        if result != 0:
            print(f"\n[ERROR] Pipeline abgebrochen bei Step {i}")
            return result

    # Final Summary
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)

    # Count images in each directory
    import glob

    categories = [
        ('datatraingesicht', 'Training Gesichter'),
        ('datavalgesicht', 'Validation Gesichter'),
        ('datatestgesicht', 'Test Gesichter'),
        ('datavalperson', 'Validation Personen'),
        ('datatestperson', 'Test Personen'),
        ('datavalauto', 'Validation Autos'),
        ('datatestauto', 'Test Autos'),
    ]

    print("\nBilder pro Kategorie:")
    total = 0
    for dirname, label in categories:
        if Path(dirname).exists():
            count = len(list(Path(dirname).glob('*.jpg')))
            print(f"  {label:30} {count:>6,} Bilder")
            total += count
        else:
            print(f"  {label:30}      - (nicht vorhanden)")

    print(f"\n{'=' * 60}")
    print(f"GESAMT: {total:,} Bilder für Tests verfügbar")
    print(f"{'=' * 60}")

    print("""
[OK] Quick Test Download ERFOLGREICH!

🎯 Nächste Schritte:

1. Test-Training durchführen:
   cd training/
   python demo_training.py

2. Einzelbild-Test:
   python quick_demo.py

3. Volles Training (später):
   python training/train_model.py

💡 Für vollständiges Training-Set später:
   python download_coco_dataset.py --splits train val
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())