#!/usr/bin/env python3
"""
WIDER FACE Dataset Downloader für LFM2-VL Face Recognition
===========================================================

Lädt WIDER FACE Dataset herunter - eines der besten öffentlichen Gesichts-Datasets:
- 32,203 Bilder
- 393,703 annotierte Gesichter
- Diverse Szenarien (Events, Crowd, Portraits)

Alternative: Falls WIDER FACE zu groß ist, wird auf CelebA-HQ oder LFW zurückgegriffen.

Usage:
    python download_face_dataset.py [--dataset widerface] [--splits train val]
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
import shutil
import argparse
import gdown


class FaceDatasetDownloader:
    """Download und organisiere Face-Detection Datasets"""

    # WIDER FACE URLs (Google Drive)
    WIDERFACE_URLS = {
        'train_images': 'https://drive.google.com/uc?id=15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M',
        'val_images': 'https://drive.google.com/uc?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q',
        'test_images': 'https://drive.google.com/uc?id=1HIfDbVEWKmsYKJZm4lchTBDLW5N7dY5T',
        'annotations': 'http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip'
    }

    # LFW (Labeled Faces in the Wild) - kleineres Dataset für schnelle Tests
    LFW_URL = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'

    def __init__(self, base_dir: str = "./face_download"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)

        self.download_dir = self.base_dir / "downloads"
        self.download_dir.mkdir(exist_ok=True)

        self.extract_dir = self.base_dir / "extracted"
        self.extract_dir.mkdir(exist_ok=True)

    def download_file(self, url: str, filename: str, use_gdown: bool = False) -> Path:
        """Download eine Datei mit Progress Bar"""
        filepath = self.download_dir / filename

        if filepath.exists():
            print(f"✓ {filename} bereits vorhanden, überspringe Download")
            return filepath

        print(f"📥 Downloade {filename}...")

        if use_gdown:
            # Google Drive Download via gdown
            gdown.download(url, str(filepath), quiet=False, fuzzy=True)
        else:
            # Regular HTTP Download
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=filename
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"✓ {filename} heruntergeladen")
        return filepath

    def extract_archive(self, archive_path: Path, extract_to: Path) -> None:
        """Extrahiere ZIP oder TAR Datei"""
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive nicht gefunden: {archive_path}")

        print(f"📦 Extrahiere {archive_path.name}...")

        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tgz', '.gz']:
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)

        print(f"✓ {archive_path.name} extrahiert")

    def download_widerface(self, splits: List[str] = ['train', 'val']) -> Path:
        """Download WIDER FACE Dataset"""
        print("\n" + "=" * 60)
        print("WIDER FACE Dataset Download")
        print("=" * 60)
        print("⚠️  WIDER FACE ist groß (~3.5 GB)")
        print("Splits:", splits)
        print("=" * 60 + "\n")

        # Download annotations
        anno_zip = self.download_file(
            self.WIDERFACE_URLS['annotations'],
            'wider_face_split.zip'
        )
        self.extract_archive(anno_zip, self.extract_dir)

        # Download image splits
        for split in splits:
            if split not in ['train', 'val', 'test']:
                print(f"⚠️  Unbekannter Split: {split}, überspringe")
                continue

            url_key = f'{split}_images'
            filename = f'WIDER_{split}.zip'

            print(f"\n📥 Lade {split} Images...")
            print("⏳ Dies kann mehrere Minuten dauern...")

            # Google Drive Download
            img_zip = self.download_file(
                self.WIDERFACE_URLS[url_key],
                filename,
                use_gdown=True
            )
            self.extract_archive(img_zip, self.extract_dir)

        return self.extract_dir / "WIDER_train" if "train" in splits else self.extract_dir

    def download_lfw(self) -> Path:
        """Download LFW Dataset (kleiner, für schnelle Tests)"""
        print("\n" + "=" * 60)
        print("LFW (Labeled Faces in the Wild) Dataset Download")
        print("=" * 60)
        print("✅ Kleineres Dataset (~170 MB)")
        print("✅ 13,000+ Gesichter-Bilder")
        print("=" * 60 + "\n")

        # Download
        lfw_tgz = self.download_file(
            self.LFW_URL,
            'lfw.tgz'
        )
        self.extract_archive(lfw_tgz, self.extract_dir)

        return self.extract_dir / "lfw"

    def organize_widerface_images(self, output_dir: Path) -> Dict[str, int]:
        """Organisiere WIDER FACE Bilder"""
        print("\n📁 Organisiere WIDER FACE Bilder...")

        stats = {}
        for split in ['train', 'val', 'test']:
            source_dir = self.extract_dir / f"WIDER_{split}" / "images"

            if not source_dir.exists():
                continue

            target_dir = output_dir / split / "face"
            target_dir.mkdir(exist_ok=True, parents=True)

            # Sammle alle Bilder aus Unterordnern
            images = list(source_dir.rglob('*.jpg'))
            print(f"\n  {split}: {len(images)} Bilder gefunden")

            # Kopiere Bilder
            copied = 0
            for img in tqdm(images, desc=f"  Kopiere {split}"):
                target_file = target_dir / f"{img.parent.name}_{img.name}"
                if not target_file.exists():
                    shutil.copy2(img, target_file)
                    copied += 1

            stats[split] = copied
            print(f"  ✓ {copied} Bilder nach {target_dir} kopiert")

        return stats

    def organize_lfw_images(self, output_dir: Path, split_ratio: tuple = (0.7, 0.2, 0.1)) -> Dict[str, int]:
        """Organisiere LFW Bilder und splitte in train/val/test"""
        print("\n📁 Organisiere LFW Bilder...")

        source_dir = self.extract_dir / "lfw"

        if not source_dir.exists():
            print(f"⚠️  LFW nicht gefunden in {source_dir}")
            return {}

        # Sammle alle Bilder
        images = list(source_dir.rglob('*.jpg'))
        print(f"  Gesamt: {len(images)} Bilder gefunden")

        # Split images
        import random
        random.shuffle(images)

        total = len(images)
        train_end = int(total * split_ratio[0])
        val_end = train_end + int(total * split_ratio[1])

        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        stats = {}
        for split, split_images in splits.items():
            target_dir = output_dir / split / "face"
            target_dir.mkdir(exist_ok=True, parents=True)

            print(f"\n  {split}: {len(split_images)} Bilder")

            copied = 0
            for img in tqdm(split_images, desc=f"  Kopiere {split}"):
                target_file = target_dir / f"{img.parent.name}_{img.name}"
                if not target_file.exists():
                    shutil.copy2(img, target_file)
                    copied += 1

            stats[split] = copied
            print(f"  ✓ {copied} Bilder nach {target_dir} kopiert")

        return stats

    def download_and_organize(
        self,
        dataset: str = 'lfw',
        splits: List[str] = ['train', 'val'],
        output_dir: str = None
    ) -> None:
        """Hauptfunktion: Download und Organisierung"""

        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = self.base_dir / "organized"

        output_path.mkdir(exist_ok=True, parents=True)

        if dataset == 'widerface':
            self.download_widerface(splits)
            stats = self.organize_widerface_images(output_path)
        elif dataset == 'lfw':
            self.download_lfw()
            stats = self.organize_lfw_images(output_path)
        else:
            raise ValueError(f"Unbekanntes Dataset: {dataset}")

        # Print summary
        print("\n" + "=" * 60)
        print("📊 ZUSAMMENFASSUNG")
        print("=" * 60)
        for split, count in stats.items():
            print(f"{split.upper()}: {count:,} Gesichter-Bilder")
        print(f"\nGESAMT: {sum(stats.values()):,} Bilder organisiert")
        print("=" * 60)
        print(f"\n✅ Daten verfügbar in: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Download Face-Detection Dataset für LFM2-VL Training'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='lfw',
        choices=['widerface', 'lfw'],
        help='Welches Dataset laden (default: lfw - kleiner & schneller)'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val'],
        choices=['train', 'val', 'test'],
        help='Welche Splits herunterladen (nur für WIDER FACE)'
    )
    parser.add_argument(
        '--download-dir',
        type=str,
        default='./face_download',
        help='Basis-Download-Verzeichnis (default: ./face_download)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output-Verzeichnis für organisierte Daten'
    )

    args = parser.parse_args()

    # Check for gdown if using widerface
    if args.dataset == 'widerface':
        try:
            import gdown
        except ImportError:
            print("\n⚠️  'gdown' ist erforderlich für WIDER FACE Download")
            print("Installation: pip install gdown")
            print("\nAlternativ: Nutze LFW Dataset mit --dataset lfw")
            return

    downloader = FaceDatasetDownloader(base_dir=args.download_dir)
    downloader.download_and_organize(
        dataset=args.dataset,
        splits=args.splits,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()