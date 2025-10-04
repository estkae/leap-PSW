#!/usr/bin/env python3
"""
COCO Dataset Downloader für LFM2-VL Vision Recognition Training
=================================================================

Lädt automatisch COCO Dataset herunter und filtert nach relevanten Kategorien:
- Personen (person)
- Autos (car)
- Gesichter (wird aus person-Bildern extrahiert)

Usage:
    python download_coco_dataset.py [--split train] [--categories person car]
"""

import os
import json
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Set
import shutil
import argparse


class COCODownloader:
    """Download und organisiere COCO Dataset für spezifische Kategorien"""

    # COCO 2017 URLs
    COCO_URLS = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'train_annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
    }

    # Relevante COCO Kategorien
    CATEGORY_MAPPING = {
        'person': 1,
        'car': 3,
        'bicycle': 2,
        'motorcycle': 4,
        'bus': 6,
        'truck': 8,
    }

    def __init__(self, base_dir: str = "./coco_download"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)

        self.download_dir = self.base_dir / "downloads"
        self.download_dir.mkdir(exist_ok=True)

        self.extract_dir = self.base_dir / "extracted"
        self.extract_dir.mkdir(exist_ok=True)

    def download_file(self, url: str, filename: str) -> Path:
        """Download eine Datei mit Progress Bar"""
        filepath = self.download_dir / filename

        if filepath.exists():
            print(f"✓ {filename} bereits vorhanden, überspringe Download")
            return filepath

        print(f"📥 Downloade {filename}...")
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

    def extract_zip(self, zip_path: Path, extract_to: Path) -> None:
        """Extrahiere ZIP-Datei"""
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP-Datei nicht gefunden: {zip_path}")

        print(f"📦 Extrahiere {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ {zip_path.name} extrahiert")

    def load_annotations(self, split: str = 'train') -> Dict:
        """Lade COCO Annotationen"""
        anno_file = self.extract_dir / "annotations" / f"instances_{split}2017.json"

        if not anno_file.exists():
            raise FileNotFoundError(f"Annotations nicht gefunden: {anno_file}")

        print(f"📖 Lade Annotationen für {split}...")
        with open(anno_file, 'r') as f:
            return json.load(f)

    def filter_images_by_category(
        self,
        annotations: Dict,
        categories: List[str]
    ) -> Dict[str, Set[int]]:
        """Filtere Bilder nach Kategorien"""
        print(f"🔍 Filtere Bilder nach Kategorien: {categories}")

        # Mapping: category_name -> image_ids
        category_images = {cat: set() for cat in categories}

        # Get category IDs
        category_ids = {
            self.CATEGORY_MAPPING[cat]
            for cat in categories
            if cat in self.CATEGORY_MAPPING
        }

        # Filter annotations
        for anno in tqdm(annotations['annotations'], desc="Filter Annotationen"):
            if anno['category_id'] in category_ids:
                image_id = anno['image_id']

                # Find category name
                for cat, cat_id in self.CATEGORY_MAPPING.items():
                    if cat_id == anno['category_id'] and cat in categories:
                        category_images[cat].add(image_id)

        # Print statistics
        for cat, img_ids in category_images.items():
            print(f"  ✓ {cat}: {len(img_ids)} Bilder gefunden")

        return category_images

    def organize_images(
        self,
        split: str,
        category_images: Dict[str, Set[int]],
        annotations: Dict,
        output_base: Path
    ) -> None:
        """Organisiere Bilder in Ordnerstruktur"""
        print(f"\n📁 Organisiere Bilder für {split}...")

        # Create image_id -> filename mapping
        image_mapping = {
            img['id']: img['file_name']
            for img in annotations['images']
        }

        source_dir = self.extract_dir / f"{split}2017"

        for category, image_ids in category_images.items():
            target_dir = output_base / category
            target_dir.mkdir(exist_ok=True, parents=True)

            print(f"\n  📂 Kopiere {category} Bilder...")
            copied = 0

            for img_id in tqdm(list(image_ids), desc=f"  {category}"):
                if img_id not in image_mapping:
                    continue

                filename = image_mapping[img_id]
                source_file = source_dir / filename
                target_file = target_dir / filename

                if source_file.exists() and not target_file.exists():
                    shutil.copy2(source_file, target_file)
                    copied += 1

            print(f"  ✓ {copied} Bilder nach {target_dir} kopiert")

    def download_and_organize(
        self,
        splits: List[str] = ['train', 'val'],
        categories: List[str] = ['person', 'car'],
        output_dir: str = None
    ) -> None:
        """Hauptfunktion: Download und Organisierung"""

        print("=" * 60)
        print("COCO Dataset Download & Organisation")
        print("=" * 60)
        print(f"Splits: {splits}")
        print(f"Kategorien: {categories}")
        print("=" * 60)

        # 1. Download Annotations (nur einmal nötig)
        anno_zip = self.download_file(
            self.COCO_URLS['train_annotations'],
            'annotations_trainval2017.zip'
        )
        self.extract_zip(anno_zip, self.extract_dir)

        # 2. Process each split
        for split in splits:
            print(f"\n{'=' * 60}")
            print(f"Verarbeite {split.upper()} Split")
            print(f"{'=' * 60}")

            # Download images
            url_key = f'{split}_images'
            if url_key not in self.COCO_URLS:
                print(f"⚠️  URL für {split} nicht verfügbar, überspringe")
                continue

            img_zip = self.download_file(
                self.COCO_URLS[url_key],
                f'{split}2017.zip'
            )
            self.extract_zip(img_zip, self.extract_dir)

            # Load annotations
            annotations = self.load_annotations(split)

            # Filter by categories
            category_images = self.filter_images_by_category(
                annotations,
                categories
            )

            # Organize images
            if output_dir:
                output_base = Path(output_dir) / f"data{split}"
            else:
                output_base = self.base_dir / "organized" / split

            self.organize_images(
                split,
                category_images,
                annotations,
                output_base
            )

        print("\n" + "=" * 60)
        print("✅ COCO Dataset erfolgreich heruntergeladen und organisiert!")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Download COCO Dataset für LFM2-VL Training'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val'],
        choices=['train', 'val'],
        help='Welche Splits herunterladen (default: train val)'
    )
    parser.add_argument(
        '--categories',
        nargs='+',
        default=['person', 'car'],
        choices=['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck'],
        help='Welche Kategorien filtern (default: person car)'
    )
    parser.add_argument(
        '--download-dir',
        type=str,
        default='./coco_download',
        help='Basis-Download-Verzeichnis (default: ./coco_download)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output-Verzeichnis für organisierte Daten (default: coco_download/organized)'
    )

    args = parser.parse_args()

    downloader = COCODownloader(base_dir=args.download_dir)
    downloader.download_and_organize(
        splits=args.splits,
        categories=args.categories,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()