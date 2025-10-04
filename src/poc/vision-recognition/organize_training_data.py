#!/usr/bin/env python3
"""
Training Data Organizer für LFM2-VL Vision Recognition
=======================================================

Organisiert heruntergeladene COCO-Daten in die korrekte Ordnerstruktur:
- datatraingesicht, datatrainperson, datatrainauto (Training)
- datavalgesicht, datavalperson, datavalauto (Validation)
- datatestgesicht, datatestperson, datatestauto (Test)

Usage:
    python organize_training_data.py --source coco_download/organized
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Dict, List
import random
from tqdm import tqdm


class TrainingDataOrganizer:
    """Organisiert Daten in LFM2-VL Ordnerstruktur"""

    # Mapping: COCO category -> LFM2-VL folder suffix
    CATEGORY_MAPPING = {
        'person': 'person',
        'car': 'auto',
        'face': 'gesicht',
    }

    def __init__(self, source_dir: str, target_base_dir: str):
        self.source_dir = Path(source_dir)
        self.target_base_dir = Path(target_base_dir)

        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory nicht gefunden: {source_dir}")

    def create_target_structure(self) -> Dict[str, Path]:
        """Erstelle die Ziel-Ordnerstruktur"""
        print("📁 Erstelle Ordnerstruktur...")

        dirs = {}
        for split in ['train', 'val', 'test']:
            for category in ['gesicht', 'person', 'auto']:
                dir_name = f"data{split}{category}"
                dir_path = self.target_base_dir / dir_name
                dir_path.mkdir(exist_ok=True, parents=True)
                dirs[f"{split}_{category}"] = dir_path
                print(f"  ✓ {dir_name}")

        return dirs

    def split_data(
        self,
        images: List[Path],
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1
    ) -> Dict[str, List[Path]]:
        """Teile Daten in Train/Val/Test auf"""

        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 0.001:
            raise ValueError("Ratios müssen zusammen 1.0 ergeben")

        # Shuffle für zufällige Aufteilung
        random.shuffle(images)

        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        return {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

    def organize_category(
        self,
        category: str,
        target_dirs: Dict[str, Path],
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1
    ) -> Dict[str, int]:
        """Organisiere eine einzelne Kategorie"""

        print(f"\n{'=' * 60}")
        print(f"Organisiere Kategorie: {category.upper()}")
        print(f"{'=' * 60}")

        # Map COCO category to LFM2-VL suffix
        if category not in self.CATEGORY_MAPPING:
            print(f"⚠️  Kategorie {category} nicht unterstützt, überspringe")
            return {}

        lfm2_suffix = self.CATEGORY_MAPPING[category]
        stats = {}

        # Process train and val directories
        for split in ['train', 'val']:
            source_cat_dir = self.source_dir / split / category

            if not source_cat_dir.exists():
                print(f"⚠️  {source_cat_dir} nicht gefunden, überspringe {split}")
                continue

            # Sammle alle Bilder
            images = list(source_cat_dir.glob('*.jpg'))
            print(f"\n📊 {split}: {len(images)} Bilder gefunden")

            if not images:
                continue

            # Wenn es ein val Split ist, verwende es direkt als Validation
            if split == 'val':
                # Split val into val and test
                val_test_split = self.split_data(
                    images,
                    train_ratio=0.0,
                    val_ratio=0.67,
                    test_ratio=0.33
                )

                # Copy validation images
                target_val = target_dirs[f"val_{lfm2_suffix}"]
                self._copy_images(val_test_split['val'], target_val, f"val_{category}")
                stats[f'val_{category}'] = len(val_test_split['val'])

                # Copy test images
                target_test = target_dirs[f"test_{lfm2_suffix}"]
                self._copy_images(val_test_split['test'], target_test, f"test_{category}")
                stats[f'test_{category}'] = len(val_test_split['test'])

            elif split == 'train':
                # Use train split as training data
                target_train = target_dirs[f"train_{lfm2_suffix}"]
                self._copy_images(images, target_train, f"train_{category}")
                stats[f'train_{category}'] = len(images)

        return stats

    def _copy_images(self, images: List[Path], target_dir: Path, desc: str) -> None:
        """Kopiere Bilder mit Progress Bar"""
        for img in tqdm(images, desc=f"  Kopiere {desc}"):
            target_file = target_dir / img.name
            if not target_file.exists():
                shutil.copy2(img, target_file)

    def organize_all(
        self,
        categories: List[str] = ['person', 'car'],
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1
    ) -> None:
        """Organisiere alle Kategorien"""

        print("=" * 60)
        print("LFM2-VL Training Data Organisation")
        print("=" * 60)
        print(f"Source: {self.source_dir}")
        print(f"Target: {self.target_base_dir}")
        print(f"Split Ratio: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}")
        print("=" * 60)

        # Create target structure
        target_dirs = self.create_target_structure()

        # Organize each category
        all_stats = {}
        for category in categories:
            stats = self.organize_category(
                category,
                target_dirs,
                train_ratio,
                val_ratio,
                test_ratio
            )
            all_stats.update(stats)

        # Print summary
        self._print_summary(all_stats)

    def _print_summary(self, stats: Dict[str, int]) -> None:
        """Drucke Zusammenfassung"""
        print("\n" + "=" * 60)
        print("📊 ZUSAMMENFASSUNG")
        print("=" * 60)

        for split in ['train', 'val', 'test']:
            print(f"\n{split.upper()}:")
            for category in ['person', 'car']:
                key = f'{split}_{category}'
                count = stats.get(key, 0)
                lfm2_suffix = self.CATEGORY_MAPPING.get(category, category)
                print(f"  data{split}{lfm2_suffix}: {count:,} Bilder")

        total = sum(stats.values())
        print(f"\n{'=' * 60}")
        print(f"GESAMT: {total:,} Bilder organisiert")
        print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description='Organisiere COCO Daten für LFM2-VL Training'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source-Verzeichnis mit COCO Daten (z.B. coco_download/organized)'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='.',
        help='Ziel-Verzeichnis (default: aktuelles Verzeichnis)'
    )
    parser.add_argument(
        '--categories',
        nargs='+',
        default=['person', 'car', 'face'],
        choices=['person', 'car', 'face'],
        help='Welche Kategorien organisieren (default: person car)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Anteil Training-Daten (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Anteil Validation-Daten (default: 0.2)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Anteil Test-Daten (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random Seed für reproduzierbare Splits (default: 42)'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    organizer = TrainingDataOrganizer(
        source_dir=args.source,
        target_base_dir=args.target
    )
    organizer.organize_all(
        categories=args.categories,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )


if __name__ == "__main__":
    main()