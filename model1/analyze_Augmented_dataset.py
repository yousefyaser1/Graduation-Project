"""
Augmented Dataset Analyzer
Analyzes the image dataset to count images and show samples
"""

import os
from pathlib import Path

# Dataset root directory
DATASET_ROOT = Path("Augmented_Dataset")

# Split directories
SPLITS = ["train", "test", "val"]

# Categories (classes)
CATEGORIES = ["acne", "eczema", "tinea"]


def count_images_in_category(split: str, category: str) -> int:
    """Count images in a specific split and category."""
    category_dir = DATASET_ROOT / split / category
    if not category_dir.exists():
        return 0
    
    # Count .jpg and .jpeg files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    count: int = sum(1 for file in category_dir.iterdir() if file.suffix in image_extensions)
    return count


def get_sample_images(split: str, category: str, num_samples: int = 3) -> list[str]:
    """Get sample image filenames from a category."""
    category_dir = DATASET_ROOT / split / category
    if not category_dir.exists():
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    result: list[str] = []
    for f in category_dir.iterdir():
        if f.suffix in image_extensions:
            result.append(f.name)
            if len(result) >= num_samples:
                break
    return result


def analyze_dataset():
    """Analyze the entire dataset."""
    print("=" * 60)
    print("AUGMENTED DATASET ANALYSIS")
    print("=" * 60)
    print()
    
    # Track totals
    total_images: int = 0
    split_totals: dict[str, int] = {}
    category_totals: dict[str, int] = {cat: 0 for cat in CATEGORIES}
    
    # Analyze each split
    for split in SPLITS:
        print(f"\n{'='*60}")
        print(f"SPLIT: {split.upper()}")
        print(f"{'='*60}")
        
        split_count: int = sum(count_images_in_category(split, category) for category in CATEGORIES)
        for category in CATEGORIES:
            count: int = count_images_in_category(split, category)
            current_count: int = category_totals.get(category, 0)
            category_totals[category] = current_count + count
            
            print(f"\n  {category.upper()}: {count} images")
            
            # Show sample images
            samples = get_sample_images(split, category, num_samples=3)
            if samples:
                print(f"    Sample images:")
                for sample in samples:
                    print(f"      - {sample}")
        
        split_totals[split] = split_count
        total_images = sum([total_images, split_count])
        print(f"\n  >>> {split.upper()} Total: {split_count} images")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nTotal Images: {total_images}")
    
    print(f"\nImages per Split:")
    for split in SPLITS:
        print(f"  - {split}: {split_totals[split]} images")
    
    print(f"\nImages per Category:")
    for category in CATEGORIES:
        print(f"  - {category}: {category_totals[category]} images")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    analyze_dataset()
