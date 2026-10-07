"""
Dataset Builder for Brain Tumor Classification
Organizes raw MRI data into train/val/test splits
"""

import os
import shutil
import random
import glob
from pathlib import Path
from tqdm import tqdm


class DatasetConfig:
    """Dataset configuration"""
    BASE_DIR = "/content/drive/MyDrive/brain_tumor"
    OUTPUT_DIR = "/content/drive/MyDrive/brain_tumor_split"
    SEED = 42
    SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, val, test


def create_directory_structure(output_dir):
    """
    Create directory structure for binary and multiclass datasets.
    
    Args:
        output_dir: Output base directory
        
    Returns:
        binary_base, multi_base: Paths to binary and multiclass directories
    """
    # Binary classification structure
    binary_base = Path(output_dir) / "binary"
    splits = ["train", "val", "test"]
    binary_classes = ["tumor", "no_tumor"]

    for split in splits:
        for cls in binary_classes:
            (binary_base / split / cls).mkdir(parents=True, exist_ok=True)

    # Multi-class classification structure
    multi_base = Path(output_dir) / "multiclass"
    multi_classes = ["Glioma", "Meningioma", "Pituitary", "Normal"]

    for split in splits:
        for cls in multi_classes:
            (multi_base / split / cls).mkdir(parents=True, exist_ok=True)

    print("✅ Created directory structure:")
    print(f"   📁 {binary_base}")
    print(f"   📁 {multi_base}\n")

    return binary_base, multi_base


def get_all_image_files(source_path):
    """
    Get all image files from a directory.
    
    Args:
        source_path: Source directory path
        
    Returns:
        list: List of image file paths
    """
    image_files = []
    
    if not source_path.exists():
        return image_files

    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

    for ext in extensions:
        pattern = str(source_path / f"*{ext}")
        image_files.extend(glob.glob(pattern))
        
        pattern_sub = str(source_path / f"*/*{ext}")
        image_files.extend(glob.glob(pattern_sub))

    return list(set([Path(f) for f in image_files]))


def copy_and_split_files(sources, output_dir, class_name, split_ratios, base_dir, seed=42):
    """
    Copy files from sources to train/val/test splits.
    
    Args:
        sources: List of source directories
        output_dir: Output directory
        class_name: Class name
        split_ratios: Tuple of (train, val, test) ratios
        base_dir: Base directory for sources
        seed: Random seed
        
    Returns:
        int: Number of files copied
    """
    all_files = []

    for source in sources:
        source_path = Path(base_dir) / source
        source_files = get_all_image_files(source_path)
        all_files.extend(source_files)

    if not all_files:
        print(f"   ⚠️ No images found for {class_name}")
        return 0

    all_files = list(set(all_files))
    random.seed(seed)
    random.shuffle(all_files)

    n = len(all_files)
    train_cut = int(split_ratios[0] * n)
    val_cut = train_cut + int(split_ratios[1] * n)

    print(f"   📊 {class_name}: {n} images → "
          f"train:{train_cut}, val:{val_cut-train_cut}, test:{n-val_cut}")

    copied_count = 0
    for i, file_path in enumerate(tqdm(all_files, desc=f"Copying {class_name}")):
        if i < train_cut:
            dest = output_dir / "train" / class_name / file_path.name
        elif i < val_cut:
            dest = output_dir / "val" / class_name / file_path.name
        else:
            dest = output_dir / "test" / class_name / file_path.name

        try:
            shutil.copy2(file_path, dest)
            copied_count += 1
        except Exception as e:
            print(f"   ❌ Error copying {file_path}: {e}")

    print(f"   ✅ Successfully copied {copied_count}/{n} images\n")
    return copied_count


def build_binary_dataset(binary_base, tumor_sources, no_tumor_sources, 
                         config):
    """
    Build binary classification dataset.
    
    Args:
        binary_base: Binary dataset base directory
        tumor_sources: List of tumor data sources
        no_tumor_sources: List of no tumor data sources
        config: Configuration object
        
    Returns:
        tuple: (tumor_count, no_tumor_count)
    """
    print("🔧 Building Binary Classification Dataset...")

    tumor_count = copy_and_split_files(
        tumor_sources, 
        binary_base, 
        "tumor",
        config.SPLIT_RATIOS,
        config.BASE_DIR,
        config.SEED
    )

    no_tumor_count = copy_and_split_files(
        no_tumor_sources, 
        binary_base, 
        "no_tumor",
        config.SPLIT_RATIOS,
        config.BASE_DIR,
        config.SEED
    )

    print(f"✅ Binary dataset created:")
    print(f"   🎯 Tumor: {tumor_count} images")
    print(f"   🎯 No Tumor: {no_tumor_count} images\n")

    return tumor_count, no_tumor_count


def build_multiclass_dataset(multi_base, config):
    """
    Build multiclass classification dataset.
    
    Args:
        multi_base: Multiclass dataset base directory
        config: Configuration object
        
    Returns:
        dict: Class counts
    """
    print("🔧 Building Multi-Class Classification Dataset...")

    multi_sources = {
        "Glioma": [
            "brain_tumor_new/Augmented/512Glioma",
            "brain_tumor_new/Raw/512Glioma"
        ],
        "Meningioma": [
            "brain_tumor_new/Augmented/512Meningioma",
            "brain_tumor_new/Raw/512Meningioma"
        ],
        "Pituitary": [
            "brain_tumor_new/Augmented/512Pituitary",
            "brain_tumor_new/Raw/512Pituitary"
        ],
        "Normal": [
            "brain_tumor_new/Augmented/512Normal",
            "brain_tumor_new/Raw/512Normal"
        ]
    }

    class_counts = {}

    for class_name, sources in multi_sources.items():
        print(f"\n   Processing {class_name}...")
        available_sources = [
            s for s in sources 
            if (Path(config.BASE_DIR) / s).exists()
        ]
        
        if available_sources:
            count = copy_and_split_files(
                available_sources, 
                multi_base, 
                class_name,
                config.SPLIT_RATIOS,
                config.BASE_DIR,
                config.SEED
            )
            class_counts[class_name] = count
        else:
            print(f"   ⚠️ No sources found for {class_name}")
            class_counts[class_name] = 0

    print(f"✅ Multi-class dataset created:")
    for class_name, count in class_counts.items():
        print(f"   🎯 {class_name}: {count} images")
    print()

    return class_counts


def main():
    """Main dataset building pipeline"""
    print("🧠 BRAIN TUMOR DATASET BUILDER")
    print("=" * 60 + "\n")
    
    config = DatasetConfig()
    
    # Set random seed
    random.seed(config.SEED)
    
    # Create directory structure
    binary_base, multi_base = create_directory_structure(config.OUTPUT_DIR)
    
    # Define data sources
    tumor_sources = [
        "brain_tumor_dataset/yes",
        "brain_tumor_new/Augmented/512Glioma",
        "brain_tumor_new/Augmented/512Meningioma",
        "brain_tumor_new/Augmented/512Pituitary",
        "brain_tumor_new/Raw/512Glioma",
        "brain_tumor_new/Raw/512Meningioma",
        "brain_tumor_new/Raw/512Pituitary",
        "mat_converted"
    ]
    
    no_tumor_sources = [
        "brain_tumor_dataset/no",
        "brain_tumor_new/Augmented/512Normal",
        "brain_tumor_new/Raw/512Normal"
    ]
    
    # Build datasets
    tumor_count, no_tumor_count = build_binary_dataset(
        binary_base, 
        tumor_sources, 
        no_tumor_sources,
        config
    )
    
    class_counts = build_multiclass_dataset(multi_base, config)
    
    # Summary
    print("=" * 60)
    print("🎉 DATASET BUILDING COMPLETED!")
    print("=" * 60)
    print(f"\n📁 Dataset location: {config.OUTPUT_DIR}")
    print(f"\n📊 Total Binary: {tumor_count + no_tumor_count}")
    print(f"📊 Total Multiclass: {sum(class_counts.values())}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
