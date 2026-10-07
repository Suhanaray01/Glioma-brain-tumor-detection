"""
Data Loading and Preprocessing for Brain Tumor Classification
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_data_generators(data_dir, class_mode='binary', batch_size=32, 
                          img_size=224, seed=42):
    """
    Create train, validation, and test data generators.
    
    Args:
        data_dir: Base directory containing train/val/test subdirectories
        class_mode: 'binary' or 'categorical'
        batch_size: Number of samples per batch
        img_size: Target size for images
        seed: Random seed for reproducibility
        
    Returns:
        train_gen, val_gen, test_gen: Data generators
    """
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        shear_range=0.15,
        fill_mode='reflect'
    )

    # Validation and test data (only rescaling)
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # Paths
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    print(f"📁 Loading data from:")
    print(f"   Train: {train_dir}")
    print(f"   Val: {val_dir}")
    print(f"   Test: {test_dir}")

    # Create generators
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode=class_mode,
        seed=seed,
        shuffle=True
    )

    val_gen = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode=class_mode,
        seed=seed,
        shuffle=False
    )

    test_gen = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode=class_mode,
        seed=seed,
        shuffle=False
    )

    print(f"✅ Data loaded:")
    print(f"   Train: {train_gen.samples} samples")
    print(f"   Val: {val_gen.samples} samples")
    print(f"   Test: {test_gen.samples} samples")
    print(f"   Classes: {train_gen.class_indices}\n")

    return train_gen, val_gen, test_gen


def calculate_class_weights(generator):
    """
    Calculate class weights for handling imbalanced datasets.
    
    Args:
        generator: Keras data generator
        
    Returns:
        dict: Class weights
    """
    class_counts = np.bincount(generator.classes)
    total = len(generator.classes)
    class_weights = {
        i: total / (len(class_counts) * count) 
        for i, count in enumerate(class_counts)
    }
    
    print(f"⚖️ Class weights: {class_weights}")
    return class_weights


def preprocess_single_image(img_path, img_size=224):
    """
    Preprocess a single image for prediction.
    
    Args:
        img_path: Path to image file
        img_size: Target size
        
    Returns:
        Preprocessed image array ready for prediction
    """
    from tensorflow.keras.preprocessing import image
    
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array
