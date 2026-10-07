"""
Training Script for Brain Tumor Classification Models
Supports both binary and multiclass classification
"""

import os
import tensorflow as tf
from model_architecture import build_hybrid_model
from data_preprocessing import create_data_generators, calculate_class_weights


class Config:
    """Training configuration"""
    BASE_DIR = "/content/drive/MyDrive/brain_tumor_split"
    SAVE_DIR = "/content/drive/MyDrive/brain_tumor_models"
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 40
    LEARNING_RATE = 2e-4
    SEED = 42


def setup_gpu():
    """Configure GPU for optimal performance"""
    print("🔧 Setting up GPU optimization...")
    
    # Mixed precision for speed
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU ready: {len(gpus)} device(s)")
    
    # XLA compilation
    tf.config.optimizer.set_jit(True)
    print("✅ Optimizations enabled\n")


def train_model(model_type='binary', config=None):
    """
    Train hybrid model for binary or multiclass classification.
    
    Args:
        model_type: 'binary' or 'multiclass'
        config: Configuration object (uses default if None)
        
    Returns:
        history: Training history
        model: Trained model
    """
    if config is None:
        config = Config()
    
    print(f"\n{'='*70}")
    print(f"🚀 TRAINING {model_type.upper()} MODEL")
    print(f"{'='*70}\n")

    # Setup paths and parameters
    if model_type == 'binary':
        data_dir = os.path.join(config.BASE_DIR, "binary")
        num_classes = 2
        class_mode = 'binary'
    else:
        data_dir = os.path.join(config.BASE_DIR, "multiclass")
        num_classes = 4
        class_mode = 'categorical'

    # Create save directory
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # Load data
    train_gen, val_gen, test_gen = create_data_generators(
        data_dir, 
        class_mode, 
        config.BATCH_SIZE,
        config.IMG_SIZE,
        config.SEED
    )

    # Calculate class weights
    class_weights = calculate_class_weights(train_gen)

    # Build model
    model, loss = build_hybrid_model(num_classes=num_classes)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=config.LEARNING_RATE,
            weight_decay=1e-4
        ),
        loss=loss,
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(config.SAVE_DIR, f'{model_type}_best.keras'),
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(config.SAVE_DIR, f'{model_type}_training_log.csv')
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.SAVE_DIR, 'logs', model_type),
            histogram_freq=1
        )
    ]

    print("🎯 Starting training...\n")

    # Train
    history = model.fit(
        train_gen,
        epochs=config.EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Save final model
    model.save(os.path.join(config.SAVE_DIR, f'{model_type}_final.keras'))

    print(f"\n✅ Training completed!")
    print(f"📊 Best val_auc: {max(history.history['val_auc']):.4f}")
    print(f"📊 Best val_accuracy: {max(history.history['val_accuracy']):.4f}")

    return history, model


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("🧠 BRAIN TUMOR CLASSIFIER - PRODUCTION TRAINING")
    print("="*70)
    print(f"💾 TensorFlow: {tf.__version__}")
    
    config = Config()
    print(f"⚡ Batch Size: {config.BATCH_SIZE}")
    print(f"⚡ Epochs: {config.EPOCHS}")
    print(f"⚡ Learning Rate: {config.LEARNING_RATE}")
    print(f"📁 Save Directory: {config.SAVE_DIR}")
    print("="*70 + "\n")

    # Setup GPU
    setup_gpu()

    try:
        # Train Binary Model
        print("🎯 Training Binary Classification Model...")
        binary_hist, binary_model = train_model('binary', config)

        # Train Multiclass Model
        print("\n🎯 Training Multiclass Classification Model...")
        multi_hist, multi_model = train_model('multiclass', config)

        # Final summary
        print("\n" + "="*70)
        print("🎉 ALL TRAINING COMPLETED!")
        print("="*70)
        print(f"\n📦 Models saved in: {config.SAVE_DIR}")
        print(f"📋 TensorBoard logs: {os.path.join(config.SAVE_DIR, 'logs')}")
        print("\n💡 To view TensorBoard: tensorboard --logdir=" + 
              os.path.join(config.SAVE_DIR, 'logs'))
        print("="*70 + "\n")

        return binary_hist, multi_hist, binary_model, multi_model

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
