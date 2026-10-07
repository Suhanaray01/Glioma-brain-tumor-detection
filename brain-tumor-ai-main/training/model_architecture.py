"""
Hybrid CNN + Vision Transformer Model Architecture
For Brain Tumor Classification
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications
from custom_layers import PatchExtractor, PatchEncoder


def create_vit_model(input_shape=(224, 224, 3), projection_dim=256, 
                     num_heads=8, transformer_layers=4):
    """
    Create Vision Transformer model.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        projection_dim: Dimension of patch projections
        num_heads: Number of attention heads
        transformer_layers: Number of transformer blocks
        
    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=input_shape)

    # Patch extraction and encoding
    patches = PatchExtractor(patch_size=16)(inputs)
    encoded_patches = PatchEncoder(num_patches=196, projection_dim=projection_dim)(patches)

    # Transformer blocks
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=projection_dim // num_heads, 
            dropout=0.15
        )(x1, x1)
        
        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = layers.Dense(projection_dim * 3, activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.15)(x3)
        x3 = layers.Dense(projection_dim)(x3)
        x3 = layers.Dropout(0.15)(x3)

        # Skip connection 2
        encoded_patches = layers.Add()([x3, x2])

    # Global average pooling
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.GlobalAveragePooling1D()(representation)
    
    return models.Model(inputs=inputs, outputs=representation, name='vision_transformer')


def build_hybrid_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Build hybrid CNN + ViT model with Grad-CAM support.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes (2 for binary, 4 for multiclass)
        
    Returns:
        model: Compiled Keras model
        loss: Loss function name
    """
    inputs = layers.Input(shape=input_shape)

    # ===== CNN Branch (EfficientNetV2B0) =====
    cnn_base = applications.EfficientNetV2B0(
        include_top=False,
        input_tensor=inputs,
        weights='imagenet',
        pooling=None
    )

    # Freeze early layers for transfer learning
    for layer in cnn_base.layers[:-40]:
        layer.trainable = False

    cnn_output = cnn_base.output

    # Add named conv layer for Grad-CAM compatibility
    cnn_output = layers.Conv2D(256, 1, padding='same', name='top_conv')(cnn_output)

    # CNN feature extraction
    cnn_features = layers.GlobalAveragePooling2D()(cnn_output)
    cnn_features = layers.Dense(256, activation='swish')(cnn_features)
    cnn_features = layers.BatchNormalization()(cnn_features)
    cnn_features = layers.Dropout(0.4)(cnn_features)

    # ===== ViT Branch =====
    vit_model = create_vit_model(
        input_shape=input_shape,
        projection_dim=256,
        num_heads=8,
        transformer_layers=4
    )
    vit_features = vit_model(inputs)
    vit_features = layers.Dense(256, activation='swish')(vit_features)
    vit_features = layers.BatchNormalization()(vit_features)
    vit_features = layers.Dropout(0.4)(vit_features)

    # ===== Attention-based Fusion =====
    cnn_attention = layers.Dense(1, activation='sigmoid')(cnn_features)
    vit_attention = layers.Dense(1, activation='sigmoid')(vit_features)

    cnn_weighted = layers.Multiply()([cnn_features, cnn_attention])
    vit_weighted = layers.Multiply()([vit_features, vit_attention])

    # Concatenate weighted features
    combined = layers.Concatenate()([cnn_weighted, vit_weighted])

    # ===== Classification Head =====
    x = layers.Dense(256, activation='swish', 
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation='swish', 
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)

    # Output layer
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid', dtype='float32', name='output')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name='output')(x)
        loss = 'categorical_crossentropy'

    model = models.Model(inputs=inputs, outputs=outputs, name='hybrid_cnn_vit')
    
    print(f"✅ Model built: {model.count_params():,} parameters")
    print(f"🎯 Grad-CAM layer: 'top_conv'")

    return model, loss


def load_model_with_custom_objects(model_path):
    """
    Load saved model with custom objects.
    
    Args:
        model_path: Path to saved .keras model file
        
    Returns:
        Loaded Keras model
    """
    custom_objects = {
        'PatchExtractor': PatchExtractor,
        'PatchEncoder': PatchEncoder
    }
    
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects=custom_objects,
        compile=False
    )
    
    return model
