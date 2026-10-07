"""
Custom Keras Layers for Brain Tumor Classification
Implements PatchExtractor and PatchEncoder for Vision Transformer
"""

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable()
class PatchExtractor(layers.Layer):
    """
    Extracts patches from input images for Vision Transformer.
    
    Args:
        patch_size (int): Size of each square patch (default: 16)
    """
    
    def __init__(self, patch_size=16, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        """
        Extract patches from images.
        
        Args:
            images: Tensor of shape (batch_size, height, width, channels)
            
        Returns:
            Tensor of shape (batch_size, num_patches, patch_size*patch_size*channels)
        """
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


@tf.keras.utils.register_keras_serializable()
class PatchEncoder(layers.Layer):
    """
    Encodes image patches with linear projection and positional embeddings.
    
    Args:
        num_patches (int): Number of patches (default: 196 for 224x224 image with 16x16 patches)
        projection_dim (int): Dimension of the projection (default: 256)
    """
    
    def __init__(self, num_patches=196, projection_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        """
        Encode patches with projection and positional embeddings.
        
        Args:
            patch: Tensor of shape (batch_size, num_patches, patch_dims)
            
        Returns:
            Tensor of shape (batch_size, num_patches, projection_dim)
        """
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection.units
        })
        return config
