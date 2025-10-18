"""
Level 2: Advanced Neural Network Models
Improved architectures with better regularization and performance.

Improvements over Level 1:
- Batch Normalization for faster training
- More sophisticated dropout strategies
- Residual connections
- Better activation functions
- Learning rate scheduling
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np


class AdvancedMedicalModels:
    """Factory class for advanced medical diagnosis models."""
    
    @staticmethod
    def create_diabetes_model_advanced(input_shape=(8,), dropout_rate=0.3):
        """
        Advanced diabetes prediction model.
        
        Improvements:
        - Batch normalization for stable training
        - Residual connections
        - L2 regularization
        - Optimized architecture
        """
        inputs = layers.Input(shape=input_shape, name='input')
        
        # First block
        x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.01), name='dense_1')(inputs)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Activation('relu', name='relu_1')(x)
        x = layers.Dropout(dropout_rate, name='dropout_1')(x)
        
        # Second block
        x2 = layers.Dense(64, kernel_regularizer=regularizers.l2(0.01), name='dense_2')(x)
        x2 = layers.BatchNormalization(name='bn_2')(x2)
        x2 = layers.Activation('relu', name='relu_2')(x2)
        x2 = layers.Dropout(dropout_rate, name='dropout_2')(x2)
        
        # Residual connection
        x = layers.Add(name='residual_1')([x, x2])
        
        # Third block
        x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), name='dense_3')(x)
        x = layers.BatchNormalization(name='bn_3')(x)
        x = layers.Dropout(dropout_rate * 0.5, name='dropout_3')(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='diabetes_advanced')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        return model
    
    @staticmethod
    def create_cancer_model_advanced(input_shape=(30,), dropout_rate=0.3):
        """Advanced breast cancer detection model."""
        inputs = layers.Input(shape=input_shape, name='input')
        
        x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.01))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Residual block 1
        x1 = layers.Dense(128, kernel_regularizer=regularizers.l2(0.01))(x)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Activation('relu')(x1)
        x1 = layers.Dropout(dropout_rate)(x1)
        x = layers.Add()([x, x1])
        
        x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate * 0.7)(x)
        
        # Residual block 2
        x2 = layers.Dense(64, kernel_regularizer=regularizers.l2(0.01))(x)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation('relu')(x2)
        x = layers.Add()([x, x2])
        
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)
        
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name='cancer_advanced')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        return model

    @staticmethod
    def create_heart_disease_model_advanced(input_shape=(13,), dropout_rate=0.3):
        """Advanced heart disease prediction model."""
        inputs = layers.Input(shape=input_shape, name='input')
        
        x = layers.Dense(96, kernel_regularizer=regularizers.l2(0.01))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Residual block
        x1 = layers.Dense(96, kernel_regularizer=regularizers.l2(0.01))(x)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Activation('relu')(x1)
        x1 = layers.Dropout(dropout_rate)(x1)
        x = layers.Add()([x, x1])
        
        x = layers.Dense(48, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)
        
        x = layers.Dense(24, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name='heart_disease_advanced')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        return model
    
    @staticmethod
    def create_cardiovascular_model_advanced(input_shape=(11,), dropout_rate=0.3):
        """Advanced cardiovascular disease prediction model."""
        inputs = layers.Input(shape=input_shape, name='input')
        
        x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.01))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # First residual block
        x1 = layers.Dense(128, kernel_regularizer=regularizers.l2(0.01))(x)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Activation('relu')(x1)
        x1 = layers.Dropout(dropout_rate)(x1)
        x = layers.Add()([x, x1])
        
        x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate * 0.7)(x)
        
        # Second residual block
        x2 = layers.Dense(64, kernel_regularizer=regularizers.l2(0.01))(x)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation('relu')(x2)
        x = layers.Add()([x, x2])
        
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)
        
        x = layers.Dense(16, activation='relu')(x)
        
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name='cardiovascular_advanced')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        return model

    @staticmethod
    def get_callbacks(patience=5, min_delta=0.001):
        """Get training callbacks for better model training."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1,
                min_delta=min_delta
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        return callbacks


def get_advanced_model(disease_type, input_shape=None, dropout_rate=0.3):
    """
    Factory function to get advanced model based on disease type.
    """
    factory = AdvancedMedicalModels()
    
    if disease_type.lower() == 'diabetes':
        shape = input_shape if input_shape else (8,)
        return factory.create_diabetes_model_advanced(shape, dropout_rate)
    
    elif disease_type.lower() == 'heart_disease':
        shape = input_shape if input_shape else (13,)
        return factory.create_heart_disease_model_advanced(shape, dropout_rate)
    
    elif disease_type.lower() == 'cardiovascular':
        shape = input_shape if input_shape else (11,)
        return factory.create_cardiovascular_model_advanced(shape, dropout_rate)
    
    elif disease_type.lower() == 'cancer':
        shape = input_shape if input_shape else (30,)
        return factory.create_cancer_model_advanced(shape, dropout_rate)
    
    else:
        raise ValueError(
            f"Unknown disease type: {disease_type}\n"
            f"Supported: 'diabetes', 'heart_disease', 'cardiovascular', 'cancer'"
        )


# ==================== TESTING ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING ADVANCED MODELS - LEVEL 2")
    print("="*70)
    
    diseases = [
        ('diabetes', (8,)),
        ('heart_disease', (13,)),
        ('cardiovascular', (11,)),
        ('cancer', (30,))
    ]
    
    for disease, shape in diseases:
        print(f"\n{'='*70}")
        print(f"Testing: {disease.upper()} MODEL")
        print(f"{'='*70}")
        
        try:
            model = get_advanced_model(disease, shape)
            model.summary()
            
            print(f"\n✓ Model created successfully!")
            print(f"  Total parameters: {model.count_params():,}")
            
            # Test with dummy data
            X_test = np.random.randn(10, shape[0])
            y_pred = model.predict(X_test, verbose=0)
            print(f"  Test prediction shape: {y_pred.shape}")
            print(f"  Sample predictions: {y_pred[:3].flatten()}")
        
        except Exception as e:
            print(f"\n❌ Error creating {disease} model: {e}")
    
    # Test callbacks
    print(f"\n{'='*70}")
    print("TESTING TRAINING CALLBACKS")
    print(f"{'='*70}")
    callbacks = AdvancedMedicalModels.get_callbacks()
    for cb in callbacks:
        print(f"  ✓ {cb.__class__.__name__}")
    
    print(f"\n{'='*70}")
    print("ALL ADVANCED MODELS READY!")
    print(f"{'='*70}\n")