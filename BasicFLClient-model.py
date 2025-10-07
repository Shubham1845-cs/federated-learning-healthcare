"""
Level 1: Basic Neural Network Model for Medical Diagnosis
This file defines the architecture for our diagnostic AI model.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_cancer_model(input_shape=(30,)):
    """
    Creates a neural network for breast cancer detection.
    
    Input features: 30 numerical features from breast cancer dataset
    Output: Binary classification (0=benign, 1=malignant)
    
    Args:
        input_shape: Shape of input features (default: 30 for cancer dataset)
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First hidden layer with dropout for regularization
        layers.Dense(64, activation='relu', name='hidden_layer_1'),
        layers.Dropout(0.3),
        
        # Second hidden layer
        layers.Dense(32, activation='relu', name='hidden_layer_2'),
        layers.Dropout(0.2),
        
        # Third hidden layer
        layers.Dense(16, activation='relu', name='hidden_layer_3'),
        
        # Output layer (binary classification)
        layers.Dense(1, activation='sigmoid', name='output_layer')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    return model


def create_diabetes_model(input_shape=(8,)):
    """
    Creates a neural network for diabetes prediction.
    
    Input features: 8 numerical features (glucose, BMI, age, etc.)
    Output: Binary classification (0=no diabetes, 1=diabetes)
    
    Args:
        input_shape: Shape of input features (default: 8 for diabetes dataset)
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_heart_disease_model(input_shape=(13,)):
    """
    Creates a neural network for heart disease prediction.
    
    Input features: 13 numerical features (age, blood pressure, cholesterol, etc.)
    Output: Binary classification (0=no disease, 1=disease)
    
    Args:
        input_shape: Shape of input features (default: 13)
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Dense(48, activation='relu'),
        layers.Dropout(0.25),
        
        layers.Dense(24, activation='relu'),
        layers.Dropout(0.25),
        
        layers.Dense(12, activation='relu'),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model(disease_type, input_shape=None):
    """
    Factory function to get the appropriate model based on disease type.
    
    Args:
        disease_type: Type of disease ('cancer', 'diabetes', 'heart_disease')
        input_shape: Optional custom input shape
    
    Returns:
        Compiled Keras model for the specified disease type
    """
    if disease_type.lower() == 'cancer':
        return create_cancer_model(input_shape if input_shape else (30,))
    elif disease_type.lower() == 'diabetes':
        return create_diabetes_model(input_shape if input_shape else (8,))
    elif disease_type.lower() == 'heart_disease':
        return create_heart_disease_model(input_shape if input_shape else (13,))
    else:
        raise ValueError(f"Unknown disease type: {disease_type}")


# Model summary helper
if __name__ == "__main__":
    print("=" * 60)
    print("CANCER DETECTION MODEL")
    print("=" * 60)
    cancer_model = create_cancer_model()
    cancer_model.summary()
    
    print("\n" + "=" * 60)
    print("DIABETES PREDICTION MODEL")
    print("=" * 60)
    diabetes_model = create_diabetes_model()
    diabetes_model.summary()
    
    print("\n" + "=" * 60)
    print("HEART DISEASE PREDICTION MODEL")
    print("=" * 60)
    heart_model = create_heart_disease_model()
    heart_model.summary()