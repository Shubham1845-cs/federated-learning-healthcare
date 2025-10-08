"""
Enhanced Data Loader - Level 1+ with Kaggle Dataset Support
This loader can use BOTH built-in datasets AND real Kaggle CSV files.

USAGE:
1. Built-in data (no download): loader = MedicalDataLoader('hospital_a', 'cancer')
2. Kaggle data (after download): loader = MedicalDataLoader('hospital_a', 'diabetes', use_kaggle=True)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import os
from pathlib import Path

class EnhancedMedicalDataLoader:
    """
    Enhanced data loader supporting both built-in and Kaggle datasets.
    """
    
    def __init__(self, hospital_id, disease_type, use_kaggle=False, kaggle_paths=None):
        """
        Initialize enhanced data loader.
        
        Args:
            hospital_id: Unique identifier (e.g., 'hospital_a', 'clinic_1')
            disease_type: 'cancer', 'diabetes', 'heart_disease', 'stroke'
            use_kaggle: If True, load from Kaggle CSV files
            kaggle_paths: A dictionary mapping disease_type to a full file path.
        """
        self.hospital_id = hospital_id
        self.disease_type = disease_type
        self.use_kaggle = use_kaggle
        self.scaler = StandardScaler()
        # Store the custom paths provided by the user
        self.kaggle_paths = kaggle_paths or {}
        
        print(f"\n{'='*60}")
        print(f"Enhanced Data Loader: {self.hospital_id}")
        print(f"Disease: {self.disease_type}")
        print(f"Source: {'Kaggle CSV' if use_kaggle else 'Built-in'}")
        print(f"{'='*60}")
    
    # ========== KAGGLE DATASET LOADERS ==========
    
    def load_diabetes_kaggle(self):
        # Use the custom path if provided, otherwise fall back to a default
        csv_path = Path(self.kaggle_paths.get('diabetes', 'datasets/diabetes/diabetes.csv'))
        
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Kaggle diabetes dataset not found at {csv_path}\n"
                f"Please verify the path in the `custom_kaggle_paths` dictionary."
            )
        
        print(f"Loading Kaggle diabetes data from: {csv_path}")
        df = pd.read_csv(csv_path)
        X = df.drop('Outcome', axis=1).values
        y = df['Outcome'].values
        
        print(f"  Total samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Positive cases: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
        
        X_local, y_local = self._partition_data(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X_local, y_local, test_size=0.2, random_state=42, stratify=y_local
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"  Hospital {self.hospital_id} partition:")
        print(f"    Training: {len(X_train)} samples, Testing: {len(X_test)} samples")
        return X_train, X_test, y_train, y_test
    
    def load_heart_disease_kaggle(self):
        # Use the custom path if provided, otherwise fall back to a default
        csv_path = Path(self.kaggle_paths.get('heart_disease', 'datasets/heart_disease/heart.csv'))
        
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Kaggle heart disease dataset not found at {csv_path}\n"
                f"Please verify the path in the `custom_kaggle_paths` dictionary."
            )
        
        print(f"Loading Kaggle heart disease data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Determine the target column based on common dataset variations
        target_col = 'target' if 'target' in df.columns else 'output'
        
        X = df.drop(target_col, axis=1).values
        y = df[target_col].values
        
        print(f"  Total samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Positive cases: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
        
        X_local, y_local = self._partition_data(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X_local, y_local, test_size=0.2, random_state=42, stratify=y_local
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"  Hospital {self.hospital_id} partition:")
        print(f"    Training: {len(X_train)} samples, Testing: {len(X_test)} samples")
        return X_train, X_test, y_train, y_test

    def load_stroke_kaggle(self):
        # Use the custom path if provided, otherwise fall back to a default
        csv_path = Path(self.kaggle_paths.get('stroke', 'datasets/stroke/healthcare-dataset-stroke-data.csv'))
        
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Kaggle stroke dataset not found at {csv_path}\n"
                f"Please verify the path in the `custom_kaggle_paths` dictionary."
            )
        
        print(f"Loading Kaggle stroke data from: {csv_path}")
        df = pd.read_csv(csv_path)
        df['bmi'].fillna(df['bmi'].median(), inplace=True)
        df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)
        
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
            
        X = df.drop('stroke', axis=1).values
        y = df['stroke'].values
        
        print(f"  Total samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Positive cases: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
        
        X_local, y_local = self._partition_data(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X_local, y_local, test_size=0.2, random_state=42
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        print(f"  Hospital {self.hospital_id} partition:")
        print(f"    Training: {len(X_train)} samples, Testing: {len(X_test)} samples")
        return X_train, X_test, y_train, y_test

    # ========== BUILT-IN DATASET LOADERS ==========
    
    def load_cancer_builtin(self):
        data = load_breast_cancer()
        X, y = data.data, data.target
        X_local, y_local = self._partition_data(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X_local, y_local, test_size=0.2, random_state=42, stratify=y_local
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        print(f"  Built-in cancer dataset loaded. Training: {len(X_train)}, Testing: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    # ... (Other built-in loaders can be added here if needed) ...
    
    # ========== HELPER METHODS ==========
    
    def _partition_data(self, X, y):
        num_hospitals = 3
        hospital_index = {
            'hospital_a': 0, 'hospital_b': 1, 'hospital_c': 2,
            'clinic_1': 0, 'clinic_2': 1,
            'hospital_1': 0, 'hospital_2': 1
        }
        idx = hospital_index.get(self.hospital_id, 0)
        partition_size = len(X) // num_hospitals
        start_idx = idx * partition_size
        end_idx = start_idx + partition_size if idx < num_hospitals - 1 else len(X)
        return X[start_idx:end_idx], y[start_idx:end_idx]
    
    # ========== MAIN LOAD METHOD ==========
    
    def load_data(self):
        if self.use_kaggle:
            if self.disease_type.lower() == 'diabetes':
                return self.load_diabetes_kaggle()
            elif self.disease_type.lower() == 'heart_disease':
                return self.load_heart_disease_kaggle()
            elif self.disease_type.lower() == 'stroke':
                return self.load_stroke_kaggle()
            elif self.disease_type.lower() == 'cancer':
                return self.load_cancer_builtin()  # Fallback to built-in for cancer
            else:
                raise ValueError(f"Unknown Kaggle disease type: {self.disease_type}")
        else:
            # Load built-in/simulated datasets
            if self.disease_type.lower() == 'cancer':
                return self.load_cancer_builtin()
            else:
                # Add other simulated loaders here if you create them
                raise ValueError(f"No built-in/simulated loader for: {self.disease_type}")
    
    def get_data_statistics(self, X_train, y_train):
        """Get statistics about the local dataset."""
        stats = {
            'hospital_id': self.hospital_id, 'disease_type': self.disease_type,
            'data_source': 'Kaggle' if self.use_kaggle else 'Built-in',
            'num_samples': len(X_train), 'num_features': X_train.shape[1],
            'positive_cases': int(np.sum(y_train)),
            'negative_cases': int(len(y_train) - np.sum(y_train)),
            'class_balance': float(np.mean(y_train))
        }
        return stats

# ========== TESTING AND DEMONSTRATION ==========

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING ENHANCED DATA LOADER")
    print("="*70)

    # --- DEFINE YOUR CUSTOM KAGGLE DATASET PATHS HERE ---
    # Use raw strings (r"...") to handle Windows paths correctly.
    custom_kaggle_paths = {
        "diabetes": r"D:\MLproject\archive (1)\diabetes.csv",
        "heart_disease": r"D:\MLproject\archive (2)\heart.csv",
        # Assuming your 'cardio_train.csv' is for heart disease.
        # If you meant the stroke dataset, change the key to "stroke".
        "stroke": r"D:\MLproject\archive (3)\cardio_train.csv" 
    }
    
    print("\n>>> TEST 1: Kaggle Datasets (Using your custom paths) <<<\n")
    
    try:
        # Test the diabetes loader
        loader_diabetes = EnhancedMedicalDataLoader(
            'hospital_a', 'diabetes', use_kaggle=True, kaggle_paths=custom_kaggle_paths
        )
        loader_diabetes.load_data()
        print("-"*70)

        # Test the heart disease loader
        loader_heart = EnhancedMedicalDataLoader(
            'hospital_b', 'heart_disease', use_kaggle=True, kaggle_paths=custom_kaggle_paths
        )
        loader_heart.load_data()
        print("-"*70)

        print("\n✓ Kaggle datasets with custom paths are working!")
        
    except FileNotFoundError as e:
        print(f"\n  A dataset was not found. Please check your paths in `custom_kaggle_paths`.")
        print(f"  Details: {e}")