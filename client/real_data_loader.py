"""
Level 2: Real Medical Data Loader
Loads YOUR actual CSV files: diabetes.csv, heart.csv, cardio_train.csv
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_breast_cancer
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RealMedicalDataLoader:
    """
    Production-ready data loader for real medical datasets.
    Handles missing values, categorical encoding, and data quality checks.
    """
    
    def __init__(self, hospital_id, disease_type, verbose=True):
        """
        Initialize data loader for real medical datasets.
        """
        self.hospital_id = hospital_id
        self.disease_type = disease_type.lower()
        self.scaler = StandardScaler()
        self.verbose = verbose

        # --- THIS IS THE FIX ---
        # Find the project's root directory (the parent of the 'client' folder)
        # and then locate the 'datasets' folder within it.
        project_root = Path(__file__).resolve().parent.parent
        self.base_path = project_root / "datasets"
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🏥 Real Medical Data Loader - Level 2")
            print(f"Hospital: {hospital_id}")
            print(f"Disease: {disease_type}")
            print(f"Looking for data in: {self.base_path}")
            print(f"{'='*70}")
    
    # ==================== DIABETES DATASET ====================
    
    def load_diabetes_real(self):
        csv_path = self.base_path / "diabetes" / "diabetes.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(
                f"\n❌ Diabetes dataset not found!\n"
                f"Expected location: {csv_path.absolute()}"
            )
        
        if self.verbose: print(f"\n📂 Loading: {csv_path.name}")
        df = pd.read_csv(csv_path)
        
        target_col = 'Outcome'
        X = df.drop(target_col, axis=1)
        y = df[target_col].values
        
        # Impute missing values (0s in these columns mean missing)
        cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in cols_to_impute:
            X[col] = X[col].replace(0, np.nan)
        X = X.fillna(X.median()).values
        
        return self._prepare_data(X, y)

    # ==================== HEART DISEASE DATASET ====================

    def load_heart_disease_real(self):
        csv_path = self.base_path / "heart_disease" / "heart.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(
                f"\n❌ Heart disease dataset not found!\n"
                f"Expected location: {csv_path.absolute()}"
            )
            
        if self.verbose: print(f"\n📂 Loading: {csv_path.name}")
        df = pd.read_csv(csv_path)

        target_col = 'target'
        X = df.drop(target_col, axis=1).values
        y = df[target_col].values
        
        return self._prepare_data(X, y)

    # ==================== CARDIOVASCULAR DATASET ====================

    def load_cardiovascular_real(self):
        csv_path = self.base_path / "cardiovascular" / "cardio_train.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(
                f"\n❌ Cardiovascular dataset not found!\n"
                f"Expected location: {csv_path.absolute()}"
            )
            
        if self.verbose: print(f"\n📂 Loading: {csv_path.name}")
        
        # This dataset often uses ';' as a separator
        try:
            df = pd.read_csv(csv_path, sep=';')
            if len(df.columns) <= 1:
                df = pd.read_csv(csv_path) # Fallback to comma
        except Exception:
            df = pd.read_csv(csv_path)

        if 'id' in df.columns: df = df.drop('id', axis=1)

        target_col = 'cardio'
        X = df.drop(target_col, axis=1)
        y = df[target_col].values
        
        # Convert age from days to years
        if 'age' in X.columns and X['age'].mean() > 365:
            X['age'] = (X['age'] / 365.25).round()
        
        X = X.values
        return self._prepare_data(X, y, max_samples_per_hospital=2000)

    # ==================== MAIN LOAD METHOD ====================

    def load_data(self):
        """Main method to load appropriate dataset based on disease type."""
        if self.disease_type == 'diabetes':
            return self.load_diabetes_real()
        elif self.disease_type == 'heart_disease':
            return self.load_heart_disease_real()
        elif self.disease_type == 'cardiovascular':
            return self.load_cardiovascular_real()
        else:
            raise ValueError(f"Unknown disease type: {self.disease_type}")

    # ==================== HELPER METHODS ====================

    def _partition_data(self, X, y, max_samples=None):
        """Partition data to simulate distribution across hospitals."""
        hospital_map = {'hospital_a': 0, 'hospital_b': 1, 'hospital_c': 2}
        idx = hospital_map.get(self.hospital_id, 0)
        num_hospitals = 3
        
        partition_size = len(X) // num_hospitals
        start_idx = idx * partition_size
        end_idx = start_idx + partition_size if idx < num_hospitals - 1 else len(X)
        
        X_local, y_local = X[start_idx:end_idx], y[start_idx:end_idx]
        
        if max_samples and len(X_local) > max_samples:
            indices = np.random.choice(len(X_local), max_samples, replace=False)
            X_local, y_local = X_local[indices], y_local[indices]
            
        return X_local, y_local

    def _prepare_data(self, X, y, max_samples_per_hospital=None):
        """Prepare data: partition, split, and standardize."""
        X_local, y_local = self._partition_data(X, y, max_samples_per_hospital)
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_local, y_local, test_size=0.2, random_state=42, stratify=y_local
            )
        except ValueError: # Fails if a class has only 1 member
            X_train, X_test, y_train, y_test = train_test_split(
                X_local, y_local, test_size=0.2, random_state=42
            )
            
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        if self.verbose:
            print(f"\n[{self.hospital_id}] Final Data Partition:")
            print(f"  Training samples: {len(X_train)}")
            print(f"  Testing samples: {len(X_test)}")
            
        return X_train, X_test, y_train, y_test
        
    def get_data_statistics(self, X_train, y_train):
        """Get comprehensive dataset statistics."""
        return {
            'num_samples': len(X_train),
            'num_features': X_train.shape[1],
            'positive_cases': int(np.sum(y_train)),
            'negative_cases': int(len(y_train) - np.sum(y_train)),
            'class_balance': float(np.mean(y_train)),
        }

# ==================== TESTING ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING REAL MEDICAL DATA LOADER - LEVEL 2")
    print("="*70)
    
    diseases_to_test = ['diabetes', 'heart_disease', 'cardiovascular']
    
    for disease in diseases_to_test:
        try:
            loader = RealMedicalDataLoader('hospital_a', disease)
            X_train, X_test, y_train, y_test = loader.load_data()
            print(f"\n✓ {disease.upper()} dataset loaded successfully for hospital_a!")
        except FileNotFoundError as e:
            print(f"\n❌ {disease.upper()} dataset not found. Please check your `datasets` folder.")
            print(f"   {e}")
        except Exception as e:
            print(f"\n❌ An error occurred loading {disease}: {e}")

    print(f"\n{'='*70}\nDATA LOADER TEST COMPLETE\n{'='*70}\n")