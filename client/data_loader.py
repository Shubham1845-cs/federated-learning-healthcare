import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

class MedicalDataLoader:
    """
    Generates dummy medical data for testing the FL client.
    """
    def __init__(self, hospital_id: str, disease_type: str):
        self.hospital_id = hospital_id
        self.disease_type = disease_type

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generates random data and splits it into train/test sets."""
        print(f"[{self.hospital_id}] Loading DUMMY data for {self.disease_type}...")
        # Generate 1000 samples with 20 features each
        X = np.random.rand(1000, 20)
        # Generate random binary labels (0 or 1)
        y = np.random.randint(0, 2, size=1000)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def get_data_statistics(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Calculates statistics about the dummy dataset."""
        num_samples = len(X_train)
        positive_cases = np.sum(y_train)
        negative_cases = num_samples - positive_cases
        class_balance = positive_cases / num_samples if num_samples > 0 else 0
        
        return {
            "num_samples": num_samples,
            "positive_cases": positive_cases,
            "negative_cases": negative_cases,
            "class_balance": class_balance
        }
    