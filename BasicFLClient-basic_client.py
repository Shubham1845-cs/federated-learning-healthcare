"""
Updated FL Client with Kaggle Dataset Support
Simply add --use-kaggle flag to use real datasets!
"""

import flwr as fl
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
import sys
import os
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.model import get_model
# Use enhanced data loader if available
try:
    from client.data_loader_enhanced import EnhancedMedicalDataLoader as DataLoader
    ENHANCED_LOADER = True
except ImportError:
    from client.data_loader import MedicalDataLoader as DataLoader
    ENHANCED_LOADER = False


class MedicalFLClient(fl.client.NumPyClient):
    """
    FL Client with support for both built-in and Kaggle datasets.
    """
    
    def __init__(
        self, 
        hospital_id: str, 
        disease_type: str, 
        local_epochs: int = 3,
        use_kaggle: bool = False
    ):
        self.hospital_id = hospital_id
        self.disease_type = disease_type
        self.local_epochs = local_epochs
        self.use_kaggle = use_kaggle
        
        print(f"\n{'='*60}")
        print(f"Initializing FL Client: {hospital_id}")
        print(f"Disease Type: {disease_type}")
        print(f"Data Source: {'Kaggle CSV' if use_kaggle else 'Built-in'}")
        print(f"{'='*60}\n")
        
        # --- THIS IS THE FIX ---
        # Define the dictionary of custom paths and pass it to the loader
        custom_kaggle_paths = {
            "diabetes": "archive (1)/diabetes.csv",
            "heart_disease": "archive (2)/heart.csv",
            "stroke": "archive (3)/cardio_train.csv"
        }

        # Load local data using the appropriate loader
        if ENHANCED_LOADER:
            self.data_loader = DataLoader(
                hospital_id, 
                disease_type, 
                use_kaggle=use_kaggle, 
                kaggle_paths=custom_kaggle_paths # Pass the dictionary here
            )
        else:
            if use_kaggle:
                print("Warning: Enhanced data loader not found. Ignoring --use-kaggle flag.")
            self.data_loader = DataLoader(hospital_id, disease_type)

        self.X_train, self.X_test, self.y_train, self.y_test = self.data_loader.load_data()
        
        # Get statistics and create model
        self.stats = self.data_loader.get_data_statistics(self.X_train, self.y_train)
        self._print_statistics()
        input_shape = (self.X_train.shape[1],)
        self.model = get_model(disease_type, input_shape)
        print(f"\n[{self.hospital_id}] Model initialized!")
        print(f"Parameters: {self.model.count_params():,}\n")
    
    def _print_statistics(self):
        """Print dataset statistics."""
        print(f"[{self.hospital_id}] Local Dataset Statistics:")
        for key, value in self.stats.items():
            if 'id' not in key and 'source' not in key:
                if 'balance' in key:
                    print(f"  {key.replace('_', ' ').title()}: {value:.2%}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Get current model parameters."""
        return self.model.get_weights()
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from server."""
        self.model.set_weights(parameters)
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model on local data."""
        self.set_parameters(parameters)
        epochs = config.get("local_epochs", self.local_epochs)
        batch_size = config.get("batch_size", 32)
        current_round = config.get("server_round", 0)
        
        print(f"\n{'='*60}")
        print(f"[{self.hospital_id}] Training - Round {current_round}")
        print(f"{'='*60}")
        print(f"  Local epochs: {epochs}, Batch size: {batch_size}, Samples: {len(self.X_train)}")
        
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=2
        )
        
        final_loss = history.history['loss'][-1]
        final_accuracy = history.history['accuracy'][-1]
        
        print(f"\n[{self.hospital_id}] Training completed!")
        print(f"  Final loss: {final_loss:.4f}, Final accuracy: {final_accuracy:.4f}")
        
        metrics = {"loss": final_loss, "accuracy": final_accuracy}
        return self.model.get_weights(), len(self.X_train), metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate model on local test data."""
        self.set_parameters(parameters)
        
        print(f"\n[{self.hospital_id}] Evaluating global model...")
        
        results = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        loss, accuracy = results[0], results[1]
        
        print(f"[{self.hospital_id}] Evaluation Results:")
        print(f"  Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
        
        return loss, len(self.X_test), {"accuracy": accuracy}


def start_client(args):
    """Start the federated learning client."""
    print(f"\n{'#'*70}")
    print(f"# FEDERATED LEARNING CLIENT")
    print(f"# Hospital: {args.hospital}, Disease: {args.disease}, Kaggle: {args.use_kaggle}")
    print(f"# Server: {args.server}")
    print(f"{'#'*70}\n")
    
    client = MedicalFLClient(
        hospital_id=args.hospital,
        disease_type=args.disease,
        local_epochs=args.epochs,
        use_kaggle=args.use_kaggle
    )
    
    print(f"\n[{args.hospital}] Connecting to FL server at {args.server}...")
    try:
        fl.client.start_client(
            server_address=args.server,
            client=client.to_client()
        )
        print(f"\n[{args.hospital}] Federated learning completed! ✓")
    except Exception as e:
        print(f"\n[{args.hospital}] Error during FL: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Medical Client")
    parser.add_argument("--hospital", type=str, required=True, help="Hospital ID")
    parser.add_argument("--disease", type=str, default="cancer",
                        choices=['cancer', 'diabetes', 'heart_disease', 'stroke'],
                        help="Disease type to predict")
    parser.add_argument("--server", type=str, default="localhost:8080",
                        help="FL server address (host:port)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of local epochs per round")
    parser.add_argument("--use-kaggle", action="store_true",
                        help="Use real Kaggle datasets instead of built-in/dummy data")
    
    args = parser.parse_args()
    start_client(args)