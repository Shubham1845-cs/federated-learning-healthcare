"""
Level 1: Basic Federated Learning Client
This is the simplest FL client that connects to a server and participates in training.

CONCEPT: This client is like a "student" that:
1. Has its own textbook (local data)
2. Studies independently (local training)
3. Shares what it learned (model updates) with the teacher (server)
4. Gets the combined knowledge from all students (global model)
"""

import flwr as fl
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.model import get_model
from client.data_loader import MedicalDataLoader


class BasicMedicalClient(fl.client.NumPyClient):
    """
    A basic federated learning client for medical diagnosis.
    
    This client:
    - Loads local hospital data
    - Trains a model on local data
    - Sends model weights to the server
    - Receives and applies global model weights from server
    """
    
    def __init__(self, hospital_id: str, disease_type: str, local_epochs: int = 5):
        """
        Initialize the FL client.
        
        Args:
            hospital_id: Unique identifier for this hospital
            disease_type: Type of disease to predict
            local_epochs: Number of epochs to train locally per round
        """
        self.hospital_id = hospital_id
        self.disease_type = disease_type
        self.local_epochs = local_epochs
        
        print(f"\n{'='*60}")
        print(f"Initializing FL Client: {hospital_id}")
        print(f"Disease Type: {disease_type}")
        print(f"{'='*60}\n")
        
        # Load local data
        self.data_loader = MedicalDataLoader(hospital_id, disease_type)
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_loader.load_data()
        
        # Get dataset statistics
        self.stats = self.data_loader.get_data_statistics(self.X_train, self.y_train)
        self._print_statistics()
        
        # Create model
        input_shape = (self.X_train.shape[1],)
        self.model = get_model(disease_type, input_shape)
        
        print(f"\n[{self.hospital_id}] Model initialized successfully!")
        print(f"Model parameters: {self.model.count_params():,}\n")
    
    def _print_statistics(self):
        """Print local dataset statistics."""
        print(f"[{self.hospital_id}] Local Dataset Statistics:")
        print(f"  Total training samples: {self.stats['num_samples']}")
        print(f"  Positive cases: {self.stats['positive_cases']}")
        print(f"  Negative cases: {self.stats['negative_cases']}")
        print(f"  Class balance: {self.stats['class_balance']:.2%}")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Get current model parameters (weights).
        
        This is called by the server to retrieve the client's model weights.
        
        Args:
            config: Configuration dictionary from server
            
        Returns:
            List of numpy arrays (model weights)
        """
        print(f"[{self.hospital_id}] Server requested model parameters")
        return self.model.get_weights()
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters (weights) received from server.
        
        This updates the local model with the global model weights.
        
        Args:
            parameters: List of numpy arrays (global model weights)
        """
        print(f"[{self.hospital_id}] Received global model from server")
        self.model.set_weights(parameters)
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model on local data.
        
        CONCEPT: This is like a student studying independently with their own textbook.
        
        Args:
            parameters: Global model weights from server
            config: Training configuration from server
            
        Returns:
            - Updated model weights (what the student learned)
            - Number of training examples used
            - Training metrics (how well the student performed)
        """
        # Update local model with global weights
        self.set_parameters(parameters)
        
        # Get training configuration
        epochs = config.get("local_epochs", self.local_epochs)
        batch_size = config.get("batch_size", 32)
        current_round = config.get("server_round", 0)
        
        print(f"\n{'='*60}")
        print(f"[{self.hospital_id}] Starting Training - Round {current_round}")
        print(f"{'='*60}")
        print(f"  Local epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Training samples: {len(self.X_train)}")
        
        # Train the model on local data
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        
        # Get final training metrics
        final_loss = history.history['loss'][-1]
        final_accuracy = history.history['accuracy'][-1]
        
        print(f"\n[{self.hospital_id}] Training completed!")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Final accuracy: {final_accuracy:.4f}")
        
        # Return updated weights and metrics
        metrics = {
            "loss": final_loss,
            "accuracy": final_accuracy,
            "hospital_id": self.hospital_id
        }
        
        return self.model.get_weights(), len(self.X_train), metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on local test data.
        
        CONCEPT: This is like a student taking a test to see how well they learned.
        
        Args:
            parameters: Global model weights from server
            config: Evaluation configuration
            
        Returns:
            - Loss value
            - Number of test examples
            - Evaluation metrics (test scores)
        """
        # Update model with global weights
        self.set_parameters(parameters)
        
        print(f"\n[{self.hospital_id}] Evaluating model on local test data...")
        
        # Evaluate on local test set
        results = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        loss = results[0]
        accuracy = results[1]
        
        print(f"[{self.hospital_id}] Evaluation Results:")
        print(f"  Test loss: {loss:.4f}")
        print(f"  Test accuracy: {accuracy:.4f}")
        
        metrics = {
            "accuracy": accuracy,
            "hospital_id": self.hospital_id,
            "test_samples": len(self.X_test)
        }
        
        return loss, len(self.X_test), metrics


def start_client(hospital_id: str, disease_type: str, server_address: str = "localhost:8080"):
    """
    Start the federated learning client.
    
    Args:
        hospital_id: Unique identifier for this hospital
        disease_type: Type of disease ('cancer', 'diabetes', 'heart_disease')
        server_address: Address of the FL server (host:port)
    """
    print(f"\n{'#'*70}")
    print(f"# FEDERATED LEARNING CLIENT - LEVEL 1")
    print(f"# Hospital: {hospital_id}")
    print(f"# Disease: {disease_type}")
    print(f"# Server: {server_address}")
    print(f"{'#'*70}\n")
    
    # Create client instance
    client = BasicMedicalClient(
        hospital_id=hospital_id,
        disease_type=disease_type,
        local_epochs=5
    )
    
    # Connect to server and start federated learning
    print(f"\n[{hospital_id}] Connecting to FL server at {server_address}...")
    
    try:
        fl.client.start_client(
            server_address=server_address,
            client=client.to_client()
        )
        print(f"\n[{hospital_id}] Federated learning completed successfully! ✓")
    except Exception as e:
        print(f"\n[{hospital_id}] Error during federated learning: {e}")


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Medical Client - Level 1")
    parser.add_argument("--hospital", type=str, required=True,
                       help="Hospital ID (e.g., hospital_a, hospital_b)")
    parser.add_argument("--disease", type=str, default="cancer",
                       choices=['cancer', 'diabetes', 'heart_disease'],
                       help="Disease type to predict")
    parser.add_argument("--server", type=str, default="localhost:8080",
                       help="FL server address (host:port)")
    
    args = parser.parse_args()
    
    start_client(
        hospital_id=args.hospital,
        disease_type=args.disease,
        server_address=args.server
    )