"""
Level 2: Advanced Federated Learning Client

Improvements over Level 1:
✅ Uses real CSV datasets (your diabetes.csv, heart.csv, cardio_train.csv)
✅ Advanced model architectures with batch normalization
✅ Comprehensive evaluation metrics
✅ Support for FedProx (proximal term)
✅ Better training with callbacks
✅ Detailed logging and monitoring
"""

import flwr as fl
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Level 2 modules
from client.real_data_loader import RealMedicalDataLoader
from client.advanced_model import get_advanced_model, AdvancedMedicalModels
from utils.metrics import MedicalMetricsEvaluator


class AdvancedMedicalClient(fl.client.NumPyClient):
    """
    Advanced FL client with enhanced features for Level 2.
    """
    
    def __init__(
        self,
        hospital_id: str,
        disease_type: str,
        local_epochs: int = 5,
        batch_size: int = 32,
        use_callbacks: bool = True,
        proximal_mu: float = 0.0,
        verbose: bool = True
    ):
        """
        Initialize advanced FL client.
        
        Args:
            hospital_id: Hospital identifier
            disease_type: Disease type ('diabetes', 'heart_disease', 'cardiovascular', 'cancer')
            local_epochs: Training epochs per round
            batch_size: Batch size for training
            use_callbacks: Use early stopping and LR scheduling
            proximal_mu: Proximal term coefficient (0 for FedAvg, >0 for FedProx)
            verbose: Print detailed information
        """
        self.hospital_id = hospital_id
        self.disease_type = disease_type
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.use_callbacks = use_callbacks
        self.proximal_mu = proximal_mu
        self.verbose = verbose
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"🏥 Advanced FL Client - Level 2")
            print(f"Hospital: {hospital_id}")
            print(f"Disease: {disease_type}")
            print(f"Strategy: {'FedProx' if proximal_mu > 0 else 'FedAvg'}")
            print(f"{'='*70}")
        
        # Load real data
        self.data_loader = RealMedicalDataLoader(
            hospital_id, 
            disease_type,
            verbose=verbose
        )
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_loader.load_data()
        
        # Initialize metrics evaluator
        self.metrics_evaluator = MedicalMetricsEvaluator(
            task_name=f"{disease_type}_{hospital_id}"
        )
        
        # Get dataset statistics
        self.stats = self.data_loader.get_data_statistics(self.X_train, self.y_train)
        
        # Create advanced model
        input_shape = (self.X_train.shape[1],)
        self.model = get_advanced_model(disease_type, input_shape)
        
        # Store global model for FedProx
        self.global_weights = None
        
        if verbose:
            print(f"\n✓ Client initialized successfully!")
            print(f"  Model parameters: {self.model.count_params():,}")
            print(f"  Training samples: {len(self.X_train)}")
            print(f"  Testing samples: {len(self.X_test)}\n")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Get current model parameters."""
        return self.model.get_weights()
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from server."""
        self.model.set_weights(parameters)
        # Store global weights for FedProx
        self.global_weights = [w.copy() for w in parameters]
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model on local data.
        
        Args:
            parameters: Global model weights from server
            config: Training configuration
            
        Returns:
            - Updated weights
            - Number of training examples
            - Training metrics
        """
        # Update local model with global weights
        self.set_parameters(parameters)
        
        # Get configuration
        epochs = config.get("local_epochs", self.local_epochs)
        batch_size = config.get("batch_size", self.batch_size)
        server_round = config.get("server_round", 0)
        proximal_mu = config.get("proximal_mu", self.proximal_mu)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[{self.hospital_id}] Training Round {server_round}")
            print(f"{'='*70}")
            print(f"  Epochs: {epochs}")
            print(f"  Batch size: {batch_size}")
            if proximal_mu > 0:
                print(f"  Proximal μ: {proximal_mu} (FedProx)")
        
        # Prepare callbacks
        callbacks = []
        if self.use_callbacks:
            callbacks = AdvancedMedicalModels.get_callbacks(patience=3)
        
        # Add custom callback for FedProx
        if proximal_mu > 0:
            from tensorflow.keras.callbacks import Callback
            
            class ProximalCallback(Callback):
                def __init__(self, global_weights, mu):
                    super().__init__()
                    self.global_weights = global_weights
                    self.mu = mu
                
                def on_batch_end(self, batch, logs=None):
                    # Add proximal term: (μ/2)||w - w_global||²
                    current_weights = self.model.get_weights()
                    proximal_loss = 0
                    for w, w_global in zip(current_weights, self.global_weights):
                        proximal_loss += self.mu / 2 * np.sum((w - w_global) ** 2)
                    
                    if logs:
                        logs['proximal_loss'] = proximal_loss
            
            callbacks.append(ProximalCallback(self.global_weights, proximal_mu))
        
        # Train the model
        history = self.model.fit(
            self.X_train, 
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1 if self.verbose else 0
        )
        
        # Get training metrics
        final_loss = history.history['loss'][-1]
        final_accuracy = history.history['accuracy'][-1]
        
        if self.verbose:
            print(f"\n[{self.hospital_id}] Training completed!")
            print(f"  Final loss: {final_loss:.4f}")
            print(f"  Final accuracy: {final_accuracy:.4f}")
        
        # Prepare metrics dictionary
        metrics = {
            "loss": float(final_loss),
            "accuracy": float(final_accuracy),
            "hospital_id": self.hospital_id,
            "round": server_round
        }
        
        # Add additional metrics if available
        for metric_name in ['precision', 'recall', 'auc']:
            if metric_name in history.history:
                metrics[metric_name] = float(history.history[metric_name][-1])
        
        return self.model.get_weights(), len(self.X_train), metrics
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local test data with comprehensive metrics.
        
        Args:
            parameters: Global model weights
            config: Evaluation configuration
            
        Returns:
            - Loss value
            - Number of test examples
            - Comprehensive evaluation metrics
        """
        # Update model with global weights
        self.set_parameters(parameters)
        
        server_round = config.get("server_round", 0)
        
        if self.verbose:
            print(f"\n[{self.hospital_id}] Evaluating Round {server_round}...")
        
        # Standard evaluation
        results = self.model.evaluate(
            self.X_test, 
            self.y_test, 
            verbose=0
        )
        
        loss = results[0]
        basic_accuracy = results[1]
        
        # Get predictions for comprehensive metrics
        y_pred_proba = self.model.predict(self.X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate comprehensive metrics
        comprehensive_metrics = self.metrics_evaluator.evaluate_predictions(
            self.y_test,
            y_pred,
            y_pred_proba
        )
        
        if self.verbose:
            print(f"\n[{self.hospital_id}] Evaluation Results:")
            print(f"  Loss: {loss:.4f}")
            print(f"  Accuracy: {comprehensive_metrics['accuracy']:.4f}")
            print(f"  Precision: {comprehensive_metrics['precision']:.4f}")
            print(f"  Recall: {comprehensive_metrics['recall']:.4f}")
            print(f"  F1-Score: {comprehensive_metrics['f1_score']:.4f}")
            if comprehensive_metrics.get('roc_auc'):
                print(f"  ROC-AUC: {comprehensive_metrics['roc_auc']:.4f}")
        
        # Add hospital ID and round to metrics
        comprehensive_metrics['hospital_id'] = self.hospital_id
        comprehensive_metrics['round'] = server_round
        comprehensive_metrics['loss'] = float(loss)
        
        # Save metrics
        results_dir = Path("results") / self.disease_type / self.hospital_id
        self.metrics_evaluator.save_metrics(
            comprehensive_metrics,
            results_dir / f"round_{server_round}_metrics.json",
            round_number=server_round
        )
        
        return loss, len(self.X_test), comprehensive_metrics
    
    def get_detailed_report(self):
        """Generate and print detailed evaluation report."""
        y_pred_proba = self.model.predict(self.X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = self.metrics_evaluator.evaluate_predictions(
            self.y_test, y_pred, y_pred_proba
        )
        
        self.metrics_evaluator.print_metrics(
            metrics,
            title=f"{self.hospital_id.upper()} - {self.disease_type.upper()}"
        )
        
        report = self.metrics_evaluator.get_classification_report(
            self.y_test, y_pred
        )
        print("\nClassification Report:")
        print(report)


def start_advanced_client(
    hospital_id: str,
    disease_type: str,
    server_address: str = "localhost:8080",
    local_epochs: int = 5,
    use_fedprox: bool = False
):
    """
    Start advanced FL client.
    
    Args:
        hospital_id: Hospital identifier
        disease_type: Disease type
        server_address: FL server address
        local_epochs: Training epochs per round
        use_fedprox: Use FedProx strategy
    """
    print(f"\n{'#'*70}")
    print(f"# ADVANCED FEDERATED LEARNING CLIENT - LEVEL 2")
    print(f"# Hospital: {hospital_id}")
    print(f"# Disease: {disease_type}")
    print(f"# Server: {server_address}")
    print(f"# Strategy: {'FedProx' if use_fedprox else 'FedAvg'}")
    print(f"{'#'*70}\n")
    
    # Create client
    client = AdvancedMedicalClient(
        hospital_id=hospital_id,
        disease_type=disease_type,
        local_epochs=local_epochs,
        proximal_mu=0.01 if use_fedprox else 0.0,
        verbose=True
    )
    
    # Connect to server
    print(f"[{hospital_id}] Connecting to FL server at {server_address}...")
    
    try:
        fl.client.start_client(
            server_address=server_address,
            client=client.to_client()
        )
        
        print(f"\n{'='*70}")
        print(f"✓ Federated Learning Completed Successfully!")
        print(f"{'='*70}\n")
        
        # Generate final detailed report
        client.get_detailed_report()
        
    except Exception as e:
        print(f"\n❌ Error during federated learning: {e}")
        import traceback
        traceback.print_exc()


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Advanced Federated Learning Client - Level 2"
    )
    parser.add_argument(
        "--hospital",
        type=str,
        required=True,
        help="Hospital ID (e.g., hospital_a, hospital_b, clinic_1)"
    )
    parser.add_argument(
        "--disease",
        type=str,
        default="diabetes",
        choices=['diabetes', 'heart_disease', 'cardiovascular', 'cancer'],
        help="Disease type to predict"
    )
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:8080",
        help="FL server address (host:port)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Local training epochs per round"
    )
    parser.add_argument(
        "--fedprox",
        action="store_true",
        help="Use FedProx strategy (adds proximal term)"
    )
    
    args = parser.parse_args()
    
    start_advanced_client(
        hospital_id=args.hospital,
        disease_type=args.disease,
        server_address=args.server,
        local_epochs=args.epochs,
        use_fedprox=args.fedprox
    )