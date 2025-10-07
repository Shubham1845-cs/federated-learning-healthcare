"""
Level 1: Basic Federated Learning Server
This is the simplest FL server that coordinates training across multiple clients.

CONCEPT: This server is like a "teacher" that:
1. Waits for students (clients) to join the class
2. Distributes homework (global model) to all students
3. Collects completed homework (model updates) from students
4. Combines all homework into a master solution (aggregated model)
5. Repeats the process for multiple rounds
"""

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from typing import List, Tuple, Dict
import argparse

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate evaluation metrics from multiple clients using weighted average.
    
    CONCEPT: If Hospital A has 100 patients and Hospital B has 50 patients,
    Hospital A's results should count twice as much when calculating averages.
    
    Args:
        metrics: List of (num_examples, metrics_dict) from each client
        
    Returns:
        Aggregated metrics dictionary
    """
    # Calculate total number of examples
    total_examples = sum([num_examples for num_examples, _ in metrics])
    
    # Initialize aggregated metrics
    aggregated = {}
    
    # Get all metric keys from first client
    if metrics:
        metric_keys = metrics[0][1].keys()
        
        for key in metric_keys:
            # Skip non-numeric metrics
            if key in ['hospital_id']:
                continue
            
            # Weighted sum
            weighted_sum = sum([
                num_examples * m[key] 
                for num_examples, m in metrics 
                if key in m and isinstance(m[key], (int, float))
            ])
            
            # Weighted average
            aggregated[f"avg_{key}"] = weighted_sum / total_examples
    
    return aggregated


class BasicFederatedServer:
    """
    Basic Federated Learning Server for medical diagnosis.
    
    This server:
    - Initializes a global model (by requesting from a client)
    - Coordinates training rounds across clients
    - Aggregates model updates from clients
    - Evaluates global model performance
    """
    
    def __init__(
        self,
        disease_type: str = "cancer",
        num_rounds: int = 10,
        min_clients: int = 2,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        server_address: str = "0.0.0.0:8080"
    ):
        """
        Initialize the FL Server.
        
        Args:
            disease_type: The type of disease being modeled.
            num_rounds: Total number of federated learning rounds.
            min_clients: Minimum number of clients required for training/evaluation.
            fraction_fit: Fraction of clients to use for training in each round.
            fraction_evaluate: Fraction of clients to use for evaluation in each round.
            server_address: The address where the server will listen for clients.
        """
        self.disease_type = disease_type
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.server_address = server_address
        
        # Define the Flower strategy (Federated Averaging)
        self.strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_clients,
            min_evaluate_clients=min_clients,
            min_available_clients=min_clients,
            evaluate_metrics_aggregation_fn=weighted_average, # Custom function for metrics
        )

    def start(self):
        """
        Start the federated learning server.
        """
        print(f"\n{'#'*70}")
        print(f"# FEDERATED LEARNING SERVER - LEVEL 1")
        print(f"# Disease Task: {self.disease_type}")
        print(f"# Number of Rounds: {self.num_rounds}")
        print(f"# Minimum Clients: {self.min_clients}")
        print(f"# Listening on: {self.server_address}")
        print(f"{'#'*70}\n")
        
        # Start the Flower server
        fl.server.start_server(
            server_address=self.server_address,
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=self.strategy
        )
        print("\nFederated learning completed.")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Medical Server - Level 1")
    parser.add_argument("--disease", type=str, default="cancer",
                        choices=['cancer', 'diabetes', 'heart_disease'],
                        help="Disease type for the FL task")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Number of federated learning rounds")
    parser.add_argument("--min_clients", type=int, default=2,
                        help="Minimum number of clients to start a round")
    parser.add_argument("--address", type=str, default="0.0.0.0:8080",
                        help="Server address and port")
    
    args = parser.parse_args()
    
    # Create and start the server
    server = BasicFederatedServer(
        disease_type=args.disease,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        server_address=args.address
    )
    server.start()