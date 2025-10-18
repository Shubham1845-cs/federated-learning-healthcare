"""
Level 2: Advanced Federated Learning Server

Improvements over Level 1:
✅ Multiple FL strategies (FedAvg, FedProx, FedAvgM, Adaptive)
✅ Comprehensive metrics aggregation
✅ Progress tracking and logging
✅ Model checkpointing
✅ Results visualization
"""

import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters
from typing import List, Tuple, Dict, Optional
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.advanced_model import get_advanced_model
from server.strategies import get_strategy
from utils.metrics import FederatedMetricsAggregator


def weighted_average_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics from clients using weighted average.
    
    Args:
        metrics: List of (num_examples, metrics_dict) from clients
        
    Returns:
        Aggregated metrics
    """
    if not metrics:
        return {}
    
    total_examples = sum([num for num, _ in metrics])
    
    # Initialize aggregated metrics
    aggregated = {}
    
    # Get all metric keys
    all_keys = set()
    for _, m in metrics:
        all_keys.update(m.keys())
    
    # Aggregate each metric
    for key in all_keys:
        if key in ['hospital_id', 'round']:
            continue
        
        weighted_sum = sum([
            num * m.get(key, 0)
            for num, m in metrics
            if isinstance(m.get(key), (int, float))
        ])
        
        if total_examples > 0:
            aggregated[key] = weighted_sum / total_examples
    
    return aggregated


class AdvancedFederatedServer:
    """
    Advanced FL server with enhanced features for Level 2.
    """
    
    def __init__(
        self,
        disease_type: str = "diabetes",
        strategy_name: str = "fedavg",
        num_rounds: int = 10,
        min_clients: int = 2,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        server_address: str = "localhost:8080",
        save_checkpoints: bool = True,
        **strategy_kwargs
    ):
        """
        Initialize advanced FL server.
        
        Args:
            disease_type: Disease to predict
            strategy_name: 'fedavg', 'fedprox', 'fedavgm', 'adaptive'
            num_rounds: Number of FL rounds
            min_clients: Minimum clients required
            fraction_fit: Fraction of clients for training
            fraction_evaluate: Fraction of clients for evaluation
            server_address: Server address
            save_checkpoints: Save model checkpoints
            **strategy_kwargs: Strategy-specific parameters
        """
        self.disease_type = disease_type
        self.strategy_name = strategy_name
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.server_address = server_address
        self.save_checkpoints = save_checkpoints
        self.strategy_kwargs = strategy_kwargs
        
        print(f"\n{'='*70}")
        print(f"🚀 ADVANCED FEDERATED LEARNING SERVER - LEVEL 2")
        print(f"{'='*70}")
        print(f"Disease: {disease_type}")
        print(f"Strategy: {strategy_name.upper()}")
        print(f"Rounds: {num_rounds}")
        print(f"Min Clients: {min_clients}")
        print(f"Server: {server_address}")
        print(f"{'='*70}\n")
        
        # Create results directory
        self.results_dir = Path("results") / disease_type / strategy_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics tracker
        self.metrics_aggregator = FederatedMetricsAggregator()
        self.round_metrics = []
        
        # Create initial model
        self.global_model = self._create_initial_model()
        self.initial_parameters = ndarrays_to_parameters(
            self.global_model.get_weights()
        )
        
        print(f"✓ Server initialized successfully!\n")
    
    def _create_initial_model(self):
        """Create initial global model."""
        print(f"Creating initial global model for {self.disease_type}...")
        
        # Determine input shape based on disease type
        input_shapes = {
            'diabetes': (8,),
            'heart_disease': (13,),
            'cardiovascular': (11,),
            'cancer': (30,)
        }
        
        input_shape = input_shapes.get(self.disease_type, (10,))
        model = get_advanced_model(self.disease_type, input_shape)
        
        print(f"✓ Model created with {model.count_params():,} parameters\n")
        return model
    
    def _create_strategy(self):
        """Create FL strategy."""
        strategy = get_strategy(
            strategy_name=self.strategy_name,
            initial_parameters=self.initial_parameters,
            min_clients=self.min_clients,
            **self.strategy_kwargs
        )
        
        # Add metric aggregation functions
        strategy.evaluate_metrics_aggregation_fn = weighted_average_metrics
        strategy.fit_metrics_aggregation_fn = weighted_average_metrics
        
        # Add config functions
        strategy.on_fit_config_fn = self._get_fit_config
        strategy.on_evaluate_config_fn = self._get_evaluate_config
        
        print(f"\n✓ Strategy configured: {self.strategy_name.upper()}")
        print(f"  Training: {int(self.fraction_fit * 100)}% of clients")
        print(f"  Evaluation: {int(self.fraction_evaluate * 100)}% of clients\n")
        
        return strategy
    
    def _get_fit_config(self, server_round: int) -> Dict:
        """Configuration for training."""
        config = {
            "server_round": server_round,
            "local_epochs": 5,
            "batch_size": 32,
        }
        
        # Add strategy-specific configs
        if self.strategy_name == 'fedprox':
            config["proximal_mu"] = self.strategy_kwargs.get('proximal_mu', 0.01)
        
        return config
    
    def _get_evaluate_config(self, server_round: int) -> Dict:
        """Configuration for evaluation."""
        return {
            "server_round": server_round,
        }
    
    def _save_checkpoint(self, server_round: int):
        """Save model checkpoint."""
        if not self.save_checkpoints:
            return
        
        checkpoint_path = self.results_dir / f"model_round_{server_round}.h5"
        self.global_model.save(checkpoint_path)
        print(f"  ✓ Checkpoint saved: {checkpoint_path}")
    
    def _save_round_metrics(self, server_round: int, metrics: Dict):
        """Save metrics for this round."""
        metrics_path = self.results_dir / f"round_{server_round}_metrics.json"
        
        metrics_with_meta = {
            'round': server_round,
            'timestamp': datetime.now().isoformat(),
            'disease_type': self.disease_type,
            'strategy': self.strategy_name,
            'metrics': metrics
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_with_meta, f, indent=2)
        
        self.round_metrics.append(metrics_with_meta)
    
    def _save_final_results(self, history):
        """Save final training results."""
        results = {
            'disease_type': self.disease_type,
            'strategy': self.strategy_name,
            'num_rounds': self.num_rounds,
            'min_clients': self.min_clients,
            'timestamp': datetime.now().isoformat(),
            'history': {
                'losses_distributed': [(r, float(l)) for r, l in history.losses_distributed],
                'losses_centralized': [(r, float(l)) for r, l in history.losses_centralized],
                'metrics_distributed': {
                    k: [(r, float(v)) for r, v in vals]
                    for k, vals in history.metrics_distributed.items()
                },
                'metrics_centralized': {
                    k: [(r, float(v)) for r, v in vals]
                    for k, vals in history.metrics_centralized.items()
                }
            },
            'round_metrics': self.round_metrics
        }
        
        results_path = self.results_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Final results saved: {results_path}")
    
    def _print_round_summary(self, server_round: int, metrics: Dict):
        """Print summary for current round."""
        print(f"\n{'='*70}")
        print(f"ROUND {server_round} SUMMARY")
        print(f"{'='*70}")
        
        if metrics:
            print(f"  Loss:      {metrics.get('loss', 0):.4f}")
            print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
            print(f"  Precision: {metrics.get('precision', 0):.4f}")
            print(f"  Recall:    {metrics.get('recall', 0):.4f}")
            print(f"  F1-Score:  {metrics.get('f1_score', 0):.4f}")
            if metrics.get('roc_auc'):
                print(f"  ROC-AUC:   {metrics.get('roc_auc', 0):.4f}")
        
        print(f"{'='*70}\n")
    
    def start(self):
        """Start the federated learning server."""
        print(f"{'='*70}")
        print(f"STARTING FEDERATED LEARNING")
        print(f"{'='*70}")
        print(f"Waiting for at least {self.min_clients} clients...")
        print(f"Server listening on: {self.server_address}\n")
        
        # Create strategy
        strategy = self._create_strategy()
        
        # Parse server address
        host, port = self.server_address.split(":")
        
        # Custom callback for tracking
        class MetricsCallback:
            def __init__(self, server_instance):
                self.server = server_instance
            
            def __call__(self, server_round, results, failures):
                """Called after each round."""
                if results:
                    # Aggregate metrics
                    metrics = weighted_average_metrics([
                        (r.num_examples, r.metrics)
                        for _, r in results
                    ])
                    
                    # Print summary
                    self.server._print_round_summary(server_round, metrics)
                    
                    # Save metrics
                    self.server._save_round_metrics(server_round, metrics)
                    
                    # Save checkpoint
                    if server_round % 5 == 0 or server_round == self.server.num_rounds:
                        self.server._save_checkpoint(server_round)
        
        # Start server
        try:
            history = fl.server.start_server(
                server_address=f"{host}:{port}",
                config=fl.server.ServerConfig(num_rounds=self.num_rounds),
                strategy=strategy,
            )
            
            print(f"\n{'='*70}")
            print(f"✓ FEDERATED LEARNING COMPLETED SUCCESSFULLY!")
            print(f"{'='*70}")
            
            # Print final summary
            self._print_final_summary(history)
            
            # Save results
            self._save_final_results(history)
            
            return history
            
        except Exception as e:
            print(f"\n❌ Error starting server: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _print_final_summary(self, history):
        """Print comprehensive final summary."""
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        
        print(f"\n📊 Training Progress:")
        print(f"  Total rounds: {len(history.losses_distributed)}")
        
        if history.losses_distributed:
            initial_loss = history.losses_distributed[0][1]
            final_loss = history.losses_distributed[-1][1]
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            
            print(f"\n  Loss Improvement:")
            print(f"    Initial: {initial_loss:.4f}")
            print(f"    Final:   {final_loss:.4f}")
            print(f"    Change:  {improvement:+.2f}%")
        
        if history.metrics_distributed.get('accuracy'):
            accuracies = history.metrics_distributed['accuracy']
            initial_acc = accuracies[0][1]
            final_acc = accuracies[-1][1]
            improvement = ((final_acc - initial_acc) / initial_acc) * 100
            
            print(f"\n  Accuracy Improvement:")
            print(f"    Initial: {initial_acc:.4f}")
            print(f"    Final:   {final_acc:.4f}")
            print(f"    Change:  {improvement:+.2f}%")
        
        # Print round-by-round progress
        print(f"\n  Round-by-Round Progress:")
        print(f"  {'Round':<8} {'Loss':<10} {'Accuracy':<10} {'F1-Score':<10}")
        print(f"  {'-'*40}")
        
        for i, (round_num, loss) in enumerate(history.losses_distributed):
            acc = history.metrics_distributed.get('accuracy', [])
            f1 = history.metrics_distributed.get('f1_score', [])
            
            acc_val = acc[i][1] if i < len(acc) else 0
            f1_val = f1[i][1] if i < len(f1) else 0
            
            print(f"  {round_num:<8} {loss:<10.4f} {acc_val:<10.4f} {f1_val:<10.4f}")
        
        print("\n" + "="*70)


def start_advanced_server(
    disease_type: str = "diabetes",
    strategy: str = "fedavg",
    num_rounds: int = 10,
    min_clients: int = 2,
    server_address: str = "localhost:8080",
    **strategy_kwargs
):
    """
    Convenience function to start advanced FL server.
    
    Args:
        disease_type: Disease type
        strategy: FL strategy name
        num_rounds: Number of rounds
        min_clients: Minimum clients
        server_address: Server address
        **strategy_kwargs: Strategy-specific parameters
    """
    server = AdvancedFederatedServer(
        disease_type=disease_type,
        strategy_name=strategy,
        num_rounds=num_rounds,
        min_clients=min_clients,
        server_address=server_address,
        **strategy_kwargs
    )
    
    return server.start()


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Advanced Federated Learning Server - Level 2"
    )
    parser.add_argument(
        "--disease",
        type=str,
        default="diabetes",
        choices=['diabetes', 'heart_disease', 'cardiovascular', 'cancer'],
        help="Disease type to predict"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="fedavg",
        choices=['fedavg', 'fedprox', 'fedavgm', 'adaptive'],
        help="FL aggregation strategy"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of federated learning rounds"
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=2,
        help="Minimum number of clients required"
    )
    parser.add_argument(
        "--address",
        type=str,
        default="localhost:8080",
        help="Server address (host:port)"
    )
    parser.add_argument(
        "--proximal-mu",
        type=float,
        default=0.01,
        help="Proximal term for FedProx (only used with --strategy fedprox)"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for FedAvgM (only used with --strategy fedavgm)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'#'*70}")
    print(f"# ADVANCED FEDERATED LEARNING SERVER - LEVEL 2")
    print(f"# Real Medical Datasets + Advanced Strategies")
    print(f"{'#'*70}\n")
    
    # Prepare strategy-specific kwargs
    strategy_kwargs = {}
    if args.strategy == 'fedprox':
        strategy_kwargs['proximal_mu'] = args.proximal_mu
    elif args.strategy == 'fedavgm':
        strategy_kwargs['momentum'] = args.momentum
        strategy_kwargs['server_learning_rate'] = 1.0
    elif args.strategy == 'adaptive':
        strategy_kwargs['performance_weight'] = 0.3
        strategy_kwargs['size_weight'] = 0.7
    
    start_advanced_server(
        disease_type=args.disease,
        strategy=args.strategy,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        server_address=args.address,
        **strategy_kwargs
    )