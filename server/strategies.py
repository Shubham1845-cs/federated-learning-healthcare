"""
Level 2: Advanced Federated Learning Strategies

Implements:
1. FedAvg (Federated Averaging) - Standard baseline
2. FedProx - Handles heterogeneous data better
3. FedAvgM (with Momentum) - Faster convergence
4. Custom weighted aggregation

These strategies improve on Level 1's basic FedAvg.
"""

import flwr as fl
from flwr.server.strategy import Strategy, FedAvg
from flwr.common import (
    Metrics,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from functools import reduce


class FedProx(FedAvg):
    """
    Federated Proximal (FedProx) Strategy.
    
    FedProx is an extension of FedAvg that adds a proximal term to handle:
    - Non-IID data (different distributions across hospitals)
    - System heterogeneity (different computational capabilities)
    - Partial participation (not all clients available every round)
    
    Paper: "Federated Optimization in Heterogeneous Networks" (MLSys 2020)
    
    The key difference from FedAvg:
    - Clients minimize: Loss(w) + (μ/2)||w - w_global||²
    - The proximal term keeps local models close to global model
    """
    
    def __init__(
        self,
        proximal_mu: float = 0.01,
        **kwargs
    ):
        """
        Initialize FedProx strategy.
        
        Args:
            proximal_mu: Proximal term coefficient (higher = stay closer to global)
            **kwargs: Arguments for FedAvg
        """
        super().__init__(**kwargs)
        self.proximal_mu = proximal_mu
        print(f"✓ FedProx Strategy initialized (μ={proximal_mu})")
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of training with proximal term."""
        config = super().configure_fit(server_round, parameters, client_manager)
        
        # Add proximal term to config
        for _, fit_config in config:
            fit_config["proximal_mu"] = self.proximal_mu
        
        return config


class FedAvgWithMomentum(FedAvg):
    """
    Federated Averaging with Server-side Momentum.
    
    This strategy adds momentum to the aggregation step:
    - Helps overcome local minima
    - Faster convergence
    - Better generalization
    
    Update rule:
    v_t = β * v_{t-1} + Δw_t
    w_{t+1} = w_t - η * v_t
    """
    
    def __init__(
        self,
        momentum: float = 0.9,
        server_learning_rate: float = 1.0,
        **kwargs
    ):
        """
        Initialize FedAvg with Momentum.
        
        Args:
            momentum: Momentum coefficient (0.9 typical)
            server_learning_rate: Server-side learning rate
            **kwargs: Arguments for FedAvg
        """
        super().__init__(**kwargs)
        self.momentum = momentum
        self.server_lr = server_learning_rate
        self.velocity = None
        print(f"✓ FedAvgM Strategy initialized (momentum={momentum}, lr={server_learning_rate})")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ):
        """Aggregate with momentum."""
        # Get standard aggregated parameters
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is None:
            return None, {}
        
        # Convert to numpy arrays
        aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
        
        # Initialize velocity on first round
        if self.velocity is None:
            self.velocity = [np.zeros_like(arr) for arr in aggregated_ndarrays]
        
        # Apply momentum
        updated_arrays = []
        for i, (param, vel) in enumerate(zip(aggregated_ndarrays, self.velocity)):
            # Update velocity: v = β*v + (1-β)*Δw
            self.velocity[i] = self.momentum * vel + (1 - self.momentum) * param
            # Update parameters with server learning rate
            updated_param = param - self.server_lr * self.velocity[i]
            updated_arrays.append(updated_param)
        
        return ndarrays_to_parameters(updated_arrays), aggregated_metrics


class AdaptiveWeightedAggregation(FedAvg):
    """
    Adaptive Weighted Aggregation Strategy.
    
    Instead of simple averaging, this strategy weighs clients based on:
    - Dataset size (more data = more weight)
    - Model performance (better accuracy = more weight)
    - Data quality (optional)
    
    This helps when hospitals have very different data quality or sizes.
    """
    
    def __init__(
        self,
        performance_weight: float = 0.3,
        size_weight: float = 0.7,
        **kwargs
    ):
        """
        Initialize adaptive weighted aggregation.
        
        Args:
            performance_weight: Weight for performance-based weighting (0-1)
            size_weight: Weight for size-based weighting (0-1)
            **kwargs: Arguments for FedAvg
        """
        super().__init__(**kwargs)
        self.perf_weight = performance_weight
        self.size_weight = size_weight
        print(f"✓ Adaptive Weighted Aggregation initialized")
        print(f"  Performance weight: {performance_weight}")
        print(f"  Size weight: {size_weight}")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ):
        """Aggregate using adaptive weights."""
        if not results:
            return None, {}
        
        # Extract weights, number of examples, and metrics
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, fit_res.metrics)
            for _, fit_res in results
        ]
        
        # Calculate adaptive weights
        total_examples = sum([num_examples for _, num_examples, _ in weights_results])
        
        adaptive_weights = []
        for _, num_examples, metrics in weights_results:
            # Size-based weight
            size_w = num_examples / total_examples
            
            # Performance-based weight (use accuracy if available)
            perf_w = metrics.get('accuracy', 0.5) if metrics else 0.5
            
            # Combined weight
            combined_w = (self.size_weight * size_w + self.perf_weight * perf_w)
            adaptive_weights.append(combined_w)
        
        # Normalize weights
        weight_sum = sum(adaptive_weights)
        adaptive_weights = [w / weight_sum for w in adaptive_weights]
        
        # Aggregate using adaptive weights
        weights_prime = [
            [layer * w for layer in weights]
            for weights, w in zip([w for w, _, _ in weights_results], adaptive_weights)
        ]
        
        weights_aggregated = [
            reduce(np.add, layer_updates)
            for layer_updates in zip(*weights_prime)
        ]
        
        # Aggregate metrics
        metrics_aggregated = {}
        if any(metrics for _, _, metrics in weights_results if metrics):
            metrics_aggregated = self._aggregate_metrics(
                [(num_examples, metrics) for _, num_examples, metrics in weights_results]
            )
        
        return ndarrays_to_parameters(weights_aggregated), metrics_aggregated
    
    def _aggregate_metrics(self, metrics_list):
        """Aggregate metrics using weighted average."""
        total_examples = sum([num for num, _ in metrics_list])
        aggregated = {}
        
        # Get all metric keys
        all_keys = set()
        for _, metrics in metrics_list:
            if metrics:
                all_keys.update(metrics.keys())
        
        # Aggregate each metric
        for key in all_keys:
            if key in ['hospital_id']:
                continue
            
            weighted_sum = sum([
                num * metrics.get(key, 0)
                for num, metrics in metrics_list
                if metrics and isinstance(metrics.get(key), (int, float))
            ])
            
            aggregated[key] = weighted_sum / total_examples
        
        return aggregated


def get_strategy(
    strategy_name: str,
    initial_parameters,
    min_clients: int = 2,
    **kwargs
):
    """
    Factory function to get FL strategy.
    
    Args:
        strategy_name: 'fedavg', 'fedprox', 'fedavgm', 'adaptive'
        initial_parameters: Initial model parameters
        min_clients: Minimum number of clients
        **kwargs: Strategy-specific arguments
        
    Returns:
        Flower Strategy instance
    """
    strategy_name = strategy_name.lower()
    
    # Common configuration
    common_config = {
        'fraction_fit': 1.0,
        'fraction_evaluate': 1.0,
        'min_fit_clients': min_clients,
        'min_evaluate_clients': min_clients,
        'min_available_clients': min_clients,
        'initial_parameters': initial_parameters,
    }
    
    if strategy_name == 'fedavg':
        print("📊 Using Strategy: FedAvg (Standard Federated Averaging)")
        return FedAvg(**common_config)
    
    elif strategy_name == 'fedprox':
        proximal_mu = kwargs.get('proximal_mu', 0.01)
        print(f"📊 Using Strategy: FedProx (μ={proximal_mu})")
        return FedProx(proximal_mu=proximal_mu, **common_config)
    
    elif strategy_name == 'fedavgm':
        momentum = kwargs.get('momentum', 0.9)
        server_lr = kwargs.get('server_learning_rate', 1.0)
        print(f"📊 Using Strategy: FedAvgM (momentum={momentum}, lr={server_lr})")
        return FedAvgWithMomentum(
            momentum=momentum,
            server_learning_rate=server_lr,
            **common_config
        )
    
    elif strategy_name == 'adaptive':
        perf_weight = kwargs.get('performance_weight', 0.3)
        size_weight = kwargs.get('size_weight', 0.7)
        print(f"📊 Using Strategy: Adaptive Weighted Aggregation")
        return AdaptiveWeightedAggregation(
            performance_weight=perf_weight,
            size_weight=size_weight,
            **common_config
        )
    
    else:
        raise ValueError(
            f"Unknown strategy: {strategy_name}\n"
            f"Available: 'fedavg', 'fedprox', 'fedavgm', 'adaptive'"
        )


# ==================== TESTING ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING ADVANCED FL STRATEGIES - LEVEL 2")
    print("="*70)
    
    # Create dummy initial parameters
    import tensorflow as tf
    dummy_model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(5,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    initial_params = ndarrays_to_parameters(dummy_model.get_weights())
    
    strategies = ['fedavg', 'fedprox', 'fedavgm', 'adaptive']
    
    for strat in strategies:
        print(f"\n{'='*70}")
        print(f"Testing: {strat.upper()}")
        print(f"{'='*70}")
        
        strategy = get_strategy(
            strat,
            initial_params,
            min_clients=2,
            proximal_mu=0.01,
            momentum=0.9,
            performance_weight=0.3
        )
        
        print(f"✓ Strategy created: {strategy.__class__.__name__}")
        print(f"  Min clients: {strategy.min_fit_clients}")
        print(f"  Fraction fit: {strategy.fraction_fit}")
    
    print(f"\n{'='*70}")
    print("ALL STRATEGIES READY!")
    print(f"{'='*70}\n")    