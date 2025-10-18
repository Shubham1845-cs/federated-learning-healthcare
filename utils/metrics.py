"""
Level 2: Enhanced Evaluation Metrics
Comprehensive evaluation beyond simple accuracy.

Metrics included:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC, PR-AUC
- Sensitivity, Specificity
- Matthews Correlation Coefficient (MCC)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    average_precision_score
)
import json
from datetime import datetime
from pathlib import Path


class MedicalMetricsEvaluator:
    """
    Comprehensive metrics evaluator for medical diagnosis models.
    """
    
    def __init__(self, task_name="medical_diagnosis"):
        """
        Initialize metrics evaluator.
        
        Args:
            task_name: Name of the task for logging
        """
        self.task_name = task_name
        self.history = []
    
    def evaluate_predictions(self, y_true, y_pred, y_pred_proba=None, threshold=0.5):
        """
        Comprehensive evaluation of model predictions.
        
        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted labels or probabilities
            y_pred_proba: Predicted probabilities (if y_pred is labels)
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        # Convert probabilities to binary predictions if needed
        if y_pred_proba is None and y_pred.max() <= 1.0 and y_pred.min() >= 0.0:
            y_pred_proba = y_pred
            y_pred = (y_pred >= threshold).astype(int).flatten()
        else:
            y_pred = y_pred.astype(int).flatten()
        
        y_true = y_true.astype(int).flatten()
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Sensitivity and Specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # ROC-AUC and PR-AUC (if probabilities available)
        roc_auc = None
        pr_auc = None
        if y_pred_proba is not None:
            try:
                y_pred_proba_flat = y_pred_proba.flatten()
                if len(np.unique(y_true)) > 1:  # Need both classes for AUC
                    roc_auc = roc_auc_score(y_true, y_pred_proba_flat)
                    pr_auc = average_precision_score(y_true, y_pred_proba_flat)
            except:
                pass
        
        # Positive and Negative Predictive Value
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value (Precision)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # False Positive Rate and False Negative Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        # Balanced Accuracy (useful for imbalanced datasets)
        balanced_acc = (sensitivity + specificity) / 2
        
        metrics = {
            # Basic metrics
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            
            # Medical-specific metrics
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'ppv': float(ppv),  # Positive Predictive Value
            'npv': float(npv),  # Negative Predictive Value
            
            # Advanced metrics
            'mcc': float(mcc),
            'balanced_accuracy': float(balanced_acc),
            'roc_auc': float(roc_auc) if roc_auc else None,
            'pr_auc': float(pr_auc) if pr_auc else None,
            
            # Confusion matrix components
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            
            # Rates
            'fpr': float(fpr),  # False Positive Rate
            'fnr': float(fnr),  # False Negative Rate
            
            # Sample counts
            'total_samples': len(y_true),
            'positive_samples': int(np.sum(y_true)),
            'negative_samples': int(len(y_true) - np.sum(y_true)),
        }
        
        return metrics
    
    def print_metrics(self, metrics, title="Evaluation Metrics"):
        """
        Pretty print metrics.
        
        Args:
            metrics: Dictionary of metrics
            title: Title for the metrics display
        """
        print(f"\n{'='*70}")
        print(f"{title}")
        print(f"{'='*70}")
        
        print(f"\n📊 CLASSIFICATION METRICS:")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  Precision (PPV):   {metrics['precision']:.4f}")
        print(f"  Recall (Sens.):    {metrics['recall']:.4f}")
        print(f"  F1-Score:          {metrics['f1_score']:.4f}")
        
        print(f"\n🏥 MEDICAL METRICS:")
        print(f"  Sensitivity:       {metrics['sensitivity']:.4f}")
        print(f"  Specificity:       {metrics['specificity']:.4f}")
        print(f"  PPV:               {metrics['ppv']:.4f}")
        print(f"  NPV:               {metrics['npv']:.4f}")
        
        if metrics.get('roc_auc'):
            print(f"\n📈 AREA UNDER CURVE:")
            print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
            if metrics.get('pr_auc'):
                print(f"  PR-AUC:            {metrics['pr_auc']:.4f}")
        
        print(f"\n🔢 CONFUSION MATRIX:")
        print(f"                 Predicted")
        print(f"               Neg      Pos")
        print(f"  Actual Neg   {metrics['true_negatives']:4d}    {metrics['false_positives']:4d}")
        print(f"         Pos   {metrics['false_negatives']:4d}    {metrics['true_positives']:4d}")
        
        print(f"\n📉 ERROR RATES:")
        print(f"  False Positive Rate: {metrics['fpr']:.4f}")
        print(f"  False Negative Rate: {metrics['fnr']:.4f}")
        
        print(f"\n💯 ADDITIONAL:")
        print(f"  MCC:               {metrics['mcc']:.4f}")
        print(f"  Total Samples:     {metrics['total_samples']}")
        
        print(f"{'='*70}\n")
    
    def compare_hospitals(self, hospital_metrics_dict):
        """
        Compare metrics across multiple hospitals.
        
        Args:
            hospital_metrics_dict: Dict of {hospital_id: metrics_dict}
        """
        print(f"\n{'='*70}")
        print(f"COMPARISON ACROSS HOSPITALS")
        print(f"{'='*70}\n")
        
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Print header
        hospitals = list(hospital_metrics_dict.keys())
        print(f"{'Metric':<20}", end="")
        for hosp in hospitals:
            print(f"{hosp:<15}", end="")
        print()
        print("-" * 70)
        
        # Print each metric
        for metric in metric_names:
            print(f"{metric:<20}", end="")
            for hosp in hospitals:
                value = hospital_metrics_dict[hosp].get(metric)
                if value is not None:
                    print(f"{value:<15.4f}", end="")
                else:
                    print(f"{'N/A':<15}", end="")
            print()
        
        print("=" * 70)
    
    def save_metrics(self, metrics, filepath, round_number=None):
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Dictionary of metrics
            filepath: Path to save file
            round_number: Optional round number for FL
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        metrics_with_meta = {
            'timestamp': datetime.now().isoformat(),
            'task_name': self.task_name,
            'round': round_number,
            'metrics': metrics
        }
        
        # Append to history
        self.history.append(metrics_with_meta)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(metrics_with_meta, f, indent=2)
        
        print(f"✓ Metrics saved to: {filepath}")
    
    def save_history(self, filepath):
        """
        Save entire metrics history to JSON file.
        
        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"✓ Metrics history saved to: {filepath}")
    
    def get_classification_report(self, y_true, y_pred, target_names=None):
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names for classes
            
        Returns:
            String report
        """
        if target_names is None:
            target_names = ['No Disease', 'Disease']
        
        report = classification_report(
            y_true.flatten(), 
            y_pred.flatten(), 
            target_names=target_names,
            digits=4
        )
        
        return report


class FederatedMetricsAggregator:
    """
    Aggregates metrics from multiple federated learning clients.
    """
    
    def __init__(self):
        self.client_metrics = {}
    
    def add_client_metrics(self, client_id, metrics):
        """
        Add metrics from a client.
        
        Args:
            client_id: Client identifier
            metrics: Dictionary of metrics
        """
        self.client_metrics[client_id] = metrics
    
    def aggregate_metrics(self):
        """
        Aggregate metrics across all clients using weighted average.
        
        Returns:
            Dictionary of aggregated metrics
        """
        if not self.client_metrics:
            return {}
        
        # Get total samples
        total_samples = sum(
            m['total_samples'] 
            for m in self.client_metrics.values()
        )
        
        # Aggregate each metric
        aggregated = {}
        
        # Metrics to aggregate
        metrics_to_agg = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'sensitivity', 'specificity', 'balanced_accuracy',
            'roc_auc', 'pr_auc', 'mcc', 'ppv', 'npv', 'fpr', 'fnr'
        ]
        
        for metric in metrics_to_agg:
            weighted_sum = 0
            valid_clients = 0
            
            for client_id, client_metrics in self.client_metrics.items():
                value = client_metrics.get(metric)
                if value is not None:
                    weight = client_metrics['total_samples'] / total_samples
                    weighted_sum += value * weight
                    valid_clients += 1
            
            if valid_clients > 0:
                aggregated[metric] = weighted_sum
        
        # Aggregate confusion matrix
        aggregated['true_positives'] = sum(
            m['true_positives'] for m in self.client_metrics.values()
        )
        aggregated['true_negatives'] = sum(
            m['true_negatives'] for m in self.client_metrics.values()
        )
        aggregated['false_positives'] = sum(
            m['false_positives'] for m in self.client_metrics.values()
        )
        aggregated['false_negatives'] = sum(
            m['false_negatives'] for m in self.client_metrics.values()
        )
        
        aggregated['total_samples'] = total_samples
        aggregated['num_clients'] = len(self.client_metrics)
        
        return aggregated
    
    def print_aggregated_metrics(self):
        """Print aggregated metrics."""
        agg_metrics = self.aggregate_metrics()
        
        evaluator = MedicalMetricsEvaluator()
        evaluator.print_metrics(
            agg_metrics, 
            title="AGGREGATED METRICS (ALL HOSPITALS)"
        )
    
    def print_per_client_comparison(self):
        """Print comparison of metrics across clients."""
        evaluator = MedicalMetricsEvaluator()
        evaluator.compare_hospitals(self.client_metrics)


# ==================== TESTING ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING EVALUATION METRICS - LEVEL 2")
    print("="*70)
    
    # Generate dummy predictions for testing
    np.random.seed(42)
    n_samples = 100
    
    # Simulate predictions (70% accuracy)
    y_true = np.random.randint(0, 2, n_samples)
    y_pred_proba = np.random.rand(n_samples)
    # Make predictions somewhat correlated with truth
    y_pred_proba = 0.3 * y_true + 0.7 * y_pred_proba
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Test single evaluation
    print("\n" + "="*70)
    print("TEST 1: Single Model Evaluation")
    print("="*70)
    
    evaluator = MedicalMetricsEvaluator(task_name="diabetes_test")
    metrics = evaluator.evaluate_predictions(y_true, y_pred, y_pred_proba)
    evaluator.print_metrics(metrics, title="Diabetes Model Evaluation")
    
    # Test classification report
    print("\n" + "="*70)
    print("TEST 2: Classification Report")
    print("="*70)
    report = evaluator.get_classification_report(y_true, y_pred)
    print(report)
    
    # Test federated metrics aggregation
    print("\n" + "="*70)
    print("TEST 3: Federated Metrics Aggregation")
    print("="*70)
    
    fed_aggregator = FederatedMetricsAggregator()
    
    # Simulate metrics from 3 hospitals
    for i, hospital in enumerate(['hospital_a', 'hospital_b', 'hospital_c']):
        y_true_h = np.random.randint(0, 2, 50 + i*20)
        y_pred_proba_h = np.random.rand(50 + i*20)
        y_pred_proba_h = 0.4 * y_true_h + 0.6 * y_pred_proba_h
        y_pred_h = (y_pred_proba_h > 0.5).astype(int)
        
        metrics_h = evaluator.evaluate_predictions(y_true_h, y_pred_h, y_pred_proba_h)
        fed_aggregator.add_client_metrics(hospital, metrics_h)
    
    # Print comparison
    fed_aggregator.print_per_client_comparison()
    
    # Print aggregated
    fed_aggregator.print_aggregated_metrics()
    
    # Test saving metrics
    print("\n" + "="*70)
    print("TEST 4: Saving Metrics")
    print("="*70)
    
    evaluator.save_metrics(metrics, "results/test_metrics.json", round_number=1)
    evaluator.save_metrics(metrics, "results/test_metrics_round2.json", round_number=2)
    evaluator.save_history("results/metrics_history.json")
    
    print("\n" + "="*70)
    print("ALL METRICS TESTS PASSED!")
    print("="*70 + "\n")