"""Domain entity for optimization results."""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class OptimizationResult:
    """Entity representing optimization results."""
    
    best_parameters: Dict[str, Any]
    best_accuracy: float
    best_f1_score: float
    best_precision: float
    best_recall: float
    optimization_history: List[Dict[str, Any]]
    execution_time: float
    feature_names: List[str]
    dataset_info: Dict[str, Any]
    algorithm_used: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "best_parameters": self.best_parameters,
            "performance_metrics": {
                "accuracy": self.best_accuracy,
                "f1_score": self.best_f1_score,
                "precision": self.best_precision,
                "recall": self.best_recall
            },
            "optimization_history": self.optimization_history,
            "execution_time": self.execution_time,
            "feature_names": self.feature_names,
            "dataset_info": self.dataset_info,
            "algorithm_used": self.algorithm_used
        } 