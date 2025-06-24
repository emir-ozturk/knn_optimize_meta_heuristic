"""Domain entity for optimization requests."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd


@dataclass
class OptimizationRequest:
    """Entity representing an optimization request."""
    
    dataset: pd.DataFrame
    target_column: str
    algorithm: str = "PSO"  # Default to Particle Swarm Optimization
    max_iterations: int = 50
    population_size: int = 20
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    additional_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate request parameters."""
        if self.additional_params is None:
            self.additional_params = {}
        
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        
        if self.population_size < 1:
            raise ValueError("population_size must be positive")
        
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2") 