"""Pydantic models for API requests."""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from enum import Enum


class OptimizationAlgorithm(str, Enum):
    """Available optimization algorithms."""
    PSO = "PSO"
    GA = "GA"
    DE = "DE"
    SA = "SA"
    WOA = "WOA"


class OptimizationRequestModel(BaseModel):
    """Request model for KNN optimization."""
    
    target_column: str = Field(..., description="Name of the target column in the dataset")
    algorithm: OptimizationAlgorithm = Field(
        default=OptimizationAlgorithm.PSO,
        description="Meta-heuristic algorithm to use for optimization"
    )
    max_iterations: int = Field(
        default=50,
        ge=1,
        le=10000,
        description="Maximum number of iterations for optimization"
    )
    population_size: int = Field(
        default=20,
        ge=5,
        le=500,
        description="Population size for the optimization algorithm"
    )
    test_size: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="Proportion of dataset to use for testing"
    )
    random_state: int = Field(
        default=42,
        description="Random state for reproducibility"
    )
    cv_folds: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Number of cross-validation folds"
    )
    additional_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional parameters for the optimization algorithm"
    )
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "target_column": "outcome",
                "algorithm": "PSO",
                "max_iterations": 100,
                "population_size": 30,
                "test_size": 0.2,
                "random_state": 42,
                "cv_folds": 5,
                "additional_params": {}
            }
        } 