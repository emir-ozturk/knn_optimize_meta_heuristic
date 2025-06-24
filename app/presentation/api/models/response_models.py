"""Pydantic models for API responses."""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class PerformanceMetrics(BaseModel):
    """Performance metrics model."""
    accuracy: float = Field(..., description="Model accuracy")
    f1_score: float = Field(..., description="F1 score")
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall score")


class DatasetInfo(BaseModel):
    """Dataset information model."""
    shape: List[int] = Field(..., description="Dataset shape (rows, columns)")
    n_features: int = Field(..., description="Number of features")
    n_samples: int = Field(..., description="Number of samples")
    target_column: str = Field(..., description="Target column name")
    unique_classes: int = Field(..., description="Number of unique classes")


class OptimizationHistoryItem(BaseModel):
    """Single optimization history item."""
    parameters: Dict[str, Any] = Field(..., description="KNN parameters used")
    fitness: float = Field(..., description="Fitness score achieved")
    std_score: float = Field(..., description="Standard deviation of CV scores")


class OptimizationResponseModel(BaseModel):
    """Response model for KNN optimization results."""
    
    best_parameters: Dict[str, Any] = Field(..., description="Best KNN parameters found")
    performance_metrics: PerformanceMetrics = Field(..., description="Performance metrics")
    optimization_history: List[OptimizationHistoryItem] = Field(
        ..., description="History of optimization process"
    )
    execution_time: float = Field(..., description="Total execution time in seconds")
    feature_names: List[str] = Field(..., description="Names of features used")
    dataset_info: DatasetInfo = Field(..., description="Information about the dataset")
    algorithm_used: str = Field(..., description="Optimization algorithm used")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "best_parameters": {
                    "n_neighbors": 5,
                    "weights": "uniform",
                    "algorithm": "auto",
                    "metric": "euclidean"
                },
                "performance_metrics": {
                    "accuracy": 0.95,
                    "f1_score": 0.94,
                    "precision": 0.93,
                    "recall": 0.96
                },
                "optimization_history": [
                    {
                        "parameters": {"n_neighbors": 3, "weights": "uniform"},
                        "fitness": 0.92,
                        "std_score": 0.05
                    }
                ],
                "execution_time": 45.2,
                "feature_names": ["feature1", "feature2", "feature3"],
                "dataset_info": {
                    "shape": [100, 4],
                    "n_features": 3,
                    "n_samples": 100,
                    "target_column": "outcome",
                    "unique_classes": 2
                },
                "algorithm_used": "PSO"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    status_code: int = Field(..., description="HTTP status code") 