"""Repository interface for optimization operations."""

from abc import ABC, abstractmethod
from typing import Protocol

from ..entities.optimization_request import OptimizationRequest
from ..entities.optimization_result import OptimizationResult


class OptimizationRepository(ABC):
    """Abstract repository for optimization operations."""
    
    @abstractmethod
    async def optimize_knn_parameters(
        self, 
        request: OptimizationRequest
    ) -> OptimizationResult:
        """Optimize KNN parameters using meta-heuristic algorithms."""
        pass


class OptimizationRepositoryProtocol(Protocol):
    """Protocol for optimization repository."""
    
    async def optimize_knn_parameters(
        self, 
        request: OptimizationRequest
    ) -> OptimizationResult:
        """Optimize KNN parameters using meta-heuristic algorithms."""
        ... 