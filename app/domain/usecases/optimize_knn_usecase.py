"""Use case for KNN optimization."""

from typing import Protocol
import logging

from ..entities.optimization_request import OptimizationRequest
from ..entities.optimization_result import OptimizationResult
from ..repositories.optimization_repository import OptimizationRepositoryProtocol
from ...core.errors.exceptions import OptimizationError, InvalidParameterError


logger = logging.getLogger(__name__)


class OptimizeKNNUseCase:
    """Use case for optimizing KNN parameters."""
    
    def __init__(self, optimization_repository: OptimizationRepositoryProtocol):
        self._optimization_repository = optimization_repository
    
    async def execute(self, request: OptimizationRequest) -> OptimizationResult:
        """Execute KNN optimization."""
        try:
            logger.info(f"Starting KNN optimization with algorithm: {request.algorithm}")
            
            # Validate the request
            self._validate_request(request)
            
            # Perform optimization
            result = await self._optimization_repository.optimize_knn_parameters(request)
            
            logger.info(f"Optimization completed. Best accuracy: {result.best_accuracy:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            if isinstance(e, (OptimizationError, InvalidParameterError)):
                raise
            raise OptimizationError(f"Unexpected error during optimization: {str(e)}")
    
    def _validate_request(self, request: OptimizationRequest) -> None:
        """Validate optimization request."""
        if request.dataset.empty:
            raise InvalidParameterError("Dataset cannot be empty")
        
        if request.target_column not in request.dataset.columns:
            raise InvalidParameterError(
                f"Target column '{request.target_column}' not found in dataset"
            )
        
        available_algorithms = ["PSO", "GA", "DE", "SA", "WOA"]
        if request.algorithm not in available_algorithms:
            raise InvalidParameterError(
                f"Algorithm '{request.algorithm}' not supported. "
                f"Available algorithms: {available_algorithms}"
            ) 