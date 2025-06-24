"""Dependency injection configuration."""

from functools import lru_cache
import logging

from ...data.repositories.optimization_repository_impl import OptimizationRepositoryImpl
from ...domain.usecases.optimize_knn_usecase import OptimizeKNNUseCase
from ...presentation.api.controllers.optimization_controller import OptimizationController


logger = logging.getLogger(__name__)


class DependencyContainer:
    """Container for managing dependencies."""
    
    def __init__(self):
        self._optimization_repository = None
        self._optimize_knn_usecase = None
        self._optimization_controller = None
    
    @property
    def optimization_repository(self) -> OptimizationRepositoryImpl:
        """Get optimization repository instance."""
        if self._optimization_repository is None:
            self._optimization_repository = OptimizationRepositoryImpl()
            logger.info("Created OptimizationRepositoryImpl instance")
        return self._optimization_repository
    
    @property
    def optimize_knn_usecase(self) -> OptimizeKNNUseCase:
        """Get optimize KNN use case instance."""
        if self._optimize_knn_usecase is None:
            self._optimize_knn_usecase = OptimizeKNNUseCase(
                optimization_repository=self.optimization_repository
            )
            logger.info("Created OptimizeKNNUseCase instance")
        return self._optimize_knn_usecase
    
    @property
    def optimization_controller(self) -> OptimizationController:
        """Get optimization controller instance."""
        if self._optimization_controller is None:
            self._optimization_controller = OptimizationController(
                optimize_knn_usecase=self.optimize_knn_usecase
            )
            logger.info("Created OptimizationController instance")
        return self._optimization_controller


@lru_cache()
def get_dependency_container() -> DependencyContainer:
    """Get singleton dependency container."""
    return DependencyContainer()


def get_optimization_controller() -> OptimizationController:
    """FastAPI dependency for getting optimization controller."""
    container = get_dependency_container()
    return container.optimization_controller 