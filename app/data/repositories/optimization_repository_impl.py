"""Concrete implementation of optimization repository."""

import time
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from mealpy import PSO, GA, DE, SA, WOA
from mealpy.utils.space import IntegerVar, FloatVar, StringVar

from ...domain.repositories.optimization_repository import OptimizationRepository
from ...domain.entities.optimization_request import OptimizationRequest
from ...domain.entities.optimization_result import OptimizationResult
from ...core.utils.data_processor import DataProcessor
from ...core.errors.exceptions import OptimizationError, ModelTrainingError


logger = logging.getLogger(__name__)


class OptimizationRepositoryImpl(OptimizationRepository):
    """Concrete implementation of optimization repository using mealpy."""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.optimization_history = []
    
    async def optimize_knn_parameters(
        self, 
        request: OptimizationRequest
    ) -> OptimizationResult:
        """Optimize KNN parameters using meta-heuristic algorithms."""
        try:
            start_time = time.time()
            
            # Preprocess data
            X_train, X_test, y_train, y_test, feature_names = self.data_processor.preprocess_data(
                request.dataset, 
                request.target_column,
                request.test_size,
                request.random_state
            )
            
            # Reset optimization history
            self.optimization_history = []
            
            # Define optimization problem
            def objective_function(solution):
                return self._evaluate_knn_solution(solution, X_train, y_train, request.cv_folds)
            
            # Set up optimization bounds
            problem_dict = {
                "bounds": self._get_parameter_bounds(),
                "minmax": "max",  # Maximize accuracy
                "obj_func": objective_function
            }
            
            # Select and run optimizer
            optimizer = self._get_optimizer(request.algorithm, request.population_size, request.max_iterations)
            
            # Run optimization - new mealpy version returns Agent object
            try:
                result = optimizer.solve(
                    problem_dict, 
                    mode="single", 
                    n_workers=1,
                    termination={"max_fe": request.max_iterations * request.population_size}
                )
                
                # Handle different return types based on mealpy version
                logger.info(f"Optimization result type: {type(result)}")
                logger.info(f"Optimization result attributes: {dir(result) if hasattr(result, '__dict__') else 'No attributes'}")
                
                if hasattr(result, 'solution') and hasattr(result, 'target'):
                    # New version: Agent object
                    best_solution = result.solution
                    best_fitness = result.target.fitness
                    logger.info(f"Using Agent object format - fitness: {best_fitness}")
                elif hasattr(result, 'solution'):
                    # Agent object without target
                    best_solution = result.solution
                    best_fitness = self._evaluate_knn_solution(best_solution, X_train, y_train, request.cv_folds)
                    logger.info(f"Using Agent object without target - calculated fitness: {best_fitness}")
                elif isinstance(result, tuple) and len(result) == 2:
                    # Old version: tuple
                    best_solution, best_fitness = result
                    logger.info(f"Using tuple format - fitness: {best_fitness}")
                elif hasattr(result, '__iter__') and not isinstance(result, str):
                    # Iterable result (list/array)
                    result_list = list(result)
                    if len(result_list) >= 1:
                        best_solution = result_list[0]
                        best_fitness = self._evaluate_knn_solution(best_solution, X_train, y_train, request.cv_folds)
                        logger.info(f"Using iterable format - calculated fitness: {best_fitness}")
                    else:
                        raise OptimizationError("Empty result from optimizer")
                else:
                    # Fallback: assume result is the solution
                    best_solution = result
                    best_fitness = self._evaluate_knn_solution(best_solution, X_train, y_train, request.cv_folds)
                    logger.info(f"Using fallback format - calculated fitness: {best_fitness}")
                    
            except Exception as e:
                logger.error(f"Optimization solve error: {str(e)}")
                raise OptimizationError(f"Optimization failed during solve: {str(e)}")
            
            # Extract best parameters
            best_params = self._solution_to_parameters(best_solution)
            
            # Train final model with best parameters
            final_knn = KNeighborsClassifier(**best_params)
            final_knn.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = final_knn.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            execution_time = time.time() - start_time
            
            # Dataset info
            dataset_info = {
                "shape": request.dataset.shape,
                "n_features": len(feature_names),
                "n_samples": len(request.dataset),
                "target_column": request.target_column,
                "unique_classes": len(np.unique(y_test))
            }
            
            result = OptimizationResult(
                best_parameters=best_params,
                best_accuracy=accuracy,
                best_f1_score=f1,
                best_precision=precision,
                best_recall=recall,
                optimization_history=self.optimization_history,
                execution_time=execution_time,
                feature_names=feature_names,
                dataset_info=dataset_info,
                algorithm_used=request.algorithm
            )
            
            logger.info(f"Optimization completed successfully in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise OptimizationError(f"Failed to optimize KNN parameters: {str(e)}")
    
    def _evaluate_knn_solution(
        self, 
        solution: np.ndarray, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        cv_folds: int
    ) -> float:
        """Evaluate KNN solution using cross-validation."""
        try:
            params = self._solution_to_parameters(solution)
            knn = KNeighborsClassifier(**params)
            
            # Use cross-validation to evaluate
            scores = cross_val_score(knn, X_train, y_train, cv=cv_folds, scoring='accuracy')
            mean_score = scores.mean()
            
            # Store in history
            self.optimization_history.append({
                "parameters": params,
                "fitness": mean_score,
                "std_score": scores.std()
            })
            
            return mean_score
            
        except Exception as e:
            logger.warning(f"Error evaluating solution: {str(e)}")
            return 0.0  # Return poor fitness for invalid solutions
    
    def _solution_to_parameters(self, solution: np.ndarray) -> Dict[str, Any]:
        """Convert solution array to KNN parameters."""
        n_neighbors = int(solution[0])  # Already bounded by IntegerVar
        weights = 'uniform' if solution[1] == 0 else 'distance'
        
        # Map algorithm index to algorithm name
        algorithm_idx = int(solution[2])  # 0-3 range
        algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
        algorithm = algorithms[algorithm_idx]
        
        # Map metric index to metric name
        metric_idx = int(solution[3])  # 0-2 range
        metrics = ['euclidean', 'manhattan', 'minkowski']
        metric = metrics[metric_idx]
        
        params = {
            'n_neighbors': n_neighbors,
            'weights': weights,
            'algorithm': algorithm,
            'metric': metric
        }
        
        # Add p parameter for minkowski
        if metric == 'minkowski':
            params['p'] = int(solution[4])  # 1-3 range
        
        return params
    
    def _get_parameter_bounds(self) -> List:
        """Get parameter bounds for optimization using mealpy space variables."""
        return [
            IntegerVar(lb=1, ub=50, name="n_neighbors"),  # n_neighbors: 1-50
            IntegerVar(lb=0, ub=1, name="weights"),       # weights: 0=uniform, 1=distance  
            IntegerVar(lb=0, ub=3, name="algorithm"),     # algorithm: 0-3 for 4 options
            IntegerVar(lb=0, ub=2, name="metric"),        # metric: 0-2 for 3 options
            IntegerVar(lb=1, ub=3, name="p")              # p: for minkowski metric (1-3)
        ]
    
    def _get_optimizer(self, algorithm: str, population_size: int, max_iterations: int = 50):
        """Get optimizer instance based on algorithm name."""
        try:
            # Use dynamic epoch based on max_iterations
            epoch = min(max_iterations, 200)  # Cap epoch at 200 for performance
            
            # Try different parameter names for compatibility
            if algorithm == "PSO":
                try:
                    return PSO.OriginalPSO(epoch=epoch, pop_size=population_size)
                except TypeError:
                    # Try alternative parameter names
                    return PSO.OriginalPSO(epoch=epoch, population_size=population_size)
            elif algorithm == "GA":
                try:
                    return GA.BaseGA(epoch=epoch, pop_size=population_size, pc=0.9, pm=0.1)
                except TypeError:
                    return GA.BaseGA(epoch=epoch, population_size=population_size, pc=0.9, pm=0.1)
            elif algorithm == "DE":
                try:
                    return DE.OriginalDE(epoch=epoch, pop_size=population_size, wf=0.8, cr=0.9)
                except TypeError:
                    return DE.OriginalDE(epoch=epoch, population_size=population_size, wf=0.8, cr=0.9)
            elif algorithm == "SA":
                try:
                    return SA.OriginalSA(epoch=epoch, max_sub_iter=10, t0=1000, t1=1)
                except TypeError:
                    return SA.OriginalSA(epoch=epoch, max_sub_iter=10, temp_init=1000, temp_final=1)
            elif algorithm == "WOA":
                try:
                    return WOA.OriginalWOA(epoch=epoch, pop_size=population_size)
                except TypeError:
                    return WOA.OriginalWOA(epoch=epoch, population_size=population_size)
            else:
                raise OptimizationError(f"Unsupported algorithm: {algorithm}")
                
        except Exception as e:
            logger.error(f"Error creating optimizer {algorithm}: {str(e)}")
            raise OptimizationError(f"Failed to create optimizer {algorithm}: {str(e)}") 