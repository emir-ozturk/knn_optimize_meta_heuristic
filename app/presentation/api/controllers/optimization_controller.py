"""FastAPI controller for optimization endpoints."""

import logging
from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
import pandas as pd

from ..models.request_models import OptimizationRequestModel
from ..models.response_models import OptimizationResponseModel, ErrorResponse
from ....domain.usecases.optimize_knn_usecase import OptimizeKNNUseCase
from ....domain.entities.optimization_request import OptimizationRequest
from ....core.utils.data_processor import DataProcessor
from ....core.errors.exceptions import (
    DataValidationError, 
    OptimizationError, 
    FileProcessingError,
    InvalidParameterError
)


logger = logging.getLogger(__name__)


class OptimizationController:
    """Controller for optimization-related endpoints."""
    
    def __init__(self, optimize_knn_usecase: OptimizeKNNUseCase):
        self.optimize_knn_usecase = optimize_knn_usecase
        self.data_processor = DataProcessor()
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        self.router.add_api_route(
            "/optimize",
            self.optimize_knn,
            methods=["POST"],
            response_model=OptimizationResponseModel,
            responses={
                400: {"model": ErrorResponse},
                422: {"model": ErrorResponse},
                500: {"model": ErrorResponse}
            }
        )
        
        self.router.add_api_route(
            "/validate-dataset",
            self.validate_dataset,
            methods=["POST"],
            response_model=Dict[str, Any]
        )
    
    async def optimize_knn(
        self,
        file: UploadFile = File(..., description="Dataset file (CSV, XLSX)"),
        target_column: str = Form("Outcome", description="Target column name"),
        algorithm: str = Form(default="PSO", description="Optimization algorithm"),
        max_iterations: int = Form(default=50, description="Maximum iterations"),
        population_size: int = Form(default=20, description="Population size"),
        test_size: float = Form(default=0.2, description="Test size"),
        random_state: int = Form(default=42, description="Random state"),
        cv_folds: int = Form(default=5, description="CV folds")
    ) -> OptimizationResponseModel:
        """Optimize KNN parameters using uploaded dataset."""
        try:
            logger.info(f"Starting KNN optimization for file: {file.filename}")
            
            # Validate file
            if not file.filename:
                raise HTTPException(status_code=400, detail="No file provided")
            
            file_extension = "." + file.filename.split(".")[-1].lower()
            if file_extension not in [".csv", ".xlsx", ".xls"]:
                raise HTTPException(
                    status_code=400, 
                    detail="Unsupported file format. Use CSV or Excel files."
                )
            
            # Read file content
            file_content = await file.read()
            
            # Load dataset
            dataset = self.data_processor.load_dataset(file_content, file_extension)
            
            # Validate dataset
            self.data_processor.validate_dataset(dataset, target_column)
            
            # Create optimization request
            request_model = OptimizationRequestModel(
                target_column=target_column,
                algorithm=algorithm,
                max_iterations=max_iterations,
                population_size=population_size,
                test_size=test_size,
                random_state=random_state,
                cv_folds=cv_folds
            )
            
            # Convert to domain entity
            optimization_request = OptimizationRequest(
                dataset=dataset,
                target_column=request_model.target_column,
                algorithm=request_model.algorithm.value,
                max_iterations=request_model.max_iterations,
                population_size=request_model.population_size,
                test_size=request_model.test_size,
                random_state=request_model.random_state,
                cv_folds=request_model.cv_folds,
                additional_params=request_model.additional_params
            )
            
            # Execute optimization
            result = await self.optimize_knn_usecase.execute(optimization_request)
            
            # Convert to response model
            response_data = result.to_dict()
            
            logger.info(f"Optimization completed successfully. Best accuracy: {result.best_accuracy:.4f}")
            
            return OptimizationResponseModel(**response_data)
            
        except (DataValidationError, InvalidParameterError) as e:
            logger.error(f"Validation error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        
        except FileProcessingError as e:
            logger.error(f"File processing error: {str(e)}")
            raise HTTPException(status_code=422, detail=str(e))
        
        except OptimizationError as e:
            logger.error(f"Optimization error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def validate_dataset(
        self,
        file: UploadFile = File(..., description="Dataset file to validate")
    ) -> Dict[str, Any]:
        """Validate dataset and return information about it."""
        try:
            if not file.filename:
                raise HTTPException(status_code=400, detail="No file provided")
            
            file_extension = "." + file.filename.split(".")[-1].lower()
            file_content = await file.read()
            
            # Load dataset
            dataset = self.data_processor.load_dataset(file_content, file_extension)
            
            # Get dataset information
            info = {
                "filename": file.filename,
                "shape": list(dataset.shape),
                "columns": list(dataset.columns),
                "dtypes": {col: str(dtype) for col, dtype in dataset.dtypes.items()},
                "missing_values": dataset.isnull().sum().to_dict(),
                "summary_stats": dataset.describe().to_dict() if len(dataset.select_dtypes(include=[float, int]).columns) > 0 else {}
            }
            
            return {
                "status": "success",
                "dataset_info": info,
                "message": "Dataset validated successfully"
            }
            
        except FileProcessingError as e:
            logger.error(f"File processing error: {str(e)}")
            raise HTTPException(status_code=422, detail=str(e))
        
        except Exception as e:
            logger.error(f"Unexpected error during validation: {str(e)}")
            raise HTTPException(status_code=500, detail="Validation failed") 