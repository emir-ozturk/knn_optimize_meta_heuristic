"""Main FastAPI application for KNN optimization."""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .infrastructure.config.dependency_injection import get_optimization_controller


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app with metadata
app = FastAPI(
    title="KNN Optimize Meta Heuristic API",
    description="""
    API for KNN parameter optimization using meta-heuristic algorithms.
    
    ## Features
    - Upload dataset (CSV, Excel)
    - Optimize KNN parameters using various meta-heuristic algorithms
    - Support for PSO, GA, DE, SA, WOA algorithms
    - Comprehensive performance metrics
    - Dataset validation
    
    ## Usage
    1. Upload your dataset using `/knn/optimize` endpoint
    2. Specify the target column name
    3. Choose optimization algorithm and parameters
    4. Get optimized KNN parameters and performance metrics
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting KNN Optimize Meta Heuristic API")
    logger.info("Clean Architecture with Domain-Driven Design")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down KNN Optimize Meta Heuristic API")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "KNN Optimize Meta Heuristic API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "optimize": "/knn/optimize",
            "validate": "/knn/validate-dataset",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "KNN Optimize Meta Heuristic API",
        "version": "1.0.0"
    }


# Include optimization routes
optimization_controller = get_optimization_controller()
app.include_router(
    optimization_controller.router,
    prefix="/knn",
    tags=["KNN Optimization"]
)


# For development - run with: python -m app.main
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Only for development
        log_level="info"
    )
