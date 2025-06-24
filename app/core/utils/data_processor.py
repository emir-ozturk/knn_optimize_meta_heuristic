"""Data processing utilities."""

import pandas as pd
import numpy as np
import io
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from ..errors.exceptions import DataValidationError, FileProcessingError


class DataProcessor:
    """Utility class for data processing operations."""
    
    @staticmethod
    def load_dataset(file_content: bytes, file_extension: str) -> pd.DataFrame:
        """Load dataset from file content."""
        try:
            if file_extension.lower() == '.csv':
                return pd.read_csv(io.BytesIO(file_content))
            elif file_extension.lower() in ['.xlsx', '.xls']:
                return pd.read_excel(io.BytesIO(file_content))
            else:
                raise FileProcessingError(
                    "Unsupported file format",
                    {"supported_formats": [".csv", ".xlsx", ".xls"]}
                )
        except Exception as e:
            raise FileProcessingError(f"Failed to load dataset: {str(e)}")
    
    @staticmethod
    def validate_dataset(df: pd.DataFrame, target_column: str) -> None:
        """Validate dataset structure and target column."""
        if df.empty:
            raise DataValidationError("Dataset is empty")
        
        if target_column not in df.columns:
            raise DataValidationError(
                f"Target column '{target_column}' not found",
                {"available_columns": list(df.columns)}
            )
        
        if df.shape[1] < 2:
            raise DataValidationError("Dataset must have at least 2 columns")
        
        if df.shape[0] < 10:
            raise DataValidationError("Dataset must have at least 10 rows")
    
    @staticmethod
    def preprocess_data(
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Preprocess data for KNN training."""
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical features
        categorical_columns = X.select_dtypes(include=['object']).columns
        feature_names = list(X.columns)
        
        # Encode categorical variables
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Encode target if categorical
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, feature_names 