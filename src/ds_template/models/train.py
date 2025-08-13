"""Training module with baseline model wrapper.

This module provides training functionality with support for baseline models
and includes model evaluation, logging, and persistence capabilities.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    model_type: str = 'random_forest'
    task_type: str = 'regression'  # 'regression' or 'classification'
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    use_mlflow: bool = True
    model_save_path: str = 'models'
    experiment_name: str = 'ds_template_experiment'
    

class ModelTrainer:
    """Trainer class for baseline models with MLflow integration."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.training_metrics = {}
        self.validation_metrics = {}
        self.model_path = None
        
        # Set up MLflow if available
        if HAS_MLFLOW and config.use_mlflow:
            mlflow.set_experiment(config.experiment_name)
    
    def _get_baseline_model(self) -> BaseEstimator:
        """Get baseline model based on configuration."""
        if self.config.task_type == 'regression':
            if self.config.model_type == 'random_forest':
                return RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.config.random_state
                )
            elif self.config.model_type == 'linear':
                return LinearRegression()
        elif self.config.task_type == 'classification':
            if self.config.model_type == 'random_forest':
                return RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.config.random_state
                )
            elif self.config.model_type == 'logistic':
                return LogisticRegression(
                    random_state=self.config.random_state
                )
        
        raise ValueError(f"Unsupported model_type '{self.config.model_type}' for task_type '{self.config.task_type}'")
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                        y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Add AUC for binary classification
        if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            
        return metrics
    
    def _log_metrics_to_mlflow(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics to MLflow if available."""
        if HAS_MLFLOW and self.config.use_mlflow:
            for key, value in metrics.items():
                mlflow.log_metric(f"{prefix}{key}" if prefix else key, value)
    
    def _save_model(self, model: BaseEstimator) -> str:
        """Save trained model to disk."""
        # Create models directory if it doesn't exist
        Path(self.config.model_save_path).mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.model_type}_{self.config.task_type}_{timestamp}.pkl"
        model_path = os.path.join(self.config.model_save_path, filename)
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save config alongside model
        config_path = model_path.replace('.pkl', '_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'model_type': self.config.model_type,
                'task_type': self.config.task_type,
                'timestamp': timestamp,
                'training_metrics': self.training_metrics,
                'validation_metrics': self.validation_metrics
            }, f, indent=2)
        
        return model_path
    
    def train_baseline(self, 
                      X: Union[pd.DataFrame, np.ndarray], 
                      y: Union[pd.Series, np.ndarray],
                      validation_data: Optional[Tuple[Union[pd.DataFrame, np.ndarray], 
                                                    Union[pd.Series, np.ndarray]]] = None
                      ) -> Dict[str, Any]:
        """Train baseline model with comprehensive evaluation.
        
        Args:
            X: Feature matrix
            y: Target variable
            validation_data: Optional validation data tuple (X_val, y_val)
            
        Returns:
            Dictionary containing training results
        """
        # Start MLflow run if available
        if HAS_MLFLOW and self.config.use_mlflow:
            mlflow.start_run()
        
        try:
            # Convert to numpy if pandas
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
            
            # Split data if no validation data provided
            if validation_data is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=self.config.test_size, 
                    random_state=self.config.random_state
                )
            else:
                X_train, y_train = X, y
                X_val, y_val = validation_data
                if isinstance(X_val, pd.DataFrame):
                    X_val = X_val.values
                if isinstance(y_val, pd.Series):
                    y_val = y_val.values
            
            # Initialize and train model
            self.model = self._get_baseline_model()
            self.model.fit(X_train, y_train)
            
            # Generate predictions
            y_train_pred = self.model.predict(X_train)
            y_val_pred = self.model.predict(X_val)
            
            # Calculate metrics
            if self.config.task_type == 'regression':
                self.training_metrics = self._calculate_regression_metrics(y_train, y_train_pred)
                self.validation_metrics = self._calculate_regression_metrics(y_val, y_val_pred)
            else:
                # Get prediction probabilities for classification
                y_val_proba = self.model.predict_proba(X_val) if hasattr(self.model, 'predict_proba') else None
                self.training_metrics = self._calculate_classification_metrics(y_train, y_train_pred)
                self.validation_metrics = self._calculate_classification_metrics(y_val, y_val_pred, y_val_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(
                self.model, X_train, y_train, 
                cv=self.config.cv_folds, 
                scoring='neg_mean_squared_error' if self.config.task_type == 'regression' else 'accuracy'
            )
            
            cv_metric = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Log to MLflow
            if HAS_MLFLOW and self.config.use_mlflow:
                # Log parameters
                mlflow.log_param('model_type', self.config.model_type)
                mlflow.log_param('task_type', self.config.task_type)
                mlflow.log_param('test_size', self.config.test_size)
                mlflow.log_param('cv_folds', self.config.cv_folds)
                
                # Log metrics
                self._log_metrics_to_mlflow(self.training_metrics, 'train_')
                self._log_metrics_to_mlflow(self.validation_metrics, 'val_')
                self._log_metrics_to_mlflow(cv_metric)
                
                # Log model
                mlflow.sklearn.log_model(self.model, "model")
            
            # Save model locally
            self.model_path = self._save_model(self.model)
            
            # Prepare results
            results = {
                'model': self.model,
                'model_path': self.model_path,
                'training_metrics': self.training_metrics,
                'validation_metrics': self.validation_metrics,
                'cross_validation': cv_metric,
                'config': self.config
            }
            
            return results
            
        finally:
            # End MLflow run if active
            if HAS_MLFLOW and self.config.use_mlflow and mlflow.active_run():
                mlflow.end_run()
    
    def load_model(self, model_path: str) -> BaseEstimator:
        """Load a saved model."""
        self.model = joblib.load(model_path)
        self.model_path = model_path
        return self.model
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions with trained model."""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train_baseline() or load_model() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict(X)


def train_baseline(X: Union[pd.DataFrame, np.ndarray], 
                  y: Union[pd.Series, np.ndarray],
                  model_type: str = 'random_forest',
                  task_type: str = 'regression',
                  **kwargs) -> Dict[str, Any]:
    """Convenient function to train baseline model.
    
    Args:
        X: Feature matrix
        y: Target variable  
        model_type: Type of model ('random_forest', 'linear', 'logistic')
        task_type: Type of task ('regression', 'classification')
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary containing training results
    """
    config = TrainingConfig(
        model_type=model_type,
        task_type=task_type,
        **kwargs
    )
    
    trainer = ModelTrainer(config)
    return trainer.train_baseline(X, y)


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression, make_classification
    
    print("Testing regression model...")
    X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    results_reg = train_baseline(X_reg, y_reg, model_type='random_forest', task_type='regression')
    print(f"Regression validation R²: {results_reg['validation_metrics']['r2']:.4f}")
    
    print("\nTesting classification model...")
    X_clf, y_clf = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    results_clf = train_baseline(X_clf, y_clf, model_type='random_forest', task_type='classification')
    print(f"Classification validation accuracy: {results_clf['validation_metrics']['accuracy']:.4f}")
    
    print("\nBaseline training completed successfully!")
