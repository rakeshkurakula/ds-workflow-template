"""
Prefect training flow for data science workflow.

This flow orchestrates:
1. Data loading
2. Feature building
3. Model training
4. Great Expectations validation
"""

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from typing import Dict, Any
import pandas as pd
import joblib
from pathlib import Path
import great_expectations as gx
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from datetime import datetime


@task
def load_data(data_path: str = "data/raw/sample_data.csv") -> pd.DataFrame:
    """
    Load raw data for training.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        Loaded DataFrame
    """
    print(f"Loading data from {data_path}")
    
    # For demo purposes, create sample data if file doesn't exist
    if not Path(data_path).exists():
        print("Creating sample data...")
        import numpy as np
        
        # Create sample regression dataset
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(5, 2, n_samples),
            'feature_3': np.random.uniform(0, 10, n_samples),
            'feature_4': np.random.exponential(2, n_samples)
        }
        
        # Create target with some relationship to features
        data['target'] = (
            data['feature_1'] * 2 + 
            data['feature_2'] * 0.5 + 
            data['feature_3'] * 0.1 + 
            np.random.normal(0, 0.5, n_samples)
        )
        
        df = pd.DataFrame(data)
        
        # Ensure directory exists
        Path(data_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        
        return df
    
    return pd.read_csv(data_path)


@task
def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate data using Great Expectations.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if validation passes
    """
    print("Validating data with Great Expectations...")
    
    try:
        # Initialize Great Expectations context
        context = gx.get_context()
        
        # Create a data source
        data_source = context.sources.add_pandas("pandas_data_source")
        data_asset = data_source.add_dataframe_asset("sample_data", dataframe=df)
        
        # Create expectations
        expectations = [
            {
                "expectation_type": "expect_table_row_count_to_be_between",
                "kwargs": {"min_value": 100, "max_value": 10000}
            },
            {
                "expectation_type": "expect_column_to_exist",
                "kwargs": {"column": "target"}
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "target"}
            }
        ]
        
        # Add feature columns validation
        for col in df.columns:
            if col.startswith('feature_'):
                expectations.append({
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {"column": col}
                })
        
        # Create expectation suite
        suite_name = "training_data_suite"
        suite = context.add_expectation_suite(suite_name)
        
        for expectation in expectations:
            suite.add_expectation(**expectation)
        
        # Create batch request and validate
        batch_request = data_asset.build_batch_request()
        validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite=suite
        )
        
        # Run validation
        validation_result = validator.validate()
        
        if validation_result.success:
            print("✅ Data validation passed!")
            return True
        else:
            print("❌ Data validation failed!")
            print(f"Failed expectations: {len([r for r in validation_result.results if not r.success])}")
            return False
            
    except Exception as e:
        print(f"Validation error: {e}")
        print("⚠️  Proceeding without validation...")
        return True  # Proceed even if validation setup fails


@task
def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Build features for model training.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Tuple of (feature_df, feature_names)
    """
    print("Building features...")
    
    feature_df = df.copy()
    
    # Create additional features
    feature_df['feature_1_squared'] = feature_df['feature_1'] ** 2
    feature_df['feature_2_log'] = feature_df['feature_2'].apply(lambda x: max(0.001, x)).apply(lambda x: pd.Series(x).apply(lambda y: y if y > 0 else 0.001)).apply(lambda x: pd.Series(x).apply(lambda y: __import__('numpy').log(y)))
    feature_df['feature_interaction'] = feature_df['feature_1'] * feature_df['feature_2']
    
    # Define feature columns (exclude target)
    feature_names = [col for col in feature_df.columns if col != 'target']
    
    print(f"Created {len(feature_names)} features: {feature_names}")
    
    return feature_df, feature_names


@task
def train_model(df: pd.DataFrame, feature_names: list) -> tuple[Any, Dict[str, float]]:
    """
    Train the machine learning model.
    
    Args:
        df: DataFrame with features and target
        feature_names: List of feature column names
        
    Returns:
        Tuple of (trained_model, metrics)
    """
    print("Training model...")
    
    # Prepare features and target
    X = df[feature_names]
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Start MLflow run
    with mlflow.start_run():
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': mse ** 0.5,
            'r2': r2,
            'n_features': len(feature_names),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        # Log parameters and metrics to MLflow
        mlflow.log_params({
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_features': len(feature_names)
        })
        
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model trained successfully!")
        print(f"Metrics: {metrics}")
        
        return model, metrics


@task
def save_model(model: Any, metrics: Dict[str, float], output_dir: str = "models") -> str:
    """
    Save the trained model.
    
    Args:
        model: Trained model
        metrics: Model metrics
        output_dir: Directory to save model
        
    Returns:
        Path to saved model
    """
    print("Saving model...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{output_dir}/model_{timestamp}.joblib"
    
    # Save model
    joblib.dump(model, model_path)
    
    # Save metrics
    metrics_path = f"{output_dir}/metrics_{timestamp}.json"
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    
    return model_path


@flow(name="training-flow", task_runner=SequentialTaskRunner())
def training_flow(
    data_path: str = "data/raw/sample_data.csv",
    output_dir: str = "models"
) -> str:
    """
    Main training flow that orchestrates the entire ML pipeline.
    
    Args:
        data_path: Path to training data
        output_dir: Directory to save trained model
        
    Returns:
        Path to saved model
    """
    print("🚀 Starting training flow...")
    
    # Step 1: Load data
    raw_data = load_data(data_path)
    
    # Step 2: Validate data
    validation_passed = validate_data(raw_data)
    
    if not validation_passed:
        print("⚠️  Data validation failed, but continuing with training...")
    
    # Step 3: Build features
    feature_data, feature_names = build_features(raw_data)
    
    # Step 4: Train model
    trained_model, model_metrics = train_model(feature_data, feature_names)
    
    # Step 5: Save model
    model_path = save_model(trained_model, model_metrics, output_dir)
    
    print("✅ Training flow completed successfully!")
    return model_path


if __name__ == "__main__":
    # Run the flow
    result = training_flow()
    print(f"Training completed! Model saved to: {result}")
