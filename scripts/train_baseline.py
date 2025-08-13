#!/usr/bin/env python3
"""
MLflow Training Script for Iris Dataset

This script demonstrates a basic machine learning workflow using MLflow for tracking:
- Parameters (hyperparameters)
- Metrics (model performance)
- Artifacts (model files, plots)
- Uses MLflow autologging for automatic tracking

Run this script with: python scripts/train_baseline.py
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import mlflow
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifacts


def setup_mlflow():
    """Initialize MLflow tracking."""
    # Set MLflow tracking URI (can be changed to remote server)
    mlflow.set_tracking_uri("mlruns")
    
    # Set or create experiment
    experiment_name = "iris-classification"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment: {experiment_name} (ID: {experiment_id})")


def load_data():
    """Load and prepare the Iris dataset."""
    print("Loading Iris dataset...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    
    # Create target names mapping
    target_names = iris.target_names
    
    return X, y, target_names


def create_visualizations(X, y, target_names, output_dir="plots"):
    """Create and save visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = X.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Pairplot
    df_plot = X.copy()
    df_plot['species'] = [target_names[i] for i in y]
    
    plt.figure(figsize=(12, 10))
    sns.pairplot(df_plot, hue='species', diag_kind='hist')
    plt.savefig(f'{output_dir}/pairplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, feature in enumerate(X.columns):
        for i, species in enumerate(target_names):
            data = X[y == i][feature]
            axes[idx].hist(data, alpha=0.7, label=species, bins=20)
        axes[idx].set_title(f'Distribution of {feature}')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved in {output_dir}/")


def train_model(X_train, y_train, model_type="random_forest", **kwargs):
    """Train a machine learning model."""
    print(f"Training {model_type} model...")
    
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            random_state=kwargs.get('random_state', 42),
            n_jobs=-1
        )
    elif model_type == "logistic_regression":
        model = LogisticRegression(
            C=kwargs.get('C', 1.0),
            max_iter=kwargs.get('max_iter', 1000),
            random_state=kwargs.get('random_state', 42)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_test, y_test, target_names, output_dir="plots"):
    """Evaluate model and create evaluation plots."""
    print("Evaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    class_report = classification_report(y_test, y_pred, target_names=target_names)
    print("\nClassification Report:")
    print(class_report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return accuracy, y_pred, y_proba


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train ML model on Iris dataset with MLflow tracking')
    parser.add_argument('--model_type', default='random_forest', 
                       choices=['random_forest', 'logistic_regression'],
                       help='Type of model to train')
    parser.add_argument('--n_estimators', type=int, default=100,
                       help='Number of estimators for RandomForest')
    parser.add_argument('--max_depth', type=int, default=None,
                       help='Max depth for RandomForest')
    parser.add_argument('--C', type=float, default=1.0,
                       help='Regularization parameter for LogisticRegression')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size ratio')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Setup MLflow
    setup_mlflow()
    
    # Load data
    X, y, target_names = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Start MLflow run
    with mlflow.start_run():
        # Enable autologging (automatically logs parameters, metrics, and artifacts)
        mlflow.sklearn.autolog()
        
        # Log parameters
        log_param("model_type", args.model_type)
        log_param("test_size", args.test_size)
        log_param("random_state", args.random_state)
        log_param("dataset_size", len(X))
        log_param("n_features", X.shape[1])
        log_param("n_classes", len(target_names))
        
        # Model-specific parameters
        if args.model_type == "random_forest":
            log_param("n_estimators", args.n_estimators)
            log_param("max_depth", args.max_depth)
        elif args.model_type == "logistic_regression":
            log_param("C", args.C)
        
        # Create output directory for artifacts
        output_dir = "artifacts"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualizations
        create_visualizations(X, y, target_names, output_dir)
        
        # Train model
        model_params = {
            'random_state': args.random_state,
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'C': args.C
        }
        
        model = train_model(X_train, y_train, args.model_type, **model_params)
        
        # Evaluate model
        accuracy, y_pred, y_proba = evaluate_model(model, X_test, y_test, target_names, output_dir)
        
        # Log metrics
        log_metric("accuracy", accuracy)
        log_metric("train_samples", len(X_train))
        log_metric("test_samples", len(X_test))
        
        # Calculate per-class metrics
        class_report_dict = classification_report(y_test, y_pred, 
                                                target_names=target_names, 
                                                output_dict=True)
        
        for class_name in target_names:
            if class_name in class_report_dict:
                log_metric(f"precision_{class_name}", class_report_dict[class_name]['precision'])
                log_metric(f"recall_{class_name}", class_report_dict[class_name]['recall'])
                log_metric(f"f1_score_{class_name}", class_report_dict[class_name]['f1-score'])
        
        # Log macro averages
        log_metric("precision_macro", class_report_dict['macro avg']['precision'])
        log_metric("recall_macro", class_report_dict['macro avg']['recall'])
        log_metric("f1_macro", class_report_dict['macro avg']['f1-score'])
        
        # Save model summary
        model_summary = {
            'model_type': args.model_type,
            'accuracy': accuracy,
            'n_features': X.shape[1],
            'feature_names': list(X.columns),
            'target_names': list(target_names),
            'model_params': {k: v for k, v in model_params.items() if v is not None}
        }
        
        # Save model summary as JSON
        import json
        with open(f'{output_dir}/model_summary.json', 'w') as f:
            json.dump(model_summary, f, indent=2, default=str)
        
        # Log all artifacts (plots, model summary, etc.)
        log_artifacts(output_dir)
        
        print(f"\n✅ Training completed successfully!")
        print(f"📊 Model Accuracy: {accuracy:.4f}")
        print(f"📁 Artifacts saved in: {output_dir}/")
        print(f"🔬 MLflow UI: Run 'mlflow ui' to view results")
        print(f"   Then open: http://localhost:5000")


if __name__ == "__main__":
    main()
