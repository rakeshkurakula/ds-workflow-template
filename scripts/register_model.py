#!/usr/bin/env python3
"""
MLflow Model Registry Integration - Model Registration Script

This script handles logging and registering models to the MLflow Model Registry.
It supports both local models and models from existing MLflow runs.

Usage:
    python register_model.py --model-path models/my_model.pkl --name "MyModel" --version "1.0"
    python register_model.py --run-id <run_id> --name "MyModel" --description "Production ready model"
"""

import argparse
import os
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelRegistrar:
    """Handles model registration and logging to MLflow Model Registry."""
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize the ModelRegistrar.
        
        Args:
            tracking_uri: MLflow tracking server URI. If None, uses default.
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.client = MlflowClient()
        logger.info(f"Connected to MLflow tracking server: {mlflow.get_tracking_uri()}")
    
    def log_and_register_model(
        self,
        model_path: str,
        model_name: str,
        version: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        experiment_name: str = "model-registration"
    ) -> str:
        """
        Log a local model and register it to the Model Registry.
        
        Args:
            model_path: Path to the serialized model file
            model_name: Name for the registered model
            version: Model version (optional)
            description: Model description
            tags: Additional tags for the model
            experiment_name: MLflow experiment name
            
        Returns:
            str: The registered model version
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Set or create experiment
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run() as run:
            logger.info(f"Started MLflow run: {run.info.run_id}")
            
            # Load and log the model
            try:
                # Try to load as pickle first
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                # Log model artifacts
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=model_name
                )
                
            except Exception as e:
                logger.warning(f"Failed to log as sklearn model: {e}")
                # Fallback: log as generic artifact
                mlflow.log_artifact(str(model_path), "model")
            
            # Log metadata
            if version:
                mlflow.log_param("model_version", version)
            if description:
                mlflow.log_param("description", description)
            
            # Log model file size and path
            mlflow.log_param("model_file_size", model_path.stat().st_size)
            mlflow.log_param("model_file_path", str(model_path))
            
            # Add tags
            if tags:
                mlflow.set_tags(tags)
            
            # Register the model
            model_uri = f"runs:/{run.info.run_id}/model"
            return self._register_model_version(model_uri, model_name, description, tags)
    
    def register_existing_run(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model",
        description: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a model from an existing MLflow run.
        
        Args:
            run_id: ID of the existing MLflow run
            model_name: Name for the registered model
            artifact_path: Path to model artifact within the run
            description: Model description
            tags: Additional tags for the model
            
        Returns:
            str: The registered model version
        """
        try:
            # Verify the run exists
            run = self.client.get_run(run_id)
            logger.info(f"Found run: {run_id} from experiment: {run.info.experiment_id}")
            
            # Construct model URI
            model_uri = f"runs:/{run_id}/{artifact_path}"
            
            return self._register_model_version(model_uri, model_name, description, tags)
            
        except MlflowException as e:
            logger.error(f"Failed to find run {run_id}: {e}")
            raise
    
    def _register_model_version(
        self,
        model_uri: str,
        model_name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a model version to the Model Registry.
        
        Args:
            model_uri: URI of the model to register
            model_name: Name for the registered model
            description: Model description
            tags: Additional tags for the model
            
        Returns:
            str: The registered model version
        """
        try:
            # Create registered model if it doesn't exist
            try:
                self.client.get_registered_model(model_name)
                logger.info(f"Using existing registered model: {model_name}")
            except MlflowException:
                logger.info(f"Creating new registered model: {model_name}")
                self.client.create_registered_model(
                    name=model_name,
                    description=description or f"Registered model: {model_name}"
                )
            
            # Register the model version
            model_version = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                description=description
            )
            
            # Add tags to the model version
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        name=model_name,
                        version=model_version.version,
                        key=key,
                        value=str(value)
                    )
            
            logger.info(
                f"Successfully registered model '{model_name}' version {model_version.version}"
            )
            logger.info(f"Model URI: {model_uri}")
            
            return model_version.version
            
        except MlflowException as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def list_registered_models(self) -> None:
        """List all registered models in the registry."""
        try:
            models = self.client.search_registered_models()
            
            if not models:
                logger.info("No registered models found.")
                return
            
            logger.info("Registered Models:")
            for model in models:
                logger.info(f"  - {model.name}: {model.description or 'No description'}")
                
                # Get latest versions
                versions = self.client.get_latest_versions(model.name)
                for version in versions:
                    logger.info(
                        f"    Version {version.version} ({version.current_stage}): "
                        f"{version.description or 'No description'}"
                    )
                    
        except MlflowException as e:
            logger.error(f"Failed to list registered models: {e}")
            raise


def main():
    """Main function to handle command line arguments and execute model registration."""
    parser = argparse.ArgumentParser(
        description="Register models to MLflow Model Registry"
    )
    
    # Model source arguments (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--model-path",
        help="Path to local model file to register"
    )
    source_group.add_argument(
        "--run-id",
        help="MLflow run ID containing the model to register"
    )
    source_group.add_argument(
        "--list",
        action="store_true",
        help="List all registered models"
    )
    
    # Registration arguments
    parser.add_argument(
        "--name",
        help="Name for the registered model (required unless --list)"
    )
    parser.add_argument(
        "--version",
        help="Model version (for local models only)"
    )
    parser.add_argument(
        "--description",
        help="Description for the registered model"
    )
    parser.add_argument(
        "--artifact-path",
        default="model",
        help="Artifact path within the run (default: model)"
    )
    parser.add_argument(
        "--experiment-name",
        default="model-registration",
        help="MLflow experiment name (default: model-registration)"
    )
    parser.add_argument(
        "--tracking-uri",
        help="MLflow tracking server URI"
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        help="Model tags in key=value format (e.g., env=prod team=data-science)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.list and not args.name:
        parser.error("--name is required unless using --list")
    
    # Parse tags
    tags = {}
    if args.tags:
        for tag in args.tags:
            if "=" not in tag:
                parser.error(f"Invalid tag format: {tag}. Use key=value format.")
            key, value = tag.split("=", 1)
            tags[key] = value
    
    try:
        # Initialize registrar
        registrar = ModelRegistrar(tracking_uri=args.tracking_uri)
        
        if args.list:
            registrar.list_registered_models()
            
        elif args.model_path:
            # Register local model
            version = registrar.log_and_register_model(
                model_path=args.model_path,
                model_name=args.name,
                version=args.version,
                description=args.description,
                tags=tags,
                experiment_name=args.experiment_name
            )
            print(f"Successfully registered model '{args.name}' version {version}")
            
        elif args.run_id:
            # Register model from existing run
            version = registrar.register_existing_run(
                run_id=args.run_id,
                model_name=args.name,
                artifact_path=args.artifact_path,
                description=args.description,
                tags=tags
            )
            print(f"Successfully registered model '{args.name}' version {version}")
            
    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
