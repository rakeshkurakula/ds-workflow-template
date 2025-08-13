.PHONY: setup run-mlflow ge-init test lint format serve precommit help

# Default help target
help:
	@echo "Available targets:"
	@echo "  setup      - Install dependencies using uv sync"
	@echo "  run-mlflow - Start MLflow UI server"
	@echo "  ge-init    - Initialize Great Expectations"
	@echo "  test       - Run tests"
	@echo "  lint       - Run linting with ruff"
	@echo "  format     - Format code with ruff format"
	@echo "  serve      - Start development server with uvicorn"
	@echo "  precommit  - Run tests and linters (CI pipeline)"

# Setup: Install dependencies using uv
setup:
	@echo "Setting up environment with uv..."
	uv sync
	@echo "Dependencies installed successfully!"

# MLflow: Start MLflow UI
run-mlflow:
	@echo "Starting MLflow UI..."
	mlflow ui

# Great Expectations: Initialize GE
ge-init:
	@echo "Initializing Great Expectations..."
	uv run great_expectations init

# Testing: Run test suite
test:
	@echo "Running tests..."
	uv run pytest

# Linting: Run ruff linter
lint:
	@echo "Running linter (ruff)..."
	uv run ruff check .

# Formatting: Format code with ruff
format:
	@echo "Formatting code with ruff..."
	uv run ruff format .

# Serve: Start development server
serve:
	@echo "Starting development server..."
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Precommit: Run tests and linters (CI pipeline)
precommit: test lint
	@echo "Pre-commit checks completed successfully!"

# Clean: Clean up temporary files and caches
clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .coverage
	rm -rf htmlcov/
