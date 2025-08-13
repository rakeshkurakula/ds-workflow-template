# Data Science Workflow Template

A comprehensive, production-ready template for data science projects with MLOps best practices, automated workflows, and enterprise-grade features.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (optional)
- Git

### Setup

1. **Clone and setup the project:**
   ```bash
   git clone https://github.com/rakeshkurakula/ds-workflow-template.git
   cd ds-workflow-template
   
   # Install dependencies
   pip install -e .
   
   # Setup pre-commit hooks
   make precommit-install
   ```

2. **Environment Setup:**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install in development mode
   pip install -e ".[dev,test,docs]"
   ```

3. **Quick Test:**
   ```bash
   # Run tests
   make test
   
   # Run data validation
   make validate-data
   
   # Start development server
   make run-app
   ```

## 🏗️ Project Structure

```
ds-workflow-template/
├── .github/                 # GitHub workflows and templates
│   ├── workflows/           # CI/CD pipelines
│   └── CODEOWNERS          # Code ownership definitions
├── app/                     # FastAPI application
│   ├── __init__.py
│   └── main.py             # API endpoints and server
├── data/                    # Data storage and samples
│   └── sample/             # Sample datasets
├── docs/                    # Documentation
│   └── templates/          # Documentation templates
├── feature_repo/           # Feature store configuration
│   ├── features.py         # Feature definitions
│   └── feature_store.yaml  # Feast configuration
├── great_expectations/     # Data quality and validation
│   ├── expectations/       # Data validation rules
│   └── checkpoints/        # Validation checkpoints
├── scripts/                # Utility and deployment scripts
│   ├── deploy.py          # Deployment automation
│   └── register_model.py   # Model registration
├── src/ds_template/        # Core package source
│   ├── data/              # Data processing modules
│   ├── features/          # Feature engineering
│   ├── models/            # Model training and evaluation
│   └── utils/             # Shared utilities
├── tests/                  # Test suites
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── e2e/               # End-to-end tests
├── Dockerfile             # Container configuration
├── Makefile              # Build and automation commands
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

## 🔧 Key Features

### MLOps & DevOps
- **CI/CD Pipelines**: Automated testing, building, and deployment
- **Pre-commit Hooks**: Code quality enforcement
- **Docker Support**: Containerized deployment
- **Model Registry**: MLflow integration for model versioning
- **Feature Store**: Feast integration for feature management

### Data Quality & Validation
- **Great Expectations**: Automated data validation and profiling
- **Data Pipeline Testing**: Point-in-time correctness validation
- **Schema Evolution**: Automated schema change detection

### Development Workflow
- **Code Quality**: Black, flake8, mypy, and pytest integration
- **Documentation**: Automated docs generation
- **Testing Strategy**: Unit, integration, and e2e test frameworks
- **Git Hooks**: Automated quality checks on commits

## 🛠️ Usage Examples

### Data Processing
```python
from ds_template.data import DataProcessor
from ds_template.features import FeatureEngineer

# Load and process data
processor = DataProcessor()
data = processor.load_data('data/sample/iris.csv')

# Feature engineering
fe = FeatureEngineer()
features = fe.transform(data)
```

### Model Training
```python
from ds_template.models import ModelTrainer
from ds_template.models.train import train_model

# Train a model
trainer = ModelTrainer(model_type='random_forest')
model = trainer.fit(X_train, y_train)

# Or use the comprehensive training pipeline
train_model(
    data_path='data/processed/train.csv',
    model_config={'type': 'random_forest', 'n_estimators': 100},
    experiment_name='iris_classification'
)
```

### API Usage
```python
# Start the FastAPI server
# uvicorn app.main:app --reload

# Example prediction request
import requests

response = requests.post(
    'http://localhost:8000/predict',
    json={
        'sepal_length': 5.1,
        'sepal_width': 3.5,
        'petal_length': 1.4,
        'petal_width': 0.2
    }
)
print(response.json())
```

### Data Validation
```bash
# Run Great Expectations validation
great_expectations checkpoint run iris_data_validation

# Custom validation pipeline
python -m ds_template.data.validate --dataset data/sample/iris.csv
```

## 🚀 MLOps Workflows

### Model Lifecycle Management

1. **Training Pipeline:**
   ```bash
   # Train with experiment tracking
   python scripts/train_model.py --config configs/model_config.yaml
   
   # Register best model
   python scripts/register_model.py --run-id <mlflow_run_id>
   ```

2. **Model Deployment:**
   ```bash
   # Deploy to staging
   make deploy-staging
   
   # Promote to production
   make deploy-production
   ```

3. **Monitoring & Drift Detection:**
   ```python
   from ds_template.monitoring import ModelMonitor
   
   monitor = ModelMonitor(model_name='iris_classifier')
   monitor.check_drift(new_data)
   monitor.log_predictions(predictions)
   ```

### Feature Store Integration

```python
from feast import FeatureStore

# Load features for training
store = FeatureStore(repo_path='feature_repo')
features = store.get_historical_features(
    entity_df=entity_df,
    features=[
        'iris_features:sepal_length',
        'iris_features:sepal_width',
        'iris_features:petal_length',
        'iris_features:petal_width'
    ]
).to_df()
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
make test

# Run specific test types
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests
pytest tests/e2e/           # End-to-end tests

# Run with coverage
pytest --cov=src/ds_template --cov-report=html
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **E2E Tests**: Full workflow testing
- **Data Tests**: Data quality and pipeline testing

## 📚 Documentation

### Available Documentation
- **API Reference**: Auto-generated from docstrings
- **User Guide**: Step-by-step tutorials
- **Architecture**: System design and patterns
- **Contributing**: Development guidelines

### Building Documentation
```bash
# Generate documentation
make docs

# Serve locally
make docs-serve
```

## 🔧 Configuration

### Environment Variables
```bash
# MLflow tracking
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME=iris_classification

# Feature store
export FEAST_REPO_PATH=./feature_repo

# Data validation
export GE_DATA_CONTEXT_ROOT_DIR=./great_expectations
```

### Configuration Files
- `pyproject.toml`: Project dependencies and build configuration
- `Makefile`: Build targets and automation commands
- `.pre-commit-config.yaml`: Code quality hooks
- `feature_repo/feature_store.yaml`: Feature store configuration

## 🚀 Deployment

### Docker Deployment
```bash
# Build container
docker build -t ds-workflow-template .

# Run container
docker run -p 8000:8000 ds-workflow-template
```

### Production Deployment
```bash
# Deploy to staging
make deploy-staging

# Run smoke tests
make test-staging

# Deploy to production
make deploy-production
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Follow the coding standards
4. **Run tests**: `make test`
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Write comprehensive tests
- Update documentation
- Use meaningful commit messages
- Run pre-commit hooks

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MLflow** for experiment tracking
- **Feast** for feature store capabilities
- **Great Expectations** for data validation
- **FastAPI** for API framework
- **pytest** for testing framework

## 📞 Support

For questions, issues, or contributions:
- **Email**: [rakeshk](mailto:rakeshk94@pm.me)

---

**Built with ❤️ for the Data Science Community**
