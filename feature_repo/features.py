"""Feature definitions for Feast feature store."""

from datetime import timedelta

from feast import Entity, Feature, FeatureView, ValueType
from feast.types import Float32, Int64, String

from data_sources import iris_source, user_activity_source

# Entity definitions
user = Entity(
    name="user",
    description="User entity for feature engineering",
    value_type=ValueType.INT64,
)

# Example: Iris features (using existing data)
iris_features = FeatureView(
    name="iris_features",
    description="Features derived from the iris dataset",
    entities=["user"],  # Mapped to user for demonstration
    ttl=timedelta(days=7),
    schema=[
        Feature(name="sepal_length", dtype=Float32),
        Feature(name="sepal_width", dtype=Float32),
        Feature(name="petal_length", dtype=Float32),
        Feature(name="petal_width", dtype=Float32),
        Feature(name="species", dtype=String),
    ],
    source=iris_source,
)

# Example: User activity features (synthetic data)
user_activity_features = FeatureView(
    name="user_activity_features",
    description="User behavioral features for ML models",
    entities=[user],
    ttl=timedelta(days=30),
    schema=[
        Feature(name="total_sessions", dtype=Int64),
        Feature(name="avg_session_duration", dtype=Float32),
        Feature(name="page_views", dtype=Int64),
        Feature(name="conversion_rate", dtype=Float32),
        Feature(name="days_since_last_activity", dtype=Int64),
    ],
    source=user_activity_source,
)

# Feature views enable point-in-time correctness for historical training data
# and serve fresh features for online inference
