"""Data sources configuration for Feast feature store."""

from datetime import datetime, timedelta
from pathlib import Path

from feast import FileSource
from feast.data_format import ParquetFormat

# Data path configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "sample"

# Iris dataset batch source
iris_source = FileSource(
    name="iris_source",
    path=str(DATA_DIR / "iris.csv"),  # Uses existing iris data
    description="Iris dataset source for feature engineering example",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Alternative: Synthetic user activity data source
# This would be created if we had synthetic user data
user_activity_source = FileSource(
    name="user_activity_source", 
    path=str(DATA_DIR / "user_activity.parquet"),
    description="Synthetic user activity data for demonstration",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Historical context for point-in-time correctness
# This demonstrates how to set up data sources with proper timestamps
# for temporal feature lookups in Feast
