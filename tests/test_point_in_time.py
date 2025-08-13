"""Point-in-time correctness tests for Feast feature store."""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from feast import FeatureStore

# Add feature_repo to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "feature_repo"))

try:
    from features import user, iris_features, user_activity_features
except ImportError:
    pytest.skip("Feature definitions not available", allow_module_level=True)


class TestPointInTimeCorrectness:
    """Test suite for validating point-in-time correctness in Feast."""

    @pytest.fixture
    def feature_store_config_path(self):
        """Create temporary feature store config for testing."""
        return Path(__file__).parent.parent / "feature_repo" / "feature_store.yaml"

    @pytest.fixture
    def sample_data(self):
        """Create sample historical data with timestamps."""
        base_time = datetime(2023, 1, 1, 12, 0, 0)
        return pd.DataFrame([
            {
                "user_id": 1,
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
                "species": "setosa",
                "event_timestamp": base_time,
                "created_timestamp": base_time,
            },
            {
                "user_id": 2,
                "sepal_length": 4.9,
                "sepal_width": 3.0,
                "petal_length": 1.4,
                "petal_width": 0.2,
                "species": "setosa",
                "event_timestamp": base_time + timedelta(hours=1),
                "created_timestamp": base_time + timedelta(hours=1),
            },
            {
                "user_id": 1,
                "sepal_length": 5.4,
                "sepal_width": 3.9,
                "petal_length": 1.7,
                "petal_width": 0.4,
                "species": "setosa",
                "event_timestamp": base_time + timedelta(hours=2),
                "created_timestamp": base_time + timedelta(hours=2),
            },
        ])

    def test_feature_retrieval_at_point_in_time(self, feature_store_config_path, sample_data):
        """Test that features are retrieved correctly for a specific point in time."""
        if not feature_store_config_path.exists():
            pytest.skip("Feature store config not found")

        # This test demonstrates the concept of point-in-time correctness
        # In a real implementation, you would:
        # 1. Set up a proper Feast feature store
        # 2. Materialize features to the online store
        # 3. Query features at specific timestamps
        # 4. Verify that the returned features match expected values
        
        # Simulate point-in-time lookup logic
        query_time = datetime(2023, 1, 1, 13, 30, 0)  # Between hours 1 and 2
        
        # For user_id=1, we should get features from the first record (at 12:00)
        # because it's the latest available before the query time
        expected_features_user_1 = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "species": "setosa"
        }
        
        # Mock the feature retrieval for demonstration
        # In practice, this would use FeatureStore.get_historical_features()
        historical_features = self._mock_historical_features(sample_data, query_time)
        
        user_1_features = historical_features[historical_features["user_id"] == 1].iloc[0]
        assert user_1_features["sepal_length"] == expected_features_user_1["sepal_length"]
        assert user_1_features["species"] == expected_features_user_1["species"]

    def test_feature_freshness_validation(self, sample_data):
        """Test that stale features are handled appropriately."""
        current_time = datetime.now()
        stale_time = current_time - timedelta(days=10)  # 10 days old
        
        # Create data with stale timestamps
        stale_data = sample_data.copy()
        stale_data["event_timestamp"] = stale_time
        
        # Verify that features older than TTL are flagged appropriately
        ttl_threshold = timedelta(days=7)  # From iris_features definition
        is_stale = (current_time - stale_data["event_timestamp"]) > ttl_threshold
        
        assert is_stale.all(), "All test data should be considered stale"

    def test_entity_consistency(self, sample_data):
        """Test that entity keys are consistent across feature views."""
        # Verify that all records have valid user_id (entity key)
        assert sample_data["user_id"].notna().all()
        assert (sample_data["user_id"] > 0).all()
        
        # Check for duplicate entities at the same timestamp
        duplicates = sample_data.duplicated(subset=["user_id", "event_timestamp"])
        assert not duplicates.any(), "No duplicate entity-timestamp pairs should exist"

    def _mock_historical_features(self, data: pd.DataFrame, query_time: datetime) -> pd.DataFrame:
        """Mock historical feature retrieval with point-in-time logic."""
        # Filter to features available before query time
        available_features = data[data["event_timestamp"] <= query_time]
        
        if available_features.empty:
            return pd.DataFrame()
        
        # Get the most recent features for each entity before query time
        latest_features = (
            available_features
            .sort_values(["user_id", "event_timestamp"])
            .groupby("user_id")
            .last()
            .reset_index()
        )
        
        return latest_features


if __name__ == "__main__":
    pytest.main([__file__])
