"""Feature building module with pandas/polars support and point-in-time join functionality.

This module provides utilities for building features with temporal consistency
and supports both pandas and polars for efficient data processing.
"""

import pandas as pd
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    
from datetime import datetime, timedelta
from typing import Union, List, Dict, Optional, Tuple
from abc import ABC, abstractmethod


class FeatureBuilder(ABC):
    """Abstract base class for feature builders."""
    
    @abstractmethod
    def build_features(self, df: Union[pd.DataFrame, 'pl.DataFrame']) -> Union[pd.DataFrame, 'pl.DataFrame']:
        """Build features from input dataframe."""
        pass


class BaseFeatureBuilder(FeatureBuilder):
    """Base feature builder with common transformations."""
    
    def __init__(self, timestamp_col: str = 'timestamp', entity_col: str = 'entity_id'):
        self.timestamp_col = timestamp_col
        self.entity_col = entity_col
    
    def build_features(self, df: Union[pd.DataFrame, 'pl.DataFrame']) -> Union[pd.DataFrame, 'pl.DataFrame']:
        """Build basic features."""
        if isinstance(df, pd.DataFrame):
            return self._build_pandas_features(df)
        elif HAS_POLARS and isinstance(df, pl.DataFrame):
            return self._build_polars_features(df)
        else:
            raise ValueError("Unsupported dataframe type or polars not installed")
    
    def _build_pandas_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features using pandas."""
        result = df.copy()
        
        # Convert timestamp if needed
        if self.timestamp_col in result.columns:
            result[self.timestamp_col] = pd.to_datetime(result[self.timestamp_col])
            
            # Add time-based features
            result['hour'] = result[self.timestamp_col].dt.hour
            result['day_of_week'] = result[self.timestamp_col].dt.dayofweek
            result['month'] = result[self.timestamp_col].dt.month
            result['quarter'] = result[self.timestamp_col].dt.quarter
            
            # Add lag features if entity column exists
            if self.entity_col in result.columns:
                result = result.sort_values([self.entity_col, self.timestamp_col])
                for col in result.select_dtypes(include=['number']).columns:
                    if col not in [self.timestamp_col, 'hour', 'day_of_week', 'month', 'quarter']:
                        result[f'{col}_lag1'] = result.groupby(self.entity_col)[col].shift(1)
                        result[f'{col}_lag7'] = result.groupby(self.entity_col)[col].shift(7)
        
        return result
    
    def _build_polars_features(self, df: 'pl.DataFrame') -> 'pl.DataFrame':
        """Build features using polars."""
        if not HAS_POLARS:
            raise ImportError("polars is required for polars operations")
            
        result = df.clone()
        
        # Convert timestamp if needed
        if self.timestamp_col in result.columns:
            result = result.with_columns(
                pl.col(self.timestamp_col).cast(pl.Datetime).alias(self.timestamp_col)
            )
            
            # Add time-based features
            result = result.with_columns([
                pl.col(self.timestamp_col).dt.hour().alias('hour'),
                pl.col(self.timestamp_col).dt.weekday().alias('day_of_week'),
                pl.col(self.timestamp_col).dt.month().alias('month'),
                pl.col(self.timestamp_col).dt.quarter().alias('quarter')
            ])
            
            # Add lag features if entity column exists
            if self.entity_col in result.columns:
                result = result.sort([self.entity_col, self.timestamp_col])
                numeric_cols = [col for col in result.columns 
                              if result[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                                                       pl.Float32, pl.Float64, pl.UInt8, pl.UInt16, 
                                                       pl.UInt32, pl.UInt64]]
                
                lag_exprs = []
                for col in numeric_cols:
                    if col not in ['hour', 'day_of_week', 'month', 'quarter']:
                        lag_exprs.extend([
                            pl.col(col).shift(1).over(self.entity_col).alias(f'{col}_lag1'),
                            pl.col(col).shift(7).over(self.entity_col).alias(f'{col}_lag7')
                        ])
                
                if lag_exprs:
                    result = result.with_columns(lag_exprs)
        
        return result


def point_in_time_join(
    left_df: Union[pd.DataFrame, 'pl.DataFrame'],
    right_df: Union[pd.DataFrame, 'pl.DataFrame'],
    entity_col: str,
    timestamp_col: str,
    tolerance: Optional[timedelta] = None
) -> Union[pd.DataFrame, 'pl.DataFrame']:
    """Perform point-in-time join to maintain temporal consistency.
    
    Args:
        left_df: Primary dataframe
        right_df: Feature dataframe to join
        entity_col: Column name for entity identifier
        timestamp_col: Column name for timestamp
        tolerance: Maximum time difference allowed for join
        
    Returns:
        Joined dataframe with point-in-time semantics
    """
    if isinstance(left_df, pd.DataFrame) and isinstance(right_df, pd.DataFrame):
        return _point_in_time_join_pandas(left_df, right_df, entity_col, timestamp_col, tolerance)
    elif HAS_POLARS and isinstance(left_df, pl.DataFrame) and isinstance(right_df, pl.DataFrame):
        return _point_in_time_join_polars(left_df, right_df, entity_col, timestamp_col, tolerance)
    else:
        raise ValueError("Both dataframes must be the same type (pandas or polars)")


def _point_in_time_join_pandas(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame, 
    entity_col: str,
    timestamp_col: str,
    tolerance: Optional[timedelta] = None
) -> pd.DataFrame:
    """Point-in-time join implementation for pandas."""
    # Ensure timestamp columns are datetime
    left_df[timestamp_col] = pd.to_datetime(left_df[timestamp_col])
    right_df[timestamp_col] = pd.to_datetime(right_df[timestamp_col])
    
    # Sort both dataframes
    left_sorted = left_df.sort_values([entity_col, timestamp_col])
    right_sorted = right_df.sort_values([entity_col, timestamp_col])
    
    # Perform asof join (point-in-time semantics)
    if tolerance:
        result = pd.merge_asof(
            left_sorted,
            right_sorted,
            on=timestamp_col,
            by=entity_col,
            tolerance=tolerance,
            direction='backward'
        )
    else:
        result = pd.merge_asof(
            left_sorted,
            right_sorted,
            on=timestamp_col,
            by=entity_col,
            direction='backward'
        )
    
    return result


def _point_in_time_join_polars(
    left_df: 'pl.DataFrame',
    right_df: 'pl.DataFrame',
    entity_col: str, 
    timestamp_col: str,
    tolerance: Optional[timedelta] = None
) -> 'pl.DataFrame':
    """Point-in-time join implementation for polars."""
    if not HAS_POLARS:
        raise ImportError("polars is required for polars operations")
        
    # Ensure timestamp columns are datetime
    left_df = left_df.with_columns(
        pl.col(timestamp_col).cast(pl.Datetime).alias(timestamp_col)
    )
    right_df = right_df.with_columns(
        pl.col(timestamp_col).cast(pl.Datetime).alias(timestamp_col)
    )
    
    # Sort both dataframes
    left_sorted = left_df.sort([entity_col, timestamp_col])
    right_sorted = right_df.sort([entity_col, timestamp_col])
    
    # Perform asof join
    if tolerance:
        tolerance_str = f"{tolerance.total_seconds()}s"
        result = left_sorted.join_asof(
            right_sorted,
            on=timestamp_col,
            by=entity_col,
            tolerance=tolerance_str,
            strategy='backward'
        )
    else:
        result = left_sorted.join_asof(
            right_sorted,
            on=timestamp_col,
            by=entity_col,
            strategy='backward'
        )
    
    return result


def build_features(
    df: Union[pd.DataFrame, 'pl.DataFrame'],
    feature_builders: Optional[List[FeatureBuilder]] = None,
    timestamp_col: str = 'timestamp',
    entity_col: str = 'entity_id'
) -> Union[pd.DataFrame, 'pl.DataFrame']:
    """Main entry point for building features.
    
    Args:
        df: Input dataframe
        feature_builders: List of feature builders to apply
        timestamp_col: Name of timestamp column
        entity_col: Name of entity column
        
    Returns:
        Dataframe with built features
    """
    if feature_builders is None:
        feature_builders = [BaseFeatureBuilder(timestamp_col, entity_col)]
    
    result = df
    for builder in feature_builders:
        result = builder.build_features(result)
    
    return result


# Example usage and demo functions
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'entity_id': [1, 1, 1, 2, 2, 2] * 5,
        'timestamp': pd.date_range('2023-01-01', periods=30, freq='D')[:30],
        'value': range(30),
        'feature_a': [x * 2 for x in range(30)]
    }
    
    df_pandas = pd.DataFrame(sample_data)
    
    # Build features
    builder = BaseFeatureBuilder()
    df_with_features = builder.build_features(df_pandas)
    
    print("Feature building completed.")
    print(f"Original columns: {list(df_pandas.columns)}")
    print(f"New columns: {list(df_with_features.columns)}")
    
    # Demo point-in-time join
    left_df = df_pandas.iloc[:15].copy()
    right_df = df_pandas.iloc[10:].copy() 
    right_df = right_df.rename(columns={'feature_a': 'feature_b'})
    
    joined_df = point_in_time_join(left_df, right_df, 'entity_id', 'timestamp')
    print(f"\nPoint-in-time join completed. Result shape: {joined_df.shape}")
