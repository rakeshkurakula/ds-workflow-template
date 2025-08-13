#!/usr/bin/env python3
"""
Data Validation Script for Sample Dataset

This script validates the sample iris dataset using Great Expectations.
It's designed to be used in CI pipelines to ensure data quality.

Usage:
    python scripts/validate_sample.py

Returns:
    Exit code 0 on success, 1 on validation failure
"""

import sys
import os
from pathlib import Path

try:
    import great_expectations as gx
    from great_expectations.core import ExpectationSuite
    from great_expectations.dataset import PandasDataset
except ImportError:
    print("Error: Great Expectations not installed.")
    print("Install with: pip install great-expectations")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Error: pandas not installed.")
    print("Install with: pip install pandas")
    sys.exit(1)


def validate_iris_dataset(data_path: str) -> bool:
    """
    Validate the iris dataset using Great Expectations.
    
    Args:
        data_path: Path to the CSV file to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return False
    
    try:
        # Load the dataset
        df = pd.read_csv(data_path)
        
        # Convert to Great Expectations dataset
        ge_df = PandasDataset(df)
        
        # Define expectations for the iris dataset
        print("Validating dataset structure and content...")
        
        # Basic structure expectations
        result1 = ge_df.expect_table_columns_to_match_ordered_list([
            "sepal_length", "sepal_width", "petal_length", "petal_width", "species"
        ])
        
        # Data type expectations
        result2 = ge_df.expect_column_values_to_be_of_type("sepal_length", "float64")
        result3 = ge_df.expect_column_values_to_be_of_type("sepal_width", "float64")
        result4 = ge_df.expect_column_values_to_be_of_type("petal_length", "float64")
        result5 = ge_df.expect_column_values_to_be_of_type("petal_width", "float64")
        result6 = ge_df.expect_column_values_to_be_of_type("species", "object")
        
        # Value range expectations for numerical columns
        result7 = ge_df.expect_column_values_to_be_between("sepal_length", 0, 10)
        result8 = ge_df.expect_column_values_to_be_between("sepal_width", 0, 10)
        result9 = ge_df.expect_column_values_to_be_between("petal_length", 0, 10)
        result10 = ge_df.expect_column_values_to_be_between("petal_width", 0, 10)
        
        # Categorical expectations for species
        result11 = ge_df.expect_column_values_to_be_in_set("species", 
                                                          ["setosa", "versicolor", "virginica"])
        
        # No null values expectations
        result12 = ge_df.expect_column_values_to_not_be_null("sepal_length")
        result13 = ge_df.expect_column_values_to_not_be_null("sepal_width")
        result14 = ge_df.expect_column_values_to_not_be_null("petal_length")
        result15 = ge_df.expect_column_values_to_not_be_null("petal_width")
        result16 = ge_df.expect_column_values_to_not_be_null("species")
        
        # Row count expectations (iris should have reasonable number of rows)
        result17 = ge_df.expect_table_row_count_to_be_between(10, 500)
        
        # Collect all results
        results = [result1, result2, result3, result4, result5, result6, result7, 
                  result8, result9, result10, result11, result12, result13, result14, 
                  result15, result16, result17]
        
        # Check if all expectations passed
        all_passed = all(result.success for result in results)
        
        if all_passed:
            print("✓ All data validation checks passed!")
            print(f"  Dataset shape: {df.shape}")
            print(f"  Species distribution: {df['species'].value_counts().to_dict()}")
        else:
            print("✗ Data validation failed!")
            for i, result in enumerate(results, 1):
                if not result.success:
                    print(f"  Failed check {i}: {result.expectation_config.expectation_type}")
                    print(f"    Details: {result.result}")
        
        return all_passed
        
    except Exception as e:
        print(f"Error during validation: {e}")
        return False


def main():
    """
    Main function to run the validation.
    """
    # Define the path to the sample dataset
    repo_root = Path(__file__).parent.parent
    data_path = repo_root / "data" / "sample" / "iris.csv"
    
    print(f"Validating dataset: {data_path}")
    
    # Run validation
    validation_passed = validate_iris_dataset(str(data_path))
    
    if validation_passed:
        print("\n🎉 Data validation completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Data validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
