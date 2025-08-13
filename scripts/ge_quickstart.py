#!/usr/bin/env python3
"""
Great Expectations Quickstart Script

This script demonstrates how to initialize and use Great Expectations
with a filesystem-based data context for data validation.

Usage:
    python scripts/ge_quickstart.py

Requirements:
    - great-expectations>=0.18.0
    - pandas
    - A sample CSV file in the ./data/ directory
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import great_expectations as gx
    from great_expectations.data_context import FileDataContext
    import pandas as pd
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install Great Expectations: pip install great-expectations")
    sys.exit(1)


def create_sample_data():
    """
    Create a sample CSV file for demonstration purposes.
    """
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    sample_file = data_dir / "sample_users.csv"
    
    if not sample_file.exists():
        print(f"Creating sample data file: {sample_file}")
        
        # Create sample data
        sample_data = {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice Johnson", "Bob Smith", "Carol Davis", "David Wilson", "Eva Brown"],
            "age": [28, 35, 42, 29, 31],
            "email": ["alice@example.com", "bob@example.com", "carol@example.com", 
                     "david@example.com", "eva@example.com"],
            "created_date": ["2023-01-15", "2023-02-20", "2023-03-10", 
                            "2023-04-05", "2023-05-12"]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(sample_file, index=False)
        print(f"Sample data created with {len(df)} rows")
    else:
        print(f"Sample data file already exists: {sample_file}")
    
    return sample_file


def initialize_data_context():
    """
    Initialize Great Expectations data context using the existing gx.yaml configuration.
    """
    ge_dir = project_root / "great_expectations"
    
    if not ge_dir.exists():
        print(f"Error: Great Expectations directory not found at {ge_dir}")
        print("Please ensure the great_expectations/ directory and gx.yaml file exist.")
        return None
    
    try:
        print(f"Initializing Data Context from: {ge_dir}")
        context = FileDataContext(context_root_dir=str(ge_dir))
        print("✓ Data Context initialized successfully")
        return context
    except Exception as e:
        print(f"Error initializing Data Context: {e}")
        return None


def validate_data_with_suite(context, data_file):
    """
    Validate data using the sample_csv_suite expectations.
    """
    try:
        # Get the datasource
        datasource_name = "sample_data"
        datasource = context.get_datasource(datasource_name)
        print(f"✓ Retrieved datasource: {datasource_name}")
        
        # Create a batch request
        batch_request = {
            "datasource_name": datasource_name,
            "data_connector_name": "default_filesystem_connector",
            "data_asset_name": data_file.stem,  # filename without extension
        }
        
        print(f"Creating batch request for: {data_file.name}")
        
        # Get validator
        expectation_suite_name = "sample_csv_suite"
        validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=expectation_suite_name
        )
        print(f"✓ Created validator with expectation suite: {expectation_suite_name}")
        
        # Run validation
        print("\nRunning data validation...")
        results = validator.validate()
        
        # Print results
        print(f"\n{'='*50}")
        print("VALIDATION RESULTS")
        print(f"{'='*50}")
        print(f"Total Expectations: {results.statistics['evaluated_expectations']}")
        print(f"Successful: {results.statistics['successful_expectations']}")
        print(f"Failed: {results.statistics['unsuccessful_expectations']}")
        print(f"Success Rate: {results.statistics['success_percent']:.1f}%")
        
        if results.statistics['unsuccessful_expectations'] > 0:
            print("\nFailed Expectations:")
            for result in results.results:
                if not result['success']:
                    expectation_type = result['expectation_config']['expectation_type']
                    print(f"  - {expectation_type}: {result.get('exception_info', {}).get('exception_message', 'Failed')}")
        else:
            print("\n✓ All expectations passed!")
        
        return results
        
    except Exception as e:
        print(f"Error during validation: {e}")
        return None


def main():
    """
    Main function to demonstrate Great Expectations usage.
    """
    print("Great Expectations Quickstart Demo")
    print("=" * 40)
    
    # Step 1: Create sample data
    print("\n1. Creating sample data...")
    sample_file = create_sample_data()
    
    # Step 2: Initialize Data Context
    print("\n2. Initializing Data Context...")
    context = initialize_data_context()
    if not context:
        return
    
    # Step 3: List available expectation suites
    print("\n3. Available Expectation Suites:")
    suites = context.list_expectation_suite_names()
    for suite in suites:
        print(f"  - {suite}")
    
    # Step 4: Validate data
    print("\n4. Validating data...")
    results = validate_data_with_suite(context, sample_file)
    
    if results:
        print("\n5. Next Steps:")
        print("  - Review the validation results above")
        print("  - Modify expectations in great_expectations/expectations/sample_csv_suite.json")
        print("  - Add more data files to ./data/ directory")
        print("  - Create additional expectation suites for different data types")
        print("  - Set up automated validation pipelines")
        
        # Show how to generate Data Docs (optional)
        try:
            print("\n6. Building Data Docs...")
            context.build_data_docs()
            docs_path = project_root / "great_expectations" / "docs" / "local_site" / "index.html"
            if docs_path.exists():
                print(f"✓ Data Docs generated at: {docs_path}")
                print(f"  Open in browser: file://{docs_path.absolute()}")
            else:
                print("Data Docs location not found")
        except Exception as e:
            print(f"Note: Could not build Data Docs: {e}")
    
    print("\nQuickstart completed!")


if __name__ == "__main__":
    main()
