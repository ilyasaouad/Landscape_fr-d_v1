# Patent Data Analysis Module

A modular Python package for analyzing patent applicant and inventor data from EPO/DOCDB databases.

## Features

- **Modular Design**: Clean separation of concerns with specialized modules
- **Flexible Data Retrieval**: Fetch patent family data by country and year ranges
- **Statistical Analysis**: Calculate applicant/inventor counts and ratios
- **Batch Processing**: Efficient handling of large datasets
- **Data Validation**: Comprehensive input validation
- **Export Options**: CSV export with configurable naming
- **Logging**: Detailed logging for monitoring and debugging

## Architecture

```
patent_analysis/
├── __init__.py          # Module interface and exports
├── config.py            # Configuration settings
├── database.py          # Database connection and sessions
├── models.py            # SQLAlchemy data models
├── validators.py        # Input validation utilities
├── data_retrieval.py    # Data fetching functions
├── data_analysis.py     # Statistical analysis functions
├── main.py              # Main orchestration functions
├── requirements.txt     # Python dependencies
├── usage_example.py     # Usage examples
└── README.md           # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure database connection in `config.py`

3. Run the example:
```bash
python usage_example.py
```

## Quick Start

```python
from patent_analysis import run_full_analysis

# Run analysis for US patents 2020-2022
family_ids, person_data, results = run_full_analysis(
    country_code="US",
    start_year=2020,
    end_year=2022,
    range_limit=1000  # Optional: limit for testing
)

print(f"Found {len(family_ids)} patent families")
print(f"Analysis results: {len(results)} records")
```

## API Reference

### Main Functions

#### `run_full_analysis()`
Complete patent analysis pipeline.

**Parameters:**
- `country_code`: Two-letter ISO country code (e.g., "US", "DE")
- `start_year`: Start year of filing date range
- `end_year`: End year of filing date range
- `range_limit`: Optional limit on number of families to process
- `save_results`: Whether to save results to CSV files
- `output_dir`: Output directory for results

**Returns:** Tuple of (family_ids, person_data, results) DataFrames

#### `get_applicants_inventors_data()`
Fetch patent family IDs and person details.

**Parameters:**
- `country_code`: Two-letter ISO country code
- `start_year`: Start year
- `end_year`: End year
- `range_limit`: Optional limit on families

**Returns:** Tuple of (family_ids, person_details) DataFrames

### Analysis Functions

#### `calculate_applicants_inventors_counts()`
Calculate applicant and inventor counts per family/country.

#### `calculate_applicants_inventors_ratios()`
Calculate proportions/ratios from counts.

#### `merge_counts_and_ratios()`
Combine all counts and ratios into final dataset.

### Utility Functions

#### Input Validation
- `validate_country_code()`: Validate and normalize country codes
- `validate_year_range()`: Validate year ranges
- `validate_family_ids()`: Validate family ID lists
- `validate_dataframe()`: Validate DataFrame structure

#### Database Management
- `create_sqlalchemy_session()`: Context manager for DB sessions
- `get_database_info()`: Get connection pool information

## Configuration

Edit `config.py` to customize:

- **Database Settings**: Connection parameters
- **Batch Sizes**: Data processing batch sizes
- **Output Locations**: File output directories
- **Logging**: Log levels and formats

## Data Output

The module generates three CSV files per analysis:

1. `{country}_{start}_{end}_*_family_ids.csv` - Patent family IDs
2. `{country}_{start}_{end}_*_person_data.csv` - Raw applicant/inventor data  
3. `{country}_{start}_{end}_*_analysis_results.csv` - Final analysis with counts/ratios

## Input Data Structure

Expected database tables:
- `tls201_appln` - Patent applications
- `tls206_person` - Persons (applicants/inventors)
- `tls207_pers_appln` - Person-application relationships

## Error Handling

- Comprehensive input validation with descriptive error messages
- Database connection error handling
- Batch processing continues on individual batch failures
- Detailed logging at all stages

## Performance

- **Batch Processing**: Large datasets split into configurable batches
- **Efficient Queries**: Optimized SQLAlchemy queries with proper joins
- **Memory Management**: Streaming processing to avoid memory issues
- **Connection Pooling**: Database connection reuse

## Dependencies

- **pandas**: Data manipulation and analysis
- **sqlalchemy**: Database ORM and queries
- **psycopg2-binary**: PostgreSQL adapter
- **python-dotenv**: Environment variable management

## Contributing

1. Follow the modular structure
2. Add comprehensive docstrings
3. Include input validation for new functions
4. Add logging for important operations
5. Update this README for new features

## License

This module is provided as-is for patent data analysis purposes.# Landscape_fr-d_v1
