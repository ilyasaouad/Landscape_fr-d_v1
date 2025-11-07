"""
Input validation utilities for the patent analysis system.
"""

import logging
from typing import Any, Union
import pandas as pd

logger = logging.getLogger(__name__)

def validate_country_code(country_code: str) -> str:
    """
    Validate and normalize country code.
    
    Args:
        country_code: Two-letter ISO country code
        
    Returns:
        str: Normalized country code
        
    Raises:
        ValueError: If country code is invalid
    """
    if not isinstance(country_code, str):
        raise ValueError("Country code must be a string")
    
    if len(country_code) != 2:
        raise ValueError("Country code must be exactly 2 characters long")
    
    if not country_code.isalpha():
        raise ValueError("Country code must contain only letters")
    
    return country_code.upper()

def validate_year_range(start_year: int, end_year: int) -> tuple[int, int]:
    """
    Validate year range for patent analysis.
    
    Args:
        start_year: Start year of range
        end_year: End year of range
        
    Returns:
        tuple: Normalized (start_year, end_year)
        
    Raises:
        ValueError: If year range is invalid
    """
    current_year = 2025
    
    # Validate individual years
    if not isinstance(start_year, int) or not isinstance(end_year, int):
        raise ValueError("Years must be integers")
    
    if not (1900 <= start_year <= current_year):
        raise ValueError(f"Start year must be between 1900 and {current_year}")
    
    if not (start_year <= end_year <= current_year):
        raise ValueError(f"End year must be >= start year and <= {current_year}")
    
    return start_year, end_year

def validate_family_ids(family_ids_list: list[int]) -> list[int]:
    """
    Validate family IDs list.
    
    Args:
        family_ids_list: List of family IDs
        
    Returns:
        list: Validated and deduplicated family IDs
        
    Raises:
        ValueError: If family IDs are invalid
    """
    if not isinstance(family_ids_list, list):
        raise ValueError("Family IDs must be a list")
    
    if not family_ids_list:
        raise ValueError("Family IDs list cannot be empty")
    
    # Check all items are integers
    if not all(isinstance(fid, int) for fid in family_ids_list):
        raise ValueError("All family IDs must be integers")
    
    # Remove duplicates and negative values
    valid_ids = [fid for fid in set(family_ids_list) if fid > 0]
    
    if not valid_ids:
        raise ValueError("No valid family IDs found")
    
    return sorted(valid_ids)

def validate_dataframe(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """
    Validate DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        pd.DataFrame: Validated DataFrame
        
    Raises:
        ValueError: If DataFrame is missing required columns
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return df

def sanitize_string(value: str) -> str:
    """
    Sanitize string values for database operations.
    
    Args:
        value: String to sanitize
        
    Returns:
        str: Sanitized string
    """
    if not isinstance(value, str):
        return str(value)
    
    # Remove null characters and normalize unicode
    sanitized = value.replace('\x00', '').strip()
    return sanitized