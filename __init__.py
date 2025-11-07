"""
Patent Data Analysis Module
A modular system for analyzing patent applicants and inventors data.

Updated to work with SQL Server and environment variable configuration.
"""

from config import Config
from database import create_sqlalchemy_session, get_database_info, test_database_connection
from data_retrieval_applicants_inventors import get_applicants_inventors_data
from data_analysis_applicants_inventors import (
    calculate_applicants_inventors_counts,
    calculate_applicants_inventors_ratios,
    merge_counts_and_ratios
)
from main import run_full_analysis, get_analysis_summary
from validators import (
    validate_country_code,
    validate_year_range,
    validate_family_ids,
    validate_dataframe
)
from services import FamilyService, ApplicantService, AnalysisService

__version__ = "2.0.0"
__author__ = "MiniMax Agent"

__all__ = [
    # Config
    "Config",
    # Database
    "create_sqlalchemy_session",
    "get_database_info",
    "test_database_connection",
    # Data retrieval
    "get_family_ids",
    "get_applicant_inventor", 
    "get_applicants_inventors_data",
    # Analysis
    "calculate_applicants_inventors_counts",
    "calculate_applicants_inventors_ratios",
    "merge_counts_and_ratios",
    # Main functions
    "run_full_analysis",
    "get_analysis_summary",
    # Validators
    "validate_country_code",
    "validate_year_range",
    "validate_family_ids",
    "validate_dataframe",
    # Services
    "FamilyService",
    "ApplicantService", 
    "AnalysisService"
]