#!/usr/bin/env python3
"""
Test script for the get_family_ids, get_main_table, and get_applicants_inventors functions.
"""

import logging
import sys
from pathlib import Path

# Add the project root to the Python path to allow imports
# This assumes the script is run from the project's root directory
try:
    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root))
except NameError:
    # __file__ is not defined, happens in some environments like notebooks
    # Assume the current working directory is the project root
    project_root = Path.cwd()
    sys.path.insert(0, str(project_root))


# --- Imports ---
from config import Config
from get_family_ids import get_family_ids
from get_main_table import get_main_table
from get_applicants_inventors import get_applicants_inventors_data  # <-- NEW IMPORT

# --- Test Configuration ---
COUNTRY_CODE = "NO"
START_YEAR = 2020
END_YEAR = 2020

# --- Main Test Execution ---
if __name__ == "__main__":
    # Setup logging to see output from the functions
    logging.basicConfig(
        level=logging.INFO,  # Use INFO level for detailed output
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info(
        f"Running tests for country '{COUNTRY_CODE}' from {START_YEAR} to {END_YEAR}"
    )

    # ==============================================================================
    # TEST 1: get_family_ids
    # ==============================================================================
    logger.info("--- Starting Test 1: get_family_ids ---")
    try:
        df_family_ids = get_family_ids(COUNTRY_CODE, START_YEAR, END_YEAR)

        if df_family_ids.empty:
            logger.warning(
                f"Test 1 PASSED (but no data): No family IDs found for the criteria."
            )
        else:
            logger.info(f"Test 1 PASSED: Found {len(df_family_ids)} unique family IDs.")
            print("\n--- Family IDs DataFrame Head ---")
            print(df_family_ids.head())
            print(f"\nShape: {df_family_ids.shape}\n")

    except Exception as e:
        logger.error(f"Test 1 FAILED: An error occurred in get_family_ids.")
        logger.error(f"Error: {e}", exc_info=True)
        # Exit if the first test fails, as the second depends on it
        sys.exit(1)

    # ==============================================================================
    # TEST 2: get_main_table
    # ==============================================================================
    # Only run this test if the first one returned data
    if not df_family_ids.empty:
        logger.info("--- Starting Test 2: get_main_table ---")
        try:
            family_ids_list = df_family_ids["docdb_family_id"].tolist()
            df_main_table = get_main_table(
                family_ids_list=family_ids_list,
                country_code=COUNTRY_CODE,
                start_year=START_YEAR,
                end_year=END_YEAR,
                save_to_csv=False,  # Ensure no file is saved during the test
                output_dir=None,
            )

            if df_main_table.empty:
                logger.warning(
                    f"Test 2 PASSED (but no data): No main table data found for the given family IDs."
                )
            else:
                logger.info(
                    f"Test 2 PASSED: Retrieved main table with {len(df_main_table)} rows."
                )
                print("\n--- Main Table DataFrame Head ---")
                print(df_main_table.head())
                print(f"\nShape: {df_main_table.shape}")
                print("\n--- Column Names ---")
                print(df_main_table.columns.tolist())
                print()

        except Exception as e:
            logger.error(f"Test 2 FAILED: An error occurred in get_main_table.")
            logger.error(f"Error: {e}", exc_info=True)
            sys.exit(1)

    # ==============================================================================
    # TEST 3: get_applicants_inventors
    # ==============================================================================
    logger.info("--- Starting Test 3: get_applicants_inventors_data ---")
    try:
        # This function is self-contained and fetches family IDs internally.
        # We test it with the same parameters to validate its full workflow.
        df_fam_ids_ai, df_applicants_inventors = get_applicants_inventors_data(
            COUNTRY_CODE, START_YEAR, END_YEAR
        )

        if df_fam_ids_ai.empty or df_applicants_inventors.empty:
            logger.warning(
                f"Test 3 PASSED (but no data): No applicant/inventor data found for the criteria."
            )
        else:
            logger.info(
                f"Test 3 PASSED: Found {len(df_fam_ids_ai)} family IDs and {len(df_applicants_inventors)} applicant/inventor records."
            )
            print("\n--- Applicants/Inventors DataFrame Head ---")
            print(df_applicants_inventors.head())
            print(f"\nShape: {df_applicants_inventors.shape}")
            print("\n--- Column Names ---")
            print(df_applicants_inventors.columns.tolist())
            print()

    except Exception as e:
        logger.error(
            f"Test 3 FAILED: An error occurred in get_applicants_inventors_data."
        )
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

    logger.info("--- All tests completed successfully ---")
