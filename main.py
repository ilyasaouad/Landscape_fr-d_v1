"""
Main module for patent analysis - orchestrates all functionality.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from config import Config
from get_family_ids import get_family_ids
from get_main_table import get_main_table
from get_classes import get_classes
from get_applicants_inventors import get_applicants_inventors_data
from data_analysis_applicants_inventors import (
    calculate_applicants_inventors_counts,
    merge_applicants_and_inventors,
    calculate_applicants_inventors_ratios,
    merge_all_ratios,
)
from validators import validate_country_code, validate_year_range

logger = logging.getLogger(__name__)


def run_full_analysis(
    country_code: str,
    start_year: int,
    end_year: int,
    range_limit: Optional[int] = None,
    save_results: bool = True,
    output_dir: Optional[Path] = None,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Run complete patent analysis pipeline with counts and ratios.

    Args:
        country_code: Two-letter ISO country code
        start_year: Start year of analysis range
        end_year: End year of analysis range
        range_limit: Optional limit on number of families to process
        save_results: Whether to save results to CSV files
        output_dir: Optional custom output directory (defaults to same folder as main.py)

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - Family IDs DataFrame
            - Applicants/Inventors data DataFrame
            - Main table DataFrame
            - Counts analysis results DataFrame
            - Ratios analysis results DataFrame
            - Merged counts and ratios DataFrame

    Raises:
        ValueError: If input parameters are invalid
    """
    logger.info(
        f"Starting patent analysis for {country_code} ({start_year}-{end_year})"
    )

    # Validate inputs
    country_code = validate_country_code(country_code)
    start_year, end_year = validate_year_range(start_year, end_year)

    # Create output directory
    base_dir = Path(__file__).resolve().parent
    target_base = output_dir or base_dir
    dir_name = f"DataTables_{country_code}_{start_year}_{end_year}"
    target_dir = target_base / dir_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get family IDs and save to CSV
    logger.info("Step 1: Getting family IDs...")
    df_family_ids = get_family_ids(country_code, start_year, end_year)

    if df_family_ids.empty:
        logger.warning("No data found - returning empty DataFrames")
        return (
            df_family_ids,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    # Apply range limit if specified
    if range_limit is not None and range_limit > 0:
        df_family_ids = df_family_ids.head(range_limit)
        logger.info(f"Limited to {range_limit} family IDs")

    # Save the family IDs to a CSV file. This file will be the input for the next steps.
    family_ids_path = target_dir / "family_ids.csv"
    df_family_ids.to_csv(family_ids_path, index=False)
    logger.info(f"Saved family IDs to {family_ids_path}")

    # Step 2: Get main table data using the saved file
    logger.info("Step 2: Getting main table data...")
    # Call the simplified function. It only needs the file path.
    df_main_table = get_main_table(
        family_ids_path, range_limit=range_limit  # Pass the range_limit here if needed
    )

    # Step 3: Get class data using the saved file
    logger.info("Step 3: Getting class data...")
    df_classes = get_classes(family_ids_path, range_limit=range_limit)

    # Step 4: Merge class data into the main table
    if not df_main_table.empty and not df_classes.empty:
        logger.info("Merging class data into main table...")
        df_main_table = pd.merge(
            df_main_table, df_classes, on="docdb_family_id", how="left"
        )
        logger.info("Class data merged successfully.")

    # Step 5: Save the enriched main table data if requested
    if save_results and not df_main_table.empty:
        main_table_path = target_dir / "main_table.csv"
        df_main_table.to_csv(main_table_path, index=False)
        logger.info(f"Saved enriched main table data to {main_table_path}")

    # Step 4: Get applicants/inventors data using the same saved file
    logger.info("Step 4: Getting applicants/inventors data...")
    df_family_ids_read, df_applicants_inventors_data = get_applicants_inventors_data(
        family_ids_path
    )

    # Step 5: Calculate counts per family-country
    logger.info("Step 5: Calculating applicant and inventor counts...")
    df_applicant_counts, df_inventor_counts = calculate_applicants_inventors_counts(
        df_applicants_inventors_data
    )

    # Step 6: Merge counts into single dataset
    logger.info("Step 6: Merging applicant and inventor counts...")
    df_counts_result = merge_applicants_and_inventors(
        df_applicant_counts, df_inventor_counts
    )

    # Step 7: Calculate combined count (applicants + inventors)
    logger.info("Step 7: Creating combined counts...")
    df_combined_counts = df_counts_result.copy()
    df_combined_counts["combined_count"] = (
        df_combined_counts["applicant_count"] + df_combined_counts["inventor_count"]
    )

    # Step 8: Calculate ratios per family-country
    logger.info("Step 8: Calculating ratios...")
    df_applicant_ratios, df_inventor_ratios, df_combined_ratios = (
        calculate_applicants_inventors_ratios(
            df_applicant_counts,
            df_inventor_counts,
            df_combined_counts[
                ["docdb_family_id", "person_ctry_code", "combined_count"]
            ],
        )
    )

    # Step 9: Create ratios result dataframe
    logger.info("Step 9: Merging ratios...")
    df_ratios_result = df_applicant_ratios.copy()
    df_ratios_result = pd.merge(
        df_ratios_result,
        df_inventor_ratios,
        on=["docdb_family_id", "person_ctry_code"],
        how="left",
    )
    df_ratios_result = pd.merge(
        df_ratios_result,
        df_combined_ratios,
        on=["docdb_family_id", "person_ctry_code"],
        how="left",
    )
    df_ratios_result = df_ratios_result.fillna(0)

    # Round ratios to 4 decimal places
    for col in ["applicant_ratio", "inventor_ratio", "combined_ratio"]:
        if col in df_ratios_result.columns:
            df_ratios_result[col] = df_ratios_result[col].round(4)

    # Step 10: Merge counts and ratios into comprehensive dataset
    logger.info("Step 10: Merging counts and ratios...")
    df_merge_result = merge_all_ratios(
        df_applicant_counts,
        df_inventor_counts,
        df_combined_counts[["docdb_family_id", "person_ctry_code", "combined_count"]],
        df_applicant_ratios,
        df_inventor_ratios,
        df_combined_ratios,
    )

    # Step 11: Save remaining results if requested
    if save_results:
        logger.info("Step 11: Saving remaining results...")
        _save_analysis_results(
            df_family_ids,
            df_applicants_inventors_data,
            df_main_table,
            df_counts_result,
            df_ratios_result,
            df_merge_result,
            country_code,
            start_year,
            end_year,
            output_dir,
        )

    logger.info(
        f"Analysis complete. Processed {len(df_family_ids)} families, "
        f"{len(df_applicants_inventors_data)} applicant/inventor records, "
        f"{len(df_main_table)} main table records, "
        f"{len(df_counts_result)} count results, "
        f"{len(df_ratios_result)} ratio results"
    )

    return (
        df_family_ids,
        df_applicants_inventors_data,
        df_main_table,
        df_counts_result,
        df_ratios_result,
        df_merge_result,
    )


def _save_analysis_results(
    df_family_ids: pd.DataFrame,
    df_applicants_inventors_data: pd.DataFrame,
    df_main_table: pd.DataFrame,
    df_counts_results: pd.DataFrame,
    df_ratios_results: pd.DataFrame,
    df_merge_results: pd.DataFrame,
    country_code: str,
    start_year: int,
    end_year: int,
    output_dir: Optional[Path],
) -> None:
    """
    Save analysis results to a structured directory.

    Note: The main_table.csv is saved directly in run_full_analysis to handle
    its specific saving logic cleanly.

    Args:
        df_family_ids: Family IDs dataframe
        df_applicants_inventors_data: Extracted raw applicant/inventor data
        df_main_table: Main application data
        df_counts_results: Applicant/inventor counts per family-country
        df_ratios_results: Applicant/inventor ratios per family-country
        df_merge_results: Merged counts and ratios per family-country
        country_code: Country code for directory naming
        start_year: Start year for directory naming
        end_year: End year for directory naming
        output_dir: Optional custom base output directory
    """
    # Base directory = same as main.py
    base_dir = Path(__file__).resolve().parent

    # Use custom output_dir if provided, else default to same directory as main.py
    target_base = output_dir or base_dir

    # Create subdirectory for this run
    dir_name = f"DataTables_{country_code}_{start_year}_{end_year}"
    target_dir = target_base / dir_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths with descriptive names
    # Note: family_ids.csv and main_table.csv are saved in the main function now.
    extracted_data_path = target_dir / "applicants_inventors_extracted_data.csv"
    counts_results_path = target_dir / "applicants_inventors_analysis_counts.csv"
    ratios_results_path = target_dir / "applicants_inventors_analysis_ratios.csv"
    merge_results_path = target_dir / "applicants_inventors_analysis_merge.csv"

    # Write DataFrames to CSV
    df_applicants_inventors_data.to_csv(extracted_data_path, index=False)
    logger.info(
        f"Saved: {extracted_data_path.name} ({len(df_applicants_inventors_data)} rows)"
    )

    df_counts_results.to_csv(counts_results_path, index=False)
    logger.info(f"Saved: {counts_results_path.name} ({len(df_counts_results)} rows)")

    df_ratios_results.to_csv(ratios_results_path, index=False)
    logger.info(f"Saved: {ratios_results_path.name} ({len(df_ratios_results)} rows)")

    df_merge_results.to_csv(merge_results_path, index=False)
    logger.info(f"Saved: {merge_results_path.name} ({len(df_merge_results)} rows)")

    # Log summary
    logger.info(f"\nAll results saved to: {target_dir}")
    logger.info("\nFile summary:")
    logger.info(f"  ├── family_ids.csv")
    logger.info(f"  ├── main_table.csv")
    logger.info(f"  ├── applicants_inventors_extracted_data.csv")
    logger.info(f"  ├── applicants_inventors_analysis_counts.csv")
    logger.info(f"  ├── applicants_inventors_analysis_ratios.csv")
    logger.info(f"  └── applicants_inventors_analysis_merge.csv")


# -------------------------------------------------------------
# Entry point for running the analysis directly
# -------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=Config.LOG_LEVEL,
        format=Config.LOG_FORMAT,
    )

    # Example run configuration (you can change these values)
    COUNTRY = "NO"
    START_YEAR = 2020
    END_YEAR = 2020
    RANGE_LIMIT = 100  # Optional, None for full dataset

    logger.info("Running patent analysis from __main__ ...")
    (
        df_family_ids,
        df_applicants_inventors_data,
        df_main_table,
        df_counts_results,
        df_ratios_results,
        df_merge_results,
    ) = run_full_analysis(
        country_code=COUNTRY,
        start_year=START_YEAR,
        end_year=END_YEAR,
        range_limit=RANGE_LIMIT,
        save_results=True,
    )

    logger.info("Analysis pipeline completed successfully!")
