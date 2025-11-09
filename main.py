"""
Main module for patent analysis - orchestrates all functionality.
"""
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from config import Config
from get_family_ids import get_family_ids
from get_main_table import get_main_table
from get_applicants_inventors import get_applicants_inventors_data
from get_classes import get_classes
from get_main_table import get_priority_auth
from data_analysis_applicants_inventors import (
    calculate_applicants_inventors_counts,
    merge_applicants_and_inventors,
    calculate_applicants_inventors_ratios,
    merge_all_ratios,
)
from validators import validate_country_code, validate_year_range

logger = logging.getLogger(__name__)


def aggregate_main_table(main_table_path: Path) -> pd.DataFrame:
    """
    Reads the main_table.csv, aggregates data by docdb_family_id,
    and returns the aggregated DataFrame (without saving).

    Args:
        main_table_path: Path to the main_table.csv file.

    Returns:
        The aggregated DataFrame.
    """
    logger.info(f"Aggregating main table from {main_table_path}")

    # Read the CSV file into a DataFrame
    df = pd.read_csv(main_table_path)

    # Create a new DataFrame with the specified columns
    df_main = df[
        [
            "docdb_family_id",
            "appln_auth",
            "appln_nr",
            "appln_kind",
            "appln_filing_year",
            "appln_nr_epodoc",
            "appln_nr_original",
            "docdb_family_size",
            "receiving_office",
            "nb_applicants",
            "nb_inventors",
            "granted",
            "priority_auth",
            "main_ipc_group",
        ]
    ].copy()

    # Create the application_number column
    df_main["application_number"] = (
        df_main["appln_auth"] + df_main["appln_nr"].astype(str) + df_main["appln_kind"]
    )

    # Remove the appln_nr and appln_kind columns
    df_main.drop(["appln_nr", "appln_kind"], axis=1, inplace=True)

    # Rearrange columns to make application_number the second column
    cols = df_main.columns.tolist()
    cols.insert(1, cols.pop(cols.index("application_number")))
    df_main = df_main[cols]

    # Aggregate all columns by docdb_family_id
    agg_funcs = {
        "application_number": "first",
        "appln_auth": "first",
        "appln_filing_year": "first",
        "appln_nr_epodoc": lambda x: ", ".join(map(str, x.unique())),
        "appln_nr_original": lambda x: ", ".join(map(str, x.unique())),
        "docdb_family_size": "first",
        "receiving_office": "first",
        "nb_applicants": "first",
        "nb_inventors": "first",
        "granted": "first",
        "priority_auth": "first",
        "main_ipc_group": "first",
    }

    df_main_agg = df_main.groupby("docdb_family_id").agg(agg_funcs)

    # Reset the index to make docdb_family_id a column again
    df_main_agg.reset_index(inplace=True)

    logger.info(f"Aggregated data: {len(df_main_agg)} families")
    logger.info(f"Columns after aggregation: {df_main_agg.columns.tolist()}")
    
    return df_main_agg


def update_missing_priority_auth(df_main_table: pd.DataFrame) -> pd.DataFrame:
    """
    Updates 'Unknown' priority_auth values with a two-stage fallback logic.

    Args:
        df_main_table: DataFrame containing priority_auth, receiving_office,
                       appln_auth, and docdb_family_id columns.

    Returns:
        The DataFrame with the 'priority_auth' column updated.
    """
    logger.info("Applying fallback for 'Unknown' priority_auth values...")

    # Identify all families that have 'Unknown' priority_auth
    unknown_families = df_main_table[df_main_table["priority_auth"] == "Unknown"][
        "docdb_family_id"
    ].unique()
    if len(unknown_families) == 0:
        logger.info("No 'Unknown' priority_auth values found. Skipping fallback.")
        return df_main_table

    logger.info(
        f"Found {len(unknown_families)} families with 'Unknown' priority_auth to update."
    )

    # Create a DataFrame containing only the rows from these families
    df_unknown_families = df_main_table[
        df_main_table["docdb_family_id"].isin(unknown_families)
    ].copy()

    # --- STAGE 1: Families where ALL receiving_office values are empty ---
    # Check if all 'receiving_office' are NaN for each family
    is_office_all_empty = df_unknown_families.groupby("docdb_family_id")[
        "receiving_office"
    ].apply(lambda x: x.isna().all())
    families_for_appln_fallback = is_office_all_empty[
        is_office_all_empty
    ].index.tolist()

    if families_for_appln_fallback:
        logger.info(
            f"Stage 1: Updating {len(families_for_appln_fallback)} families using appln_auth."
        )
        # Get the first appln_auth for these families
        appln_auth_map = (
            df_unknown_families[
                df_unknown_families["docdb_family_id"].isin(families_for_appln_fallback)
            ]
            .groupby("docdb_family_id")["appln_auth"]
            .first()
        )
        # Create mask for families to update
        mask_for_appln_fallback = df_main_table["docdb_family_id"].isin(
            families_for_appln_fallback
        )
        # Update the DataFrame
        df_main_table.loc[mask_for_appln_fallback, "priority_auth"] = df_main_table.loc[
            mask_for_appln_fallback, "docdb_family_id"
        ].map(appln_auth_map)

    # --- STAGE 2: Families where SOME receiving office values are NOT empty ---
    families_for_office_fallback = list(
        set(unknown_families) - set(families_for_appln_fallback)
    )

    if families_for_office_fallback:
        logger.info(
            f"Stage 2: Updating {len(families_for_office_fallback)} families using receiving_office."
        )
        # Get the first non-empty receiving_office for these families
        office_map = (
            df_unknown_families[
                df_unknown_families["docdb_family_id"].isin(
                    families_for_office_fallback
                )
            ]
            .dropna(subset=["receiving_office"])
            .groupby("docdb_family_id")["receiving_office"]
            .first()
        )
        # Update the DataFrame
        mask_for_office_fallback = df_main_table["docdb_family_id"].isin(
            families_for_office_fallback
        )
        df_main_table.loc[mask_for_office_fallback, "priority_auth"] = (
            df_main_table.loc[mask_for_office_fallback, "docdb_family_id"].map(
                office_map
            )
        )

    # Final check for any remaining NaNs
    df_main_table["priority_auth"] = df_main_table["priority_auth"].fillna("Unknown")

    logger.info("Priority authority update complete.")

    return df_main_table


def run_full_analysis(
    country_code: str,
    start_year: int,
    end_year: int,
    range_limit: Optional[int] = None,
    save_results: bool = True,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run the complete patent analysis pipeline.

    Args:
        country_code: Two-letter ISO country code
        start_year: Start year of analysis range
        end_year: End year of analysis range
        range_limit: Optional limit on number of families to process
        save_results: Whether to save results to CSV files
        output_dir: Optional custom output directory

    Returns:
        A single, merged DataFrame containing all relevant data.
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
        logger.warning("No data found - returning empty DataFrame")
        return pd.DataFrame()

    # Apply range limit if specified
    if range_limit is not None and range_limit > 0:
        df_family_ids = df_family_ids.head(range_limit)
        logger.info(f"Limited to {range_limit} family IDs")

    family_ids_path = target_dir / "family_ids.csv"
    df_family_ids.to_csv(family_ids_path, index=False)
    logger.info(f"Saved family IDs to {family_ids_path}")

    # Step 2: Get ALL main table data and save to CSV
    logger.info("Step 2: Getting all main table data...")
    df_main_table = get_main_table(family_ids_path, range_limit=range_limit)

    main_table_path = target_dir / "main_table.csv"
    df_main_table.to_csv(main_table_path, index=False)
    logger.info(f"Saved main table data to {main_table_path}")

    # Step 3: Add priority authority from the priority chain
    logger.info("Step 3: Adding priority authority...")
    unique_family_ids = df_main_table["docdb_family_id"].unique().tolist()
    df_priority_auth = get_priority_auth(unique_family_ids)

    # Merge it into the main table
    df_main_table = df_main_table.copy()
    df_main_table_priority = pd.merge(
        df_main_table,
        df_priority_auth,
        on="docdb_family_id",
        how="left",
        suffixes=("_base", "_priority"),
    )

    # Drop the old base column and rename the new priority column
    if "priority_auth_base" in df_main_table_priority.columns:
        df_main_table_priority = df_main_table_priority.drop(columns=['priority_auth_base'])
    if "priority_auth_priority" in df_main_table_priority.columns:
        df_main_table_priority = df_main_table_priority.rename(columns={'priority_auth_priority': 'priority_auth'})

    # Fill any remaining NaNs
    df_main_table_priority["priority_auth"] = df_main_table_priority["priority_auth"].fillna("Unknown")

    # Apply the comprehensive fallback logic
    df_main_table_priority = update_missing_priority_auth(df_main_table_priority)

    # Step 4: Get class data for representative applications
    logger.info("Step 4: Getting class data...")
    df_classes = get_classes(family_ids_path, range_limit=range_limit)

    # Step 5: Merge priority table with classes
    logger.info("Step 5: Merging main table with class data...")
    df_main_table_priority_classes = pd.merge(
        df_main_table_priority,
        df_classes,
        on="docdb_family_id",
        how="left",
        suffixes=("_main", "_classes"),
    )

    # Clean up duplicate columns from merge
    if "cpc_classes_main" in df_main_table_priority_classes.columns:
        df_main_table_priority_classes = df_main_table_priority_classes.drop(columns=["cpc_classes_main"])
    if "cpc_classes_classes" in df_main_table_priority_classes.columns:
        df_main_table_priority_classes = df_main_table_priority_classes.rename(
            columns={"cpc_classes_classes": "cpc_classes"}
        )
    
    if "main_ipc_group_main" in df_main_table_priority_classes.columns:
        df_main_table_priority_classes = df_main_table_priority_classes.drop(columns=["main_ipc_group_main"])
    if "main_ipc_group_classes" in df_main_table_priority_classes.columns:
        df_main_table_priority_classes = df_main_table_priority_classes.rename(
            columns={"main_ipc_group_classes": "main_ipc_group"}
        )

    # Save intermediate result
    if save_results:
        main_table_priority_classes_path = target_dir / "main_table_priority_classes.csv"
        df_main_table_priority_classes.to_csv(main_table_priority_classes_path, index=False)
        logger.info(f"Saved main table with priority and classes to {main_table_priority_classes_path}")

    # Step 6: Aggregate main table data
    logger.info("Step 6: Aggregating main table data...")
    # First save the merged table to CSV, then aggregate it
    temp_priority_classes_path = target_dir / "temp_main_table_priority_classes.csv"
    df_main_table_priority_classes.to_csv(temp_priority_classes_path, index=False)
    df_main_agg = aggregate_main_table(temp_priority_classes_path)
    
    # Step 6b: Add auth_family column with aggregated appln_auth values
    logger.info("Step 6b: Adding auth_family column...")
    # Group by docdb_family_id and aggregate all unique appln_auth values
    auth_family_map = (
        df_main_table_priority_classes.groupby("docdb_family_id")["appln_auth"]
        .apply(lambda x: ", ".join(sorted(set(x))))
        .to_dict()
    )
    df_main_agg["auth_family"] = df_main_agg["docdb_family_id"].map(auth_family_map)

    # Step 7: Add sector and field columns from IPC mapping
    logger.info("Step 7: Adding sector and field columns from IPC mapping...")
    ipc_mapping_path = Path(r"C:\Users\iao\Desktop\Landscape_Fråd_v1\ipc_technology_eng.xlsx")
    
    try:
        df_ipc_mapping = pd.read_excel(ipc_mapping_path)
        logger.info(f"Loaded IPC mapping from {ipc_mapping_path}")
        
        # Initialize sector and field columns
        df_main_agg["sector"] = "N/A"
        df_main_agg["field"] = "N/A"
        
        # Function to normalize IPC code
        def normalize_ipc(ipc_code):
            """Normalize IPC code to base format for comparison"""
            normalized = str(ipc_code).replace(' ', '').upper()
            normalized = normalized.split('%')[0]
            normalized = normalized.split('/')[0]
            return normalized
        
        # Function to find matching sector and field
        def find_sector_field(ipc_code):
            if pd.isna(ipc_code) or ipc_code == '' or ipc_code == 'N/A':
                return "N/A", "N/A"
            
            ipc_normalized = normalize_ipc(ipc_code)
            
            for idx, row in df_ipc_mapping.iterrows():
                try:
                    ipc_codes_str = str(row['IPC_code']).strip()
                    sector = str(row['Sector_en']) if 'Sector_en' in df_ipc_mapping.columns else "N/A"
                    field = str(row['Field_en']) if 'Field_en' in df_ipc_mapping.columns else "N/A"
                    
                    # Split multiple IPC codes by comma
                    ipc_patterns = [p.strip() for p in ipc_codes_str.split(',')]
                    
                    for pattern in ipc_patterns:
                        pattern = pattern.strip()
                        if not pattern:
                            continue
                        
                        # Normalize the pattern
                        pattern_normalized = normalize_ipc(pattern)
                        
                        # Check for exact match
                        if ipc_normalized == pattern_normalized:
                            return sector, field
                        
                        # Check if our code starts with the pattern
                        if len(pattern_normalized) < len(ipc_normalized):
                            if ipc_normalized.startswith(pattern_normalized):
                                return sector, field
                
                except Exception as e:
                    continue
            
            return "N/A", "N/A"
        
        # Apply the matching function to each IPC group
        if "main_ipc_group" in df_main_agg.columns:
            logger.info("Mapping sector and field from IPC codes...")
            
            # Debug: Show unique IPC codes
            unique_ipc = df_main_agg["main_ipc_group"].dropna().unique()
            logger.info(f"Found {len(unique_ipc)} unique IPC codes in data")
            logger.info(f"Sample IPC codes: {list(unique_ipc[:5])}")
            
            # Show what they normalize to
            for ipc in unique_ipc[:5]:
                normalized = normalize_ipc(ipc)
                logger.info(f"  IPC: '{ipc}' → Normalized: '{normalized}'")
            
            results = df_main_agg["main_ipc_group"].apply(find_sector_field)
            df_main_agg["sector"] = results.apply(lambda x: x[0])
            df_main_agg["field"] = results.apply(lambda x: x[1])
            
            # Count matches
            matched = (df_main_agg["sector"] != "N/A").sum()
            total = len(df_main_agg)
            logger.info(f"Successfully mapped {matched}/{total} IPC codes to sector/field")
        else:
            logger.warning(f"main_ipc_group column NOT found in aggregated table.")
            logger.warning(f"Available columns: {df_main_agg.columns.tolist()}")
            df_main_agg["sector"] = "N/A"
            df_main_agg["field"] = "N/A"
            
    except FileNotFoundError:
        logger.warning(f"IPC mapping file not found at {ipc_mapping_path}. Setting sector/field to N/A.")
        df_main_agg["sector"] = "N/A"
        df_main_agg["field"] = "N/A"
    except Exception as e:
        logger.warning(f"Error loading IPC mapping: {e}. Setting sector/field to N/A.")
        logger.exception(e)
        df_main_agg["sector"] = "N/A"
        df_main_agg["field"] = "N/A"

    # Step 8b: Fill empty priority_auth when appln_auth equals family_auth
    logger.info("Step 8b: Filling empty priority_auth values...")
    
    def fill_priority_auth(row):
        if pd.isna(row['priority_auth']) or row['priority_auth'] == '' or row['priority_auth'] == 'Unknown':
            appln_auth = str(row['appln_auth']).strip()
            family_auth = str(row['auth_family']).strip()
            
            # If appln_auth and family_auth are the same (single authority), use it
            if appln_auth == family_auth and appln_auth != 'nan' and appln_auth != '':
                return appln_auth
        return row['priority_auth']
    
    df_main_agg['priority_auth'] = df_main_agg.apply(fill_priority_auth, axis=1)
    df_main_agg['priority_auth'] = df_main_agg['priority_auth'].fillna('Unknown')
    
    logger.info("Priority authority fill complete.")

    # Step 8c: Reorganize main_table_agg columns
    logger.info("Step 8c: Reorganizing columns...")
    key_columns = [
        "docdb_family_id",
        "application_number",
        "appln_auth",
        "auth_family",
        "priority_auth"
    ]
    
    # Identify class columns (IPC/CPC)
    class_cols = [col for col in df_main_agg.columns if 'class' in col.lower() or 'ipc' in col.lower()]
    
    # Identify sector and field columns
    sector_field_cols = []
    if "sector" in df_main_agg.columns:
        sector_field_cols.append("sector")
    if "field" in df_main_agg.columns:
        sector_field_cols.append("field")
    
    # Get remaining columns
    remaining_columns = [col for col in df_main_agg.columns 
                         if col not in key_columns and col not in class_cols and col not in sector_field_cols]
    
    # New order: key columns + other columns + class columns + sector + field
    final_column_order = key_columns + remaining_columns + class_cols + sector_field_cols
    
    # Keep only columns that exist
    final_column_order = [col for col in final_column_order if col in df_main_agg.columns]
    df_main_agg = df_main_agg[final_column_order]
    
    logger.info(f"Final column order: {', '.join(final_column_order[:10])}...")
    
    # **SAVE THE FINAL AGGREGATED TABLE WITH SECTOR AND FIELD**
    agg_output_path = target_dir / "main_table_agg.csv"
    df_main_agg.to_csv(agg_output_path, index=False)
    logger.info(f"Saved final aggregated data with sector and field to {agg_output_path}")

    # Step 9: Get applicant/inventor data
    logger.info("Step 9: Getting applicants/inventors data...")
    _, df_applicants_inventors = get_applicants_inventors_data(family_ids_path)

    # Step 10: Final merge
    logger.info("Step 10: Merging all data on 'docdb_family_id'...")
    df_final = pd.merge(df_main_agg, df_applicants_inventors, on="docdb_family_id", how="left")

    # Step 11: Save results if requested
    if save_results:
        final_path = target_dir / "analysis_results_final.csv"
        df_final.to_csv(final_path, index=False)
        logger.info(f"Saved final merged data to {final_path}")

    logger.info(
        f"Analysis complete. Processed {len(df_family_ids)} families."
    )

    return df_final


# Entry point
if __name__ == "__main__":
    logging.basicConfig(
        level=Config.LOG_LEVEL,
        format=Config.LOG_FORMAT,
    )

    # Example run configuration
    COUNTRY = "NO"
    START_YEAR = 2020
    END_YEAR = 2020
    RANGE_LIMIT = 100

    logger.info("Running patent analysis from __main__ ...")
    df_final_result = run_full_analysis(
        country_code=COUNTRY,
        start_year=START_YEAR,
        end_year=END_YEAR,
        range_limit=RANGE_LIMIT,
        save_results=True,
    )

    logger.info("Analysis pipeline completed successfully!")