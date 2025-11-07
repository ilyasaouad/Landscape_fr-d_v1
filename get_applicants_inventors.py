"""
Module for retrieving applicant and inventor data from patent tables.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import aliased
from sqlalchemy.exc import SQLAlchemyError

from config import Config
from connect_database import get_session
from models_tables import TLS201_APPLN, TLS206_PERSON, TLS207_PERS_APPLN
from validators import validate_family_ids

logger = logging.getLogger(__name__)

# Aliased models for clarity in joins
t201 = aliased(TLS201_APPLN)
t206 = aliased(TLS206_PERSON)
t207 = aliased(TLS207_PERS_APPLN)

# Simplified list of columns to retrieve, focusing on applicant/inventor details.
APPLICANT_INVENTOR_COLUMNS = [
    "docdb_family_id",
    "nb_applicants",
    "nb_inventors",
    "person_ctry_code",
    "person_name",
    "person_id",
    "doc_std_name_id",
    "psn_sector",
    "applt_seq_nr",
    "invt_seq_nr",
]


def get_applicants_inventors_data(
    family_ids_path: Path,
    range_limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieve applicant and inventor data using a pre-saved list of family IDs.

    This function reads a CSV file containing 'docdb_family_id' and then
    fetches the corresponding applicant and inventor details.

    Args:
        family_ids_path: Path to the CSV file containing family IDs.
                         The file must have a 'docdb_family_id' column.
        range_limit: Optional limit on number of families to process.

    Returns:
        Tuple of (family_ids_df, applicants_inventors_df)

    Raises:
        FileNotFoundError: If the family_ids_path does not exist.
        ValueError: If the file is malformed or validation fails.
        SQLAlchemyError: If database query fails.
    """
    logger.info(f"Reading family IDs from file: {family_ids_path}")

    if not family_ids_path.exists():
        logger.error(f"Family IDs file not found at: {family_ids_path}")
        raise FileNotFoundError(f"The file {family_ids_path} was not found.")

    try:
        df_family_ids = pd.read_csv(family_ids_path)
    except Exception as e:
        logger.error(f"Failed to read CSV file {family_ids_path}: {e}")
        raise ValueError(f"Could not parse the CSV file. Error: {e}")

    # Ensure the required column exists
    if "docdb_family_id" not in df_family_ids.columns:
        raise ValueError("The CSV file must contain a 'docdb_family_id' column.")

    # Apply range limit if specified
    if range_limit is not None and range_limit > 0:
        df_family_ids = df_family_ids.head(range_limit)
        logger.info(f"Limited to {range_limit} family IDs from the file.")

    # Get applicant/inventor data
    family_ids_list = df_family_ids["docdb_family_id"].tolist()
    df_applicants_inventors = get_applicant_inventor(family_ids_list)

    return df_family_ids, df_applicants_inventors


def get_applicant_inventor(
    family_ids_list: List[int],
    *,
    save_to_csv: bool = False,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Retrieve detailed applicant/inventor data for given DOCDB family IDs.

    Args:
        family_ids_list: List of DOCDB family IDs
        save_to_csv: Whether to save results to CSV (default: False)
        output_dir: Directory to save CSV (default: Config.OUTPUT_DIR/data)

    Returns:
        DataFrame with applicant/inventor details and role column

    Raises:
        ValueError: If validation fails
        SQLAlchemyError: If database query fails
    """
    family_ids_list = validate_family_ids(family_ids_list)

    all_batches = []
    batches = [
        family_ids_list[i : i + Config.BATCH_SIZE]
        for i in range(0, len(family_ids_list), Config.BATCH_SIZE)
    ]

    try:
        for i, batch in enumerate(batches, 1):
            logger.info(
                f"Processing batch {i}/{len(batches)} ({len(batch)} family IDs)"
            )

            with get_session() as session:
                # Query only the specified columns
                query = (
                    session.query(
                        t201.docdb_family_id,
                        t201.nb_applicants,
                        t201.nb_inventors,
                        t206.person_ctry_code,
                        t206.person_name,
                        t206.person_id,
                        t206.doc_std_name_id,
                        t206.psn_sector,
                        t207.applt_seq_nr,
                        t207.invt_seq_nr,
                    )
                    .join(t207, t201.appln_id == t207.appln_id)
                    .join(t206, t207.person_id == t206.person_id)
                    .filter(t201.docdb_family_id.in_(batch))
                )

                results = query.all()
                if results:
                    # Pandas will automatically use the column names from the query result
                    df_batch = pd.DataFrame(results).drop_duplicates()
                    all_batches.append(df_batch)

    except SQLAlchemyError as e:
        logger.error(
            f"Database error fetching applicant/inventor data: {e}", exc_info=True
        )
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error fetching applicant/inventor data: {e}", exc_info=True
        )
        raise

    df_appl_invt = (
        pd.concat(all_batches, ignore_index=True)
        if all_batches
        else pd.DataFrame(columns=APPLICANT_INVENTOR_COLUMNS)
    )

    logger.info(f"Retrieved {len(df_appl_invt)} applicant/inventor records total.")

    # Add role column efficiently
    if not df_appl_invt.empty:
        conditions = [
            (df_appl_invt["applt_seq_nr"] > 0) & (df_appl_invt["invt_seq_nr"] > 0),
            (df_appl_invt["applt_seq_nr"] > 0) & ~(df_appl_invt["invt_seq_nr"] > 0),
            (df_appl_invt["invt_seq_nr"] > 0) & ~(df_appl_invt["applt_seq_nr"] > 0),
        ]
        choices = ["Inventor, Applicant", "Applicant", "Inventor"]

        df_appl_invt["role"] = np.select(conditions, choices, default="Unknown")

    # Optionally save
    if save_to_csv:
        target_dir = output_dir or (Config.OUTPUT_DIR / "data")
        target_dir.mkdir(parents=True, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_path = target_dir / f"applicant_inventor_details_{timestamp}.csv"
        df_appl_invt.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")

    return df_appl_invt
