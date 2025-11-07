"""
Data retrieval functions for patent family and person data.
Optimized for reuse, clarity, and consistent logging.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
from sqlalchemy.orm import aliased
from sqlalchemy.exc import SQLAlchemyError  # ← ADD THIS IMPORT

from config import Config
from connect_database import get_session  # ← CORRECT (should already be this)
from models_tables import TLS201_APPLN, TLS206_PERSON, TLS207_PERS_APPLN
from validators import validate_country_code, validate_year_range, validate_family_ids

logger = logging.getLogger(__name__)

# Aliased models for clarity in joins
t201 = aliased(TLS201_APPLN)
t206 = aliased(TLS206_PERSON)
t207 = aliased(TLS207_PERS_APPLN)

APPLICANT_INVENTOR_COLUMNS = [
    "docdb_family_id",
    "appln_id",
    "appln_filing_year",
    "appln_auth",
    "appln_nr",
    "docdb_family_size",
    "earliest_publn_date",
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


def get_family_ids(country_code: str, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Fetch unique DOCDB family IDs for applications linked to people
    from a given country and filing year range.

    Args:
        country_code: ISO 2-letter country code (e.g., 'NO', 'SE')
        start_year: Start year (inclusive)
        end_year: End year (inclusive)

    Returns:
        DataFrame with 'docdb_family_id' column

    Raises:
        ValueError: If validation fails
        SQLAlchemyError: If database query fails
    """
    country_code = validate_country_code(country_code)
    start_year, end_year = validate_year_range(start_year, end_year)

    try:
        with get_session() as session:
            logger.info(
                f"Fetching family IDs for {country_code} ({start_year}-{end_year})"
            )

            query = (
                session.query(t201.docdb_family_id)
                .join(t207, t201.appln_id == t207.appln_id)
                .join(t206, t207.person_id == t206.person_id)
                .filter(
                    t206.person_ctry_code == country_code,
                    t201.appln_filing_year.between(start_year, end_year),
                )
                .distinct()
            )

            results = query.all()

    except SQLAlchemyError as e:
        logger.error(
            f"Database error fetching family IDs for {country_code}: {e}", exc_info=True
        )
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching family IDs: {e}", exc_info=True)
        raise

    if not results:
        logger.warning(
            f"No family IDs found for {country_code} ({start_year}-{end_year})"
        )
        return pd.DataFrame(columns=["docdb_family_id"])

    df_family_ids = pd.DataFrame(results, columns=["docdb_family_id"])
    logger.info(f"Found {len(df_family_ids)} unique family IDs")
    return df_family_ids


def get_applicant_inventor(
    family_ids_list: List[int],
    *,
    save_to_csv: bool = True,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Retrieve detailed applicant/inventor data for given DOCDB family IDs.

    Args:
        family_ids_list: List of DOCDB family IDs
        save_to_csv: Whether to save results to CSV (default: True)
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

            # ✅ CHANGED: create_sqlalchemy_session() → get_session()
            # ✅ CHANGED: db → session
            with get_session() as session:
                query = (
                    session.query(
                        t201.docdb_family_id,
                        t201.appln_id,
                        t201.appln_filing_year,
                        t201.appln_auth,
                        t201.appln_nr,
                        t201.docdb_family_size,
                        t201.earliest_publn_date,
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
                    df_batch = pd.DataFrame(
                        results, columns=APPLICANT_INVENTOR_COLUMNS
                    ).drop_duplicates()
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


def get_applicants_inventors_data(
    country_code: str,
    start_year: int,
    end_year: int,
    range_limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieves patent family IDs and their applicant/inventor details
    for a given country and filing year range.

    Args:
        country_code: ISO 2-letter country code
        start_year: Start year (inclusive)
        end_year: End year (inclusive)
        range_limit: Maximum number of families to process (optional)

    Returns:
        Tuple of (family_ids_df, applicant_inventor_details_df)

    Raises:
        ValueError: If validation fails
    """
    country_code = validate_country_code(country_code)
    start_year, end_year = validate_year_range(start_year, end_year)

    df_family_ids = get_family_ids(country_code, start_year, end_year)

    if df_family_ids.empty:
        logger.warning("No family IDs found for the given criteria.")
        return df_family_ids, pd.DataFrame()

    # Apply range limit if provided
    if range_limit and range_limit > 0:
        df_family_ids = df_family_ids.head(range_limit)
        logger.info(f"Applied range limit of {range_limit} family IDs.")

    df_details = get_applicant_inventor(
        df_family_ids["docdb_family_id"].tolist(),
        save_to_csv=False,
    )

    logger.info(
        f"Retrieved details for {len(df_details)} persons across {len(df_family_ids)} families."
    )
    return df_family_ids, df_details
