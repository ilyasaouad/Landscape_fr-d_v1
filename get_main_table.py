"""
Module for retrieving main application data from TLS201_APPLN table. With priority_auth from table TLS204_APPLN_PRIOR
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sqlalchemy.orm import aliased
from sqlalchemy.exc import SQLAlchemyError

from config import Config
from connect_database import get_session
from models_tables import TLS201_APPLN, TLS204_APPLN_PRIOR

logger = logging.getLogger(__name__)

# Aliased models for clarity
t201 = aliased(TLS201_APPLN)
t204 = aliased(TLS204_APPLN_PRIOR)


def get_priority_auth(family_ids_list: List[int]) -> pd.DataFrame:
    """
    Retrieve the authority (appln_auth) of the earliest priority application for a list of family IDs.

    Args:
        family_ids_list: List of DOCDB family IDs.

    Returns:
        DataFrame with 'docdb_family_id' and 'priority_auth' columns.
    """
    if not family_ids_list:
        return pd.DataFrame(columns=["docdb_family_id", "priority_auth"])

    all_batches = []
    batches = [
        family_ids_list[i : i + Config.BATCH_SIZE]
        for i in range(0, len(family_ids_list), Config.BATCH_SIZE)
    ]

    try:
        for i, batch in enumerate(batches, 1):
            logger.info(
                f"Processing priority auth batch {i}/{len(batches)} ({len(batch)} family IDs)"
            )

            with get_session() as session:
                # Aliased tables for the join
                t201_prior = aliased(TLS201_APPLN)  # The priority application
                t201_later = aliased(TLS201_APPLN)  # The application claiming priority

                # This query finds the 'appln_auth' of the priority application
                # for each family ID in the batch.
                query = (
                    session.query(
                        t201_later.docdb_family_id,
                        t201_prior.appln_auth.label("priority_auth"),
                    )
                    .join(t204, t201_later.appln_id == t204.appln_id)
                    .join(t201_prior, t204.prior_appln_id == t201_prior.appln_id)
                    .filter(t201_later.docdb_family_id.in_(batch))
                    .distinct()
                )

                results = query.all()
                if results:
                    df_batch = pd.DataFrame(
                        results, columns=["docdb_family_id", "priority_auth"]
                    )
                    all_batches.append(df_batch)

    except SQLAlchemyError as e:
        logger.error(
            f"Database error fetching priority authority data: {e}", exc_info=True
        )
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error fetching priority authority data: {e}", exc_info=True
        )
        raise

    if not all_batches:
        logger.warning(f"No priority authority data found for the provided family IDs.")
        return pd.DataFrame(columns=["docdb_family_id", "priority_auth"])

    df_priority_auth = pd.concat(all_batches, ignore_index=True)
    logger.info(f"Retrieved {len(df_priority_auth)} priority authority records.")
    return df_priority_auth


def get_main_table(
    family_ids_path: Path,
    *,
    range_limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Retrieve all application data from TLS201_APPLN using a pre-saved list of family IDs.

    This function reads a CSV file containing 'docdb_family_id', fetches the main
    application data, and enriches it with the 'priority_auth' from TLS204_APPLN_PRIOR.

    Args:
        family_ids_path: Path to the CSV file containing family IDs.
                         The file must have a 'docdb_family_id' column.
        range_limit: Optional limit on number of families to process from the file.

    Returns:
        DataFrame with all columns from the main application table (TLS201_APPLN),
        plus an added 'priority_auth' column.

    Raises:
        FileNotFoundError: If the family_ids_path does not exist.
        ValueError: If the file is malformed.
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

    family_ids_list = df_family_ids["docdb_family_id"].tolist()

    # --- Step 1: Get the main TLS201 data ---
    all_batches = []
    batches = [
        family_ids_list[i : i + Config.BATCH_SIZE]
        for i in range(0, len(family_ids_list), Config.BATCH_SIZE)
    ]

    try:
        for i, batch in enumerate(batches, 1):
            logger.info(
                f"Processing main table batch {i}/{len(batches)} ({len(batch)} family IDs)"
            )

            with get_session() as session:
                # Query the entire model (equivalent to SELECT *)
                query = session.query(t201).filter(t201.docdb_family_id.in_(batch))
                results = query.all()

                if results:
                    # Convert list of ORM objects to a DataFrame
                    df_batch = pd.DataFrame([vars(row) for row in results])
                    # Drop the SQLAlchemy internal state column if it exists
                    if "_sa_instance_state" in df_batch.columns:
                        df_batch = df_batch.drop(columns=["_sa_instance_state"])
                    all_batches.append(df_batch)

    except SQLAlchemyError as e:
        logger.error(f"Database error fetching main table data: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching main table data: {e}", exc_info=True)
        raise

    if not all_batches:
        logger.warning(
            f"No main table data found for the provided {len(family_ids_list)} family IDs."
        )
        with get_session() as session:
            column_names = [col.name for col in t201.__table__.columns]
        return pd.DataFrame(columns=column_names)

    main_table = pd.concat(all_batches, ignore_index=True).drop_duplicates()
    logger.info(f"Retrieved {len(main_table)} main table records total.")

    # --- Step 2: Get priority authority data and merge ---
    if not main_table.empty:
        logger.info("Step 2: Fetching and merging priority authority data...")
        unique_family_ids = main_table["docdb_family_id"].unique().tolist()
        df_priority_auth = get_priority_auth(unique_family_ids)

        if not df_priority_auth.empty:
            # Merge the priority auth data into the main table
            main_table = pd.merge(
                main_table, df_priority_auth, on="docdb_family_id", how="left"
            )
            # Fill NaN values for families with no priority claim
            main_table["priority_auth"].fillna("Unknown", inplace=True)
            logger.info("Successfully merged priority authority data.")
        else:
            logger.warning(
                "No priority authority data found to merge. Adding 'priority_auth' column with 'Unknown' values."
            )
            main_table["priority_auth"] = "Unknown"

    # The function now only returns the DataFrame. The caller is responsible for saving.
    return main_table
