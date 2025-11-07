"""
Module for retrieving and aggregating IPC and CPC classification data.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy.orm import aliased
from sqlalchemy.exc import SQLAlchemyError

from config import Config
from connect_database import get_session
from models_tables import TLS201_APPLN, TLS209_APPLN_IPC, TLS224_APPLN_CPC

logger = logging.getLogger(__name__)

# Aliased models for clarity
t201 = aliased(TLS201_APPLN)
t209 = aliased(TLS209_APPLN_IPC)
t224 = aliased(TLS224_APPLN_CPC)


def _get_ipc_data(family_ids_list: list) -> pd.DataFrame:
    """
    Helper function to fetch MAIN IPC data using the 'F' position flag.
    It returns only the main group (e.g., 'H04H 1' from 'H04H 1/00').
    """
    all_batches = []
    batches = [
        family_ids_list[i : i + Config.BATCH_SIZE]
        for i in range(0, len(family_ids_list), Config.BATCH_SIZE)
    ]
    for i, batch in enumerate(batches, 1):
        logger.info(
            f"Processing IPC batch {i}/{len(batches)} ({len(batch)} family IDs)"
        )
        with get_session() as session:
            query = (
                session.query(t201.docdb_family_id, t209.ipc_class_symbol)
                .join(t209, t201.appln_id == t209.appln_id)
                .filter(
                    t201.docdb_family_id.in_(batch),
                    t209.ipc_position == "F",  # Filter for 'First' position
                )
                .distinct()
            )
            results = query.all()
            if results:
                df_batch = pd.DataFrame(
                    results, columns=["docdb_family_id", "ipc_class_symbol"]
                )
                all_batches.append(df_batch)

    if not all_batches:
        return pd.DataFrame(columns=["docdb_family_id", "main_ipc_group"])

    df_ipc = pd.concat(all_batches, ignore_index=True)

    # Extract the main group (e.g., "H04H 1" from "H04H 1/00")
    df_ipc["main_ipc_group"] = df_ipc["ipc_class_symbol"].apply(
        lambda x: x.split("/")[0].strip() if isinstance(x, str) and "/" in x else x
    )
    return df_ipc[["docdb_family_id", "main_ipc_group"]]


def _get_cpc_data(family_ids_list: list) -> pd.DataFrame:
    """
    Helper function to fetch all CPC data and aggregate it.
    """
    all_batches = []
    batches = [
        family_ids_list[i : i + Config.BATCH_SIZE]
        for i in range(0, len(family_ids_list), Config.BATCH_SIZE)
    ]
    for i, batch in enumerate(batches, 1):
        logger.info(
            f"Processing CPC batch {i}/{len(batches)} ({len(batch)} family IDs)"
        )
        with get_session() as session:
            query = (
                session.query(t201.docdb_family_id, t224.cpc_class_symbol)
                .join(t224, t201.appln_id == t224.appln_id)
                .filter(t201.docdb_family_id.in_(batch))
            )
            results = query.all()
            if results:
                df_batch = pd.DataFrame(
                    results, columns=["docdb_family_id", "cpc_class_symbol"]
                )
                all_batches.append(df_batch)

    if not all_batches:
        return pd.DataFrame(columns=["docdb_family_id", "cpc_classes"])

    df_cpc = pd.concat(all_batches, ignore_index=True)

    # Aggregate all CPC symbols into a single, comma-separated string
    df_cpc_agg = (
        df_cpc.groupby("docdb_family_id")["cpc_class_symbol"]
        .apply(lambda x: ", ".join(sorted(list(set(x)))))
        .reset_index()
        .rename(columns={"cpc_class_symbol": "cpc_classes"})
    )
    return df_cpc_agg


def get_classes(
    family_ids_path: Path,  # <-- REVERTED: Takes family IDs path
    *,
    range_limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Retrieve simplified classification data for a list of family IDs.

    For IPC, it fetches the main class group using the 'F' position flag.
    For CPC, it fetches and aggregates all classes.

    Args:
        family_ids_path: Path to the CSV file containing family IDs.
        range_limit: Optional limit on number of families to process.

    Returns:
        DataFrame with 'docdb_family_id', 'main_ipc_group', and 'cpc_classes'.

    Raises:
        FileNotFoundError: If the family_ids_path does not exist.
        ValueError: If the file is malformed.
        SQLAlchemyError: If database query fails.
    """
    logger.info(f"Reading family IDs from file for class data: {family_ids_path}")

    # --- Standard file reading and validation ---
    if not family_ids_path.exists():
        logger.error(f"Family IDs file not found at: {family_ids_path}")
        raise FileNotFoundError(f"The file {family_ids_path} was not found.")

    try:
        df_family_ids = pd.read_csv(family_ids_path)
    except Exception as e:
        logger.error(f"Failed to read CSV file {family_ids_path}: {e}")
        raise ValueError(f"Could not parse the CSV file. Error: {e}")

    if "docdb_family_id" not in df_family_ids.columns:
        raise ValueError("The CSV file must contain a 'docdb_family_id' column.")

    if range_limit is not None and range_limit > 0:
        df_family_ids = df_family_ids.head(range_limit)
        logger.info(f"Limited to {range_limit} family IDs for class data.")

    family_ids_list = df_family_ids["docdb_family_id"].tolist()

    # --- Fetch IPC and CPC data ---
    try:
        logger.info("Fetching main IPC data...")
        df_ipc = _get_ipc_data(family_ids_list)

        logger.info("Fetching and aggregating all CPC data...")
        df_cpc = _get_cpc_data(family_ids_list)

    except SQLAlchemyError as e:
        logger.error(f"Database error fetching class data: {e}", exc_info=True)
        raise

    # --- Merge results ---
    logger.info("Merging IPC and CPC data...")
    df_classes = pd.merge(
        df_family_ids[["docdb_family_id"]],  # Start with original family IDs
        df_ipc,
        on="docdb_family_id",
        how="left",
    ).merge(df_cpc, on="docdb_family_id", how="left")

    # Fill NaN values with a placeholder
    df_classes["main_ipc_group"].fillna("N/A", inplace=True)
    df_classes["cpc_classes"].fillna("N/A", inplace=True)

    logger.info(f"Successfully created class data for {len(df_classes)} families.")
    return df_classes
