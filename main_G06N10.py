"""
G06N 10 FULL DATA EXTRACTION
Gets ALL columns from TLS201, TLS206, TLS207, TLS209, TLS224
Outputs: main_G06N10_table.csv
"""

import logging
import atexit
from pathlib import Path
from typing import Optional, List

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import or_, func

from config import Config
from connect_database import get_session, cleanup_engine
from models_tables import (
    TLS201_APPLN,
    TLS206_PERSON,
    TLS207_PERS_APPLN,
    TLS209_APPLN_IPC,
    TLS224_APPLN_CPC,
)
from validators import validate_country_code, validate_year_range

logger = logging.getLogger(__name__)


def get_G06N10_full_year(
    year: int,
    countries: Optional[List[str]] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Retrieve FULL data (all table columns) for G06N 10 for one year.
    Returns combined IPC + CPC rows with all attributes.
    """

    if countries is None:
        countries = ["NO", "DK", "SE", "FI", "IS"]

    validate_year_range(year, year)
    for c in countries:
        validate_country_code(c)

    logger.info(f"Querying FULL G06N10 data for {year}...")

    with get_session() as session:

        # ---------- IPC (2 spaces: "G06N  10") ----------
        q_ipc = (
            session.query(
                TLS201_APPLN, TLS207_PERS_APPLN, TLS206_PERSON, TLS209_APPLN_IPC
            )
            .join(
                TLS207_PERS_APPLN, TLS201_APPLN.appln_id == TLS207_PERS_APPLN.appln_id
            )
            .join(TLS206_PERSON, TLS207_PERS_APPLN.person_id == TLS206_PERSON.person_id)
            .join(TLS209_APPLN_IPC, TLS201_APPLN.appln_id == TLS209_APPLN_IPC.appln_id)
            .filter(
                TLS201_APPLN.appln_filing_year == year,
                TLS206_PERSON.person_ctry_code.in_(countries),
                TLS209_APPLN_IPC.ipc_class_symbol.like("G06N  10%"),
            )
        )

        # ---------- CPC (3 spaces: "G06N   10") + spacing fix ----------
        q_cpc = (
            session.query(
                TLS201_APPLN, TLS207_PERS_APPLN, TLS206_PERSON, TLS224_APPLN_CPC
            )
            .join(
                TLS207_PERS_APPLN, TLS201_APPLN.appln_id == TLS207_PERS_APPLN.appln_id
            )
            .join(TLS206_PERSON, TLS207_PERS_APPLN.person_id == TLS206_PERSON.person_id)
            .join(TLS224_APPLN_CPC, TLS201_APPLN.appln_id == TLS224_APPLN_CPC.appln_id)
            .filter(
                TLS201_APPLN.appln_filing_year == year,
                TLS206_PERSON.person_ctry_code.in_(countries),
                or_(
                    TLS224_APPLN_CPC.cpc_class_symbol.like("G06N   10%"),  # 3 spaces
                    func.replace(TLS224_APPLN_CPC.cpc_class_symbol, " ", "").like(
                        "G06N10%"
                    ),
                ),
            )
        )

        # Execute both queries
        df_ipc = pd.read_sql(q_ipc.statement, session.bind)
        df_cpc = pd.read_sql(q_cpc.statement, session.bind)

        if debug:
            logger.info(f"IPC rows: {len(df_ipc)}, CPC rows: {len(df_cpc)}")

        # Merge IPC + CPC
        df = pd.concat([df_ipc, df_cpc], ignore_index=True).drop_duplicates()

        logger.info(f"Total FULL records retrieved for {year}: {len(df)}")
        return df


def get_G06N10_full_range(
    year_start: int = 2020,
    year_end: int = 2020,
    countries: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Retrieve full data for multiple years and combine into one master table.
    """

    if countries is None:
        countries = ["NO", "DK", "SE", "FI", "IS"]

    validate_year_range(year_start, year_end)
    for c in countries:
        validate_country_code(c)

    all_years = []
    logger.info(f"Running G06N10 full extraction for years {year_start}-{year_end}")

    for year in range(year_start, year_end + 1):
        try:
            df_year = get_G06N10_full_year(year, countries)
            if not df_year.empty:
                all_years.append(df_year)
        except Exception as e:
            logger.error(f"Error processing year {year}: {e}", exc_info=True)
            continue

    if not all_years:
        logger.warning("No data found for any year")
        return pd.DataFrame()

    df_master = pd.concat(all_years, ignore_index=True).drop_duplicates()

    logger.info(f"Final master count: {len(df_master)} records")
    return df_master


def main():
    # Init logging
    logging.basicConfig(level=logging.INFO, format=Config.LOG_FORMAT)

    logger.info("Starting FULL G06N10 extraction")
    Config.initialize()
    atexit.register(cleanup_engine)

    try:
        df = get_G06N10_full_range(
            year_start=2020,
            year_end=2020,
            countries=["NO", "DK", "SE", "FI", "IS"],
        )

        if df.empty:
            logger.warning("No G06N10 data found")
            return

        out_dir = Config.OUTPUT_DIR / "G06N10"
        out_dir.mkdir(parents=True, exist_ok=True)

        output_file = out_dir / "main_G06N10_table.csv"
        df.to_csv(output_file, index=False)

        logger.info("======================================================")
        logger.info(f"✅ Saved FULL master file: {output_file.resolve()}")
        logger.info(f"✅ Total records: {len(df)}")
        logger.info("======================================================")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
