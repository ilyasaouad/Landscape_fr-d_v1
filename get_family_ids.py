"""
Module for retrieving family IDs based on country and year range.
"""

import logging
from typing import List

import pandas as pd
from sqlalchemy.orm import aliased
from sqlalchemy.exc import SQLAlchemyError

from config import Config
from connect_database import get_session
from models_tables import TLS201_APPLN, TLS206_PERSON, TLS207_PERS_APPLN
from validators import validate_country_code, validate_year_range

logger = logging.getLogger(__name__)

# Aliased models for clarity in joins
t201 = aliased(TLS201_APPLN)
t206 = aliased(TLS206_PERSON)
t207 = aliased(TLS207_PERS_APPLN)


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
