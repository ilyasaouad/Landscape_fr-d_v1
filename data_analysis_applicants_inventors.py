"""
Data analysis functions for applicants and inventors data.
Strategy: Collect ALL applicants and inventors from all applications in the family,
then intelligently deduplicate based on company/individual status.

FIXED: Corrected sequence number validation to use > 0 instead of .notna()
"""

import logging
import pandas as pd
import unicodedata
import re
from typing import Tuple

logger = logging.getLogger(__name__)


def standardize_name(name: str) -> str:
    """
    Standardize a person or entity name for consistent deduplication.

    Handles:
    - Unicode normalization (accents, special characters)
    - Case normalization (convert to uppercase)
    - Extra whitespace (strip and collapse multiple spaces)
    - Common abbreviations and synonyms
    - Punctuation standardization
    """
    if pd.isna(name) or name == "":
        return ""

    name = str(name).strip()

    # Unicode normalization - remove accents and special characters
    # NFD decomposes characters, then filter out combining marks
    name = unicodedata.normalize("NFD", name)
    name = "".join(char for char in name if unicodedata.category(char) != "Mn")

    # Convert to uppercase
    name = name.upper()

    # Standardize common entity name variations
    entity_replacements = {
        r"\bCORPORATION\b": "CORP",
        r"\bCOMPANY\b": "CO",
        r"\bINCORPORATED\b": "INC",
        r"\bLIMITED\b": "LTD",
        r"\bGESELLSCHAFT\s*MIT\s*BESCHRAENKTER\s*HAFTUNG\b": "GMBH",
        r"\bPRIVATE\s*LIMITED\b": "PVT LTD",
        r"\&": "AND",
        r"\s*,\s*": " ",  # Normalize commas to spaces
    }

    for pattern, replacement in entity_replacements.items():
        name = re.sub(pattern, replacement, name)

    # Remove extra whitespace
    name = re.sub(r"\s+", " ", name).strip()

    # Remove trailing punctuation
    name = re.sub(r"[,\.\;]+$", "", name).strip()

    return name


def is_company_or_entity(person_name: str) -> bool:
    """
    Determine if a person_name is a company/entity or an individual.
    Companies typically have keywords like: INC, CORP, LTD, AG, AS, SA, GMBH, etc.
    """
    if pd.isna(person_name) or person_name == "":
        return False

    name_upper = str(person_name).upper()

    company_keywords = [
        "INC",
        "CORP",
        "CORPORATION",
        "LTD",
        "LIMITED",
        "LLC",
        "GMBH",
        "AG",
        "SA",
        "AS",
        "APS",
        "COMPANY",
        "INDUSTRIES",
        "SYSTEMS",
        "TECHNOLOGIES",
        "SOLUTIONS",
        "GROUP",
        "HOLDINGS",
        "ENTERPRISES",
        "ASSOCIATES",
        "PARTNERS",
        "CONSORTIUM",
        "INSTITUTE",
        "UNIVERSITY",
        "FOUNDATION",
        "LABORATORY",
        "LABS",
        "DEPARTMENT",
    ]

    for keyword in company_keywords:
        if keyword in name_upper:
            return True

    return False


def calculate_applicants_inventors_counts(
    df_applicants_inventors_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate number of applicants and inventors per family per country.

    Strategy:
    1. Collect ALL applicants and inventors from ALL applications in the family
    2. Remove duplicates (same person_id)
    3. For each family-country combination:
       - Check if applicants include any companies/entities
       - If YES: Remove inventors from applicants (they're duplicates), keep applicants and inventors separate
       - If NO: Keep inventors in applicants (they're the same individuals in both roles)

    FIXED: Now correctly validates sequence numbers:
    - Sequence number > 0 means ACTIVE role
    - Sequence number 0 or NULL means NO role
    (Previously used .notna() which incorrectly counted 0 as active)

    Returns:
        Tuple of:
            - df_applicant_counts: applicant_count per family-country
            - df_inventor_counts: inventor_count per family-country
    """
    if df_applicants_inventors_data.empty:
        logger.warning("No applicants/inventors data provided for counting.")
        return (
            pd.DataFrame(
                columns=["docdb_family_id", "person_ctry_code", "applicant_count"]
            ),
            pd.DataFrame(
                columns=["docdb_family_id", "person_ctry_code", "inventor_count"]
            ),
        )

    logger.info("Calculating applicants/inventors counts per family and country...")

    # Clean up data
    df = df_applicants_inventors_data.copy()
    df["person_ctry_code"] = df["person_ctry_code"].astype(str).str.strip()
    df = df[df["person_ctry_code"].notna() & (df["person_ctry_code"] != "")].copy()

    # Standardize person names for consistent deduplication
    df["person_name_standardized"] = df["person_name"].apply(standardize_name)

    # Collect ALL applicants and inventors from ALL applications in each family
    all_results = []

    for family_id in df["docdb_family_id"].unique():
        family_data = df[df["docdb_family_id"] == family_id].copy()

        # FIXED: Get ALL applicants from this family where applt_seq_nr > 0 (ACTIVE applicant role)
        # Previously: used .notna() which incorrectly included applt_seq_nr == 0
        applicants = family_data[
            (family_data["applt_seq_nr"].notna())
            & (family_data["applt_seq_nr"] > 0)
        ].copy()

        # FIXED: Get ALL inventors from this family where invt_seq_nr > 0 (ACTIVE inventor role)
        # Previously: used .notna() which incorrectly included invt_seq_nr == 0
        inventors = family_data[
            (family_data["invt_seq_nr"].notna())
            & (family_data["invt_seq_nr"] > 0)
        ].copy()

        # Remove duplicates (same person_name and country)
        # Using standardized person_name instead of person_id because person_id can differ across applications
        applicants = applicants.drop_duplicates(
            subset=["docdb_family_id", "person_ctry_code", "person_name_standardized"]
        )
        inventors = inventors.drop_duplicates(
            subset=["docdb_family_id", "person_ctry_code", "person_name_standardized"]
        )

        # Check for each country if applicants have companies
        for country in set(
            list(applicants["person_ctry_code"].unique())
            + list(inventors["person_ctry_code"].unique())
        ):
            country_applicants = applicants[applicants["person_ctry_code"] == country]
            country_inventors = inventors[inventors["person_ctry_code"] == country]

            # Check if applicants in this country include companies
            has_company_applicants = False
            if not country_applicants.empty:
                has_company_applicants = (
                    country_applicants["person_name"].apply(is_company_or_entity).any()
                )

            # Get applicant and inventor person names (standardized)
            applicant_names = set(
                country_applicants["person_name_standardized"].unique()
            )
            inventor_names = set(country_inventors["person_name_standardized"].unique())

            if has_company_applicants:
                # Case A: Applicants have companies
                # Remove inventors that are also in applicants (they're duplicates)
                inventor_names_to_count = inventor_names - applicant_names
            else:
                # Case B: Applicants are ONLY individuals
                # Keep all inventors (applicants and inventors are the same people)
                inventor_names_to_count = inventor_names

            # Count applicants and inventors by unique names
            applicant_count = len(applicant_names)
            inventor_count = len(inventor_names_to_count)

            all_results.append(
                {
                    "docdb_family_id": family_id,
                    "person_ctry_code": country,
                    "applicant_count": applicant_count,
                    "inventor_count": inventor_count,
                }
            )

    # Convert to DataFrames
    if all_results:
        df_results = pd.DataFrame(all_results)

        df_applicant_counts = df_results[
            ["docdb_family_id", "person_ctry_code", "applicant_count"]
        ].copy()
        df_inventor_counts = df_results[
            ["docdb_family_id", "person_ctry_code", "inventor_count"]
        ].copy()
    else:
        df_applicant_counts = pd.DataFrame(
            columns=["docdb_family_id", "person_ctry_code", "applicant_count"]
        )
        df_inventor_counts = pd.DataFrame(
            columns=["docdb_family_id", "person_ctry_code", "inventor_count"]
        )

    logger.info(
        f"Counts calculated for {len(df_applicant_counts)} family-country pairs."
    )
    return df_applicant_counts, df_inventor_counts


def merge_applicants_and_inventors(
    df_applicant_counts: pd.DataFrame,
    df_inventor_counts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge applicant and inventor counts into one dataset.
    """
    logger.info("Merging applicant and inventor counts...")

    df_final = df_applicant_counts.merge(
        df_inventor_counts,
        on=["docdb_family_id", "person_ctry_code"],
        how="outer",
    ).fillna(0)

    # Convert counts to integers for cleaner output
    df_final["applicant_count"] = df_final["applicant_count"].astype(int)
    df_final["inventor_count"] = df_final["inventor_count"].astype(int)

    logger.info(f"Final merged dataset created with {len(df_final)} rows.")
    return df_final


def calculate_applicants_inventors_ratios(
    df_applicant_counts: pd.DataFrame,
    df_inventor_counts: pd.DataFrame,
    df_combined_counts: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate applicant, inventor, and combined ratios per family-country.

    Ratios represent the proportion of applicants/inventors from each country
    relative to the total for that patent family.

    Strategy:
    1. For each family, calculate total applicants/inventors/combined across all countries
    2. Calculate each country's proportion of the family total
    3. Return three ratio dataframes (one per metric)

    Args:
        df_applicant_counts: DataFrame with columns [docdb_family_id, person_ctry_code, applicant_count]
        df_inventor_counts: DataFrame with columns [docdb_family_id, person_ctry_code, inventor_count]
        df_combined_counts: DataFrame with columns [docdb_family_id, person_ctry_code, combined_count]

    Returns:
        Tuple of three DataFrames:
            - df_applicant_ratios: applicant ratio per family-country
            - df_inventor_ratios: inventor ratio per family-country
            - df_combined_ratios: combined ratio per family-country

    Example:
        Family 47090380 with:
            NO: 4 applicants, 10 inventors
            US: 6 applicants, 12 inventors
            Total: 10 applicants, 22 inventors

        Results:
            NO applicant_ratio: 4/10 = 0.40
            NO inventor_ratio: 10/22 = 0.45
            US applicant_ratio: 6/10 = 0.60
            US inventor_ratio: 12/22 = 0.55
    """
    logger.info("Calculating applicants/inventors ratios per family and country...")

    def calculate_ratio_per_metric(
        df: pd.DataFrame, count_column: str, ratio_column: str
    ) -> pd.DataFrame:
        """
        Calculate ratio for a specific metric (applicant/inventor/combined).

        For each family-country pair, calculates:
            ratio = count_for_country / sum_of_all_countries_in_family
        """
        if df.empty:
            logger.warning(f"Empty DataFrame provided for {ratio_column} calculation")
            return pd.DataFrame(
                columns=["docdb_family_id", "person_ctry_code", ratio_column]
            )

        df = df.copy()

        # Step 1: Calculate family-level totals
        # Sum all counts for each family across all countries
        family_totals = (
            df.groupby("docdb_family_id")[count_column]
            .sum()
            .reset_index(name="family_total")
        )

        logger.info(
            f"Calculated family-level totals for {len(family_totals)} families"
        )

        # Step 2: Merge family totals back to original data
        df = pd.merge(df, family_totals, on="docdb_family_id", how="left")

        # Step 3: Calculate ratio per family-country
        # Avoid division by zero: if family_total is 0, ratio is 0
        df[ratio_column] = df[count_column] / df["family_total"]
        df[ratio_column] = df[ratio_column].fillna(0)

        # Step 4: Return only needed columns
        result = df[["docdb_family_id", "person_ctry_code", ratio_column]].copy()

        logger.info(f"Calculated {len(result)} {ratio_column} ratios")
        return result

    # Calculate ratios for each metric
    df_applicant_ratios = calculate_ratio_per_metric(
        df_applicant_counts, "applicant_count", "applicant_ratio"
    )

    df_inventor_ratios = calculate_ratio_per_metric(
        df_inventor_counts, "inventor_count", "inventor_ratio"
    )

    df_combined_ratios = calculate_ratio_per_metric(
        df_combined_counts, "combined_count", "combined_ratio"
    )

    logger.info(
        f"Ratio calculation complete: "
        f"{len(df_applicant_ratios)} applicant, "
        f"{len(df_inventor_ratios)} inventor, "
        f"{len(df_combined_ratios)} combined ratios"
    )

    return df_applicant_ratios, df_inventor_ratios, df_combined_ratios


def merge_all_ratios(
    df_applicant_counts: pd.DataFrame,
    df_inventor_counts: pd.DataFrame,
    df_combined_counts: pd.DataFrame,
    df_applicant_ratios: pd.DataFrame,
    df_inventor_ratios: pd.DataFrame,
    df_combined_ratios: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all counts and ratios into a single comprehensive dataset.

    Combines counts and ratios to provide complete analysis per family-country.

    Args:
        df_applicant_counts: Applicant counts per family-country
        df_inventor_counts: Inventor counts per family-country
        df_combined_counts: Combined counts per family-country
        df_applicant_ratios: Applicant ratios per family-country
        df_inventor_ratios: Inventor ratios per family-country
        df_combined_ratios: Combined ratios per family-country

    Returns:
        Merged DataFrame with all metrics (counts and ratios)

    Example:
        docdb_family_id | person_ctry_code | applicant_count | applicant_ratio | ...
        47090380        | NO               | 4               | 0.40            | ...
        47090380        | US               | 6               | 0.60            | ...
    """
    logger.info("Merging all counts and ratios into comprehensive dataset...")

    # Start with base counts
    df_final = df_applicant_counts.copy()

    # Merge inventor counts
    df_final = pd.merge(
        df_final,
        df_inventor_counts,
        on=["docdb_family_id", "person_ctry_code"],
        how="outer",
    )

    # Merge combined counts
    df_final = pd.merge(
        df_final,
        df_combined_counts,
        on=["docdb_family_id", "person_ctry_code"],
        how="outer",
    )

    # Merge applicant ratios
    df_final = pd.merge(
        df_final,
        df_applicant_ratios,
        on=["docdb_family_id", "person_ctry_code"],
        how="left",
    )

    # Merge inventor ratios
    df_final = pd.merge(
        df_final,
        df_inventor_ratios,
        on=["docdb_family_id", "person_ctry_code"],
        how="left",
    )

    # Merge combined ratios
    df_final = pd.merge(
        df_final,
        df_combined_ratios,
        on=["docdb_family_id", "person_ctry_code"],
        how="left",
    )

    # Fill NaN values (from outer joins)
    df_final = df_final.fillna(0)

    # Convert counts to integers
    for col in ["applicant_count", "inventor_count", "combined_count"]:
        if col in df_final.columns:
            df_final[col] = df_final[col].astype(int)

    # Round ratios to 4 decimal places for readability
    for col in ["applicant_ratio", "inventor_ratio", "combined_ratio"]:
        if col in df_final.columns:
            df_final[col] = df_final[col].round(4)

    logger.info(f"Final merged dataset created with {len(df_final)} rows")
    return df_final


def get_ratio_statistics(df_ratios: pd.DataFrame, ratio_column: str) -> pd.DataFrame:
    """
    Calculate summary statistics for ratios (useful for analysis).

    Args:
        df_ratios: DataFrame containing ratio column
        ratio_column: Name of the ratio column to analyze

    Returns:
        DataFrame with statistics: mean, median, std, min, max

    Example:
        applicant_ratio statistics:
            mean: 0.25
            median: 0.20
            std: 0.18
            min: 0.00
            max: 1.00
    """
    if df_ratios.empty or ratio_column not in df_ratios.columns:
        logger.warning(f"Cannot calculate statistics for {ratio_column}")
        return pd.DataFrame()

    logger.info(f"Calculating statistics for {ratio_column}...")

    stats = {
        "metric": ratio_column,
        "count": df_ratios[ratio_column].count(),
        "mean": df_ratios[ratio_column].mean(),
        "median": df_ratios[ratio_column].median(),
        "std": df_ratios[ratio_column].std(),
        "min": df_ratios[ratio_column].min(),
        "max": df_ratios[ratio_column].max(),
        "q25": df_ratios[ratio_column].quantile(0.25),
        "q75": df_ratios[ratio_column].quantile(0.75),
    }

    stats_df = pd.DataFrame([stats])
    logger.info(
        f"Statistics calculated: mean={stats['mean']:.4f}, median={stats['median']:.4f}"
    )

    return stats_df