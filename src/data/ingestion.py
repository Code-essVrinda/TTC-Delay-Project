"""
loads the TTC files from different years and joins them into one csv.
handles csv (2025) and xlsx (2023/2024). fixes column names, dates, spaces.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#columns in the output file
FINAL_COLUMNS: list[str] = [
    "date", "time", "day", "station", "code",
    "min_delay", "min_gap", "bound", "line", "vehicle", "year",
]

#raw column name -> standard name
_COLUMN_RENAME: dict[str, str] = {
    "date":      "date",
    "time":      "time",
    "day":       "day",
    "station":   "station",
    "code":      "code",
    "min delay": "min_delay",
    "min_delay": "min_delay",
    "min gap":   "min_gap",
    "min_gap":   "min_gap",
    "bound":     "bound",
    "line":      "line",
    "vehicle":   "vehicle",
    "_id":       "_id",  #dropped later
}

#date formats to try in order
_DATE_FORMATS: list[str] = [
    "%Y-%m-%d",   #2024/2023 xlsx
    "%d-%m-%Y",   #2025 csv
    "%m/%d/%Y",
    "%d/%m/%Y",
]

OUTPUT_PATH = Path("data/processed/combined_raw.csv")


def _detect_file_type(filepath: Path) -> str:
    #csv or xlsx from the extension
    suffix = filepath.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in (".xlsx", ".xls"):
        return "xlsx"
    raise ValueError(f"Unsupported file type '{suffix}' for file: {filepath}")


def _load_file(filepath: Path) -> pd.DataFrame:
    #read one file as-is
    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    file_type = _detect_file_type(filepath)

    if file_type == "csv":
        df = pd.read_csv(filepath, dtype=str, keep_default_na=False)
        logger.info("Loaded CSV  %-55s — shape %s", filepath.name, df.shape)
    else:
        df = pd.read_excel(filepath, dtype=str, keep_default_na=False)
        logger.info("Loaded XLSX %-55s — shape %s", filepath.name, df.shape)

    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    #fix column names, drop _id
    df.columns = [col.strip().lower() for col in df.columns]

    unknown_cols = [c for c in df.columns if c not in _COLUMN_RENAME]
    if unknown_cols:
        logger.warning("Unrecognised columns (will be kept as-is): %s", unknown_cols)

    df = df.rename(columns=_COLUMN_RENAME)

    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
        logger.debug("Dropped '_id' column.")

    return df


def _parse_date_series(series: pd.Series, source_name: str) -> pd.Series:
    #try each format, else let pandas guess
    parsed = pd.Series(pd.NaT, index=series.index)
    remaining_mask = pd.Series(True, index=series.index)

    for fmt in _DATE_FORMATS:
        if not remaining_mask.any():
            break
        subset = series[remaining_mask]
        try:
            result = pd.to_datetime(subset, format=fmt, errors="coerce")
            success_mask = result.notna()
            parsed[subset[success_mask].index] = result[success_mask]
            remaining_mask[subset[success_mask].index] = False
        except Exception:
            continue

    if remaining_mask.any():
        subset = series[remaining_mask]
        result = pd.to_datetime(subset, errors="coerce")
        parsed[subset.index] = result

    unparseable = parsed.isna().sum()
    if unparseable > 0:
        logger.warning("[%s] %d date values could not be parsed → set to NaT.", source_name, unparseable)

    return parsed


def _standardize_dates(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df["date"] = _parse_date_series(df["date"], source_name)
    return df


def _standardize_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    #strip spaces from text columns
    text_cols = df.select_dtypes(include="object").columns
    df[text_cols] = df[text_cols].apply(lambda col: col.str.strip())
    return df


def _add_year(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["date"].dt.year.astype("Int64")
    return df


def _validate(df: pd.DataFrame) -> None:
    #basic checks, just logs a summary
    logger.info("VALIDATION REPORT")

    required = [c for c in FINAL_COLUMNS if c != "year"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Combined DataFrame is missing columns: {missing_cols}")
    logger.info("Schema check: PASSED")
    logger.info("Combined shape: %s rows × %s cols", *df.shape)

    null_counts = df.isnull().sum()
    null_report = null_counts[null_counts > 0]
    if null_report.empty:
        logger.info("Null values: none detected.")
    else:
        logger.info("Null values per column:")
        for col, n in null_report.items():
            pct = 100.0 * n / len(df)
            logger.info("  %-20s %6d  (%.1f%%)", col, n, pct)

    if "year" in df.columns:
        dist = df["year"].value_counts().sort_index().to_dict()
        logger.info("Rows per year: %s", dist)

    if "min_delay" in df.columns:
        delay_numeric = pd.to_numeric(df["min_delay"], errors="coerce")
        logger.info("Min Delay range: %.0f – %.0f min  (mean %.1f)",
                    delay_numeric.min(), delay_numeric.max(), delay_numeric.mean())


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    #standard column order
    ordered = [c for c in FINAL_COLUMNS if c in df.columns]
    extras  = [c for c in df.columns if c not in FINAL_COLUMNS]
    return df[ordered + extras]


def _save(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved combined dataset → %s  (%d rows)", output_path, len(df))


def load_and_combine_data(
    file_paths: list[str | Path],
    output_path: str | Path = OUTPUT_PATH,
    save: bool = True,
) -> pd.DataFrame:
    #load every file, clean it, stack them
    if not file_paths:
        raise ValueError("file_paths must contain at least one file.")

    frames: list[pd.DataFrame] = []

    for raw_path in file_paths:
        path = Path(raw_path)
        logger.info("--- Processing: %s ---", path.name)

        df = _load_file(path)
        logger.info("  Raw shape: %s", df.shape)

        df = _standardize_columns(df)

        missing = [c for c in FINAL_COLUMNS if c not in ("year",) and c not in df.columns]
        if missing:
            raise ValueError(f"File '{path.name}' is missing required columns: {missing}")

        df = _standardize_dates(df, source_name=path.name)
        df = _standardize_text_fields(df)
        df = _add_year(df)

        logger.info("  Cleaned shape: %s", df.shape)
        frames.append(df)

    logger.info("Concatenating %d dataset(s)...", len(frames))
    combined = pd.concat(frames, ignore_index=True)
    combined = _reorder_columns(combined)

    _validate(combined)

    if save:
        _save(combined, Path(output_path))

    return combined
