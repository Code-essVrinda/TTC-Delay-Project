"""
rebuilds data/processed/cleaned.csv from the raw files.
ingest + combine, then clean. run again when new files are added.
python -m src.data.build_dataset
"""

import logging
from pathlib import Path

from src.data.ingestion import load_and_combine_data
from src.data.cleaning import clean_data

RAW_DIR = Path("data/raw")
PROCESSED = Path("data/processed")
RAW_FILES = [
    "TTC Subway Delay Data since 2025.csv",
    "ttc-subway-delay-data-2024.xlsx",
    "ttc-subway-delay-data-2023.xlsx",
]


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    combined = PROCESSED / "combined_raw.csv"
    cleaned = PROCESSED / "cleaned.csv"
    load_and_combine_data([RAW_DIR / f for f in RAW_FILES], output_path=combined)
    clean_data(input_path=str(combined), output_path=str(cleaned))
    print(f"\nRebuilt -> {cleaned}")


if __name__ == "__main__":
    main()
