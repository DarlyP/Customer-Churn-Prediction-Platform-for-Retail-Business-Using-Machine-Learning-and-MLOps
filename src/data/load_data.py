from pathlib import Path

import pandas as pd

from src.utils.config import load_config


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names:
    - lowercase
    - replace spaces with underscores
    - remove leading/trailing spaces
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def load_raw_data(raw_path: str) -> pd.DataFrame:
    """
    Load raw CSV data.
    """
    raw_file = Path(raw_path)

    if not raw_file.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    df = pd.read_csv(raw_file)
    return df


def check_data_quality(df: pd.DataFrame) -> None:
    """
    Print basic data quality checks.
    """
    print("Data shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())


def save_interim_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save cleaned raw data to interim folder.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)
    print(f"\nInterim data saved to: {output_path}")


def main() -> None:
    config = load_config()

    raw_path = config["data"]["raw_path"]
    interim_path = config["data"]["interim_path"]

    df = load_raw_data(raw_path)
    df = standardize_column_names(df)

    check_data_quality(df)

    save_interim_data(df, interim_path)


if __name__ == "__main__":
    main()