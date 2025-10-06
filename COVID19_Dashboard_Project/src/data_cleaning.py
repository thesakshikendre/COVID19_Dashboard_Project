"""
data_cleaning.py - Data loading & preprocessing utilities
"""
import pandas as pd
from pathlib import Path

def load_csvs(folder: Path) -> pd.DataFrame:
    csvs = list(Path(folder).glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {folder}")
    dfs = [pd.read_csv(p) for p in csvs]
    return pd.concat(dfs, ignore_index=True)

def clean_dates(df: pd.DataFrame, date_col="date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df

if __name__ == "__main__":
    print("Data Cleaning Module Loaded")