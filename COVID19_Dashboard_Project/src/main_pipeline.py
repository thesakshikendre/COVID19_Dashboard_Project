"""
main_pipeline.py - Full ETL pipeline execution
"""
from pathlib import Path
from src.data_cleaning import load_csvs, clean_dates

def run_pipeline(raw_folder="data/raw", out_file="data/processed/processed.csv"):
    raw = Path(raw_folder)
    out = Path(out_file)
    df = load_csvs(raw)
    df = clean_dates(df)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Processed data saved at {out}")

if __name__ == "__main__":
    run_pipeline()