"""
predictive_models.py - ARIMA & Prophet-based forecasting
"""
import pandas as pd

def prepare_series(df: pd.DataFrame, date_col="date", value_col="confirmed"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values(date_col).set_index(date_col)[value_col]

if __name__ == "__main__":
    print("Predictive Models Module Loaded")