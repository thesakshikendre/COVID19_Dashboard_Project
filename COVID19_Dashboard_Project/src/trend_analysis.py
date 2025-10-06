"""
trend_analysis.py - Statistical trend analysis & visualization
"""
import pandas as pd

def moving_average(series: pd.Series, window: int = 7) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()

if __name__ == "__main__":
    print("Trend Analysis Module Loaded")