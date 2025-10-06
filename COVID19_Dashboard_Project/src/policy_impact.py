"""
policy_impact.py - Policy correlation analysis
"""
import pandas as pd

def correlation_policy_cases(df: pd.DataFrame, policy_col="stringency", target_col="new_cases"):
    if policy_col not in df.columns or target_col not in df.columns:
        raise KeyError("Required columns missing")
    return df[policy_col].corr(df[target_col])

if __name__ == "__main__":
    print("Policy Impact Module Loaded")