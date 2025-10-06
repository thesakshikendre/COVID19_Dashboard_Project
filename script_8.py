# Create utility functions and final documentation

# 1. Create utils.py
utils_code = '''"""
COVID-19 Project Utilities
=========================
Helper functions and utilities for the COVID-19 data science project.

Author: [Your Name]
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Data Processing Utilities
def standardize_country_names(df, country_col='Country'):
    """
    Standardize country names to ensure consistency across datasets.
    
    Args:
        df (pd.DataFrame): DataFrame with country column
        country_col (str): Name of country column
        
    Returns:
        pd.DataFrame: DataFrame with standardized country names
    """
    country_mapping = {
        'US': 'United States',
        'Korea, South': 'South Korea', 
        'Taiwan*': 'Taiwan',
        'Congo (Kinshasa)': 'Democratic Republic of Congo',
        'Congo (Brazzaville)': 'Republic of Congo',
        'Cote d\\'Ivoire': 'Ivory Coast',
        'Burma': 'Myanmar',
        'Holy See': 'Vatican City'
    }
    
    df[country_col] = df[country_col].replace(country_mapping)
    return df

def calculate_rates(df, population_df=None):
    """
    Calculate per capita rates if population data is available.
    
    Args:
        df (pd.DataFrame): COVID case data
        population_df (pd.DataFrame): Population data by country
        
    Returns:
        pd.DataFrame: DataFrame with rate calculations
    """
    if population_df is None:
        return df
    
    # Merge with population data
    merged = pd.merge(df, population_df, on='Country', how='left')
    
    # Calculate rates per 100,000
    date_cols = [col for col in df.columns if col not in ['Country', 'Lat', 'Long']]
    for col in date_cols:
        if 'Population' in merged.columns:
            merged[f'{col}_per_100k'] = (merged[col] / merged['Population']) * 100000
    
    return merged

def handle_missing_values(df, method='forward_fill'):
    """
    Handle missing values in time series data.
    
    Args:
        df (pd.DataFrame): DataFrame with potential missing values
        method (str): Method to handle missing values
        
    Returns:
        pd.DataFrame: DataFrame with missing values handled
    """
    date_cols = [col for col in df.columns if col not in ['Country', 'Lat', 'Long']]
    
    if method == 'forward_fill':
        df[date_cols] = df[date_cols].fillna(method='ffill', axis=1)
    elif method == 'interpolate':
        df[date_cols] = df[date_cols].interpolate(axis=1)
    elif method == 'zero':
        df[date_cols] = df[date_cols].fillna(0)
    
    return df

def detect_outliers(series, method='iqr', factor=1.5):
    """
    Detect outliers in a time series.
    
    Args:
        series (pd.Series): Time series data
        method (str): Outlier detection method ('iqr', 'zscore')
        factor (float): Multiplier for outlier threshold
        
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > factor
    
    return pd.Series([False] * len(series), index=series.index)

# Analysis Utilities
def calculate_doubling_time(series, min_cases=100):
    """
    Calculate doubling time for case growth.
    
    Args:
        series (pd.Series): Cumulative case time series
        min_cases (int): Minimum cases threshold for calculation
        
    Returns:
        float: Doubling time in days
    """
    # Filter to values above minimum threshold
    filtered_series = series[series >= min_cases]
    
    if len(filtered_series) < 2:
        return np.nan
    
    # Calculate growth rate
    values = filtered_series.values
    days = np.arange(len(values))
    
    # Fit exponential growth: y = a * exp(b * x)
    # Log transform: log(y) = log(a) + b * x
    try:
        log_values = np.log(values)
        coeffs = np.polyfit(days, log_values, 1)
        growth_rate = coeffs[0]  # b coefficient
        
        # Doubling time = ln(2) / growth_rate
        doubling_time = np.log(2) / growth_rate if growth_rate > 0 else np.inf
        return doubling_time
    except:
        return np.nan

def calculate_reproduction_number(new_cases, serial_interval=7):
    """
    Estimate basic reproduction number (R0) using simple method.
    
    Args:
        new_cases (pd.Series): Daily new cases
        serial_interval (int): Serial interval in days
        
    Returns:
        pd.Series: Estimated R values over time
    """
    # Simple R estimation: R(t) = cases(t) / cases(t - serial_interval)
    r_values = new_cases / new_cases.shift(serial_interval)
    return r_values.fillna(1.0)  # Fill initial values with 1.0

def identify_waves(series, prominence=0.1, min_distance=14):
    """
    Identify epidemic waves in time series data.
    
    Args:
        series (pd.Series): Time series of cases
        prominence (float): Minimum prominence for peak detection
        min_distance (int): Minimum distance between peaks
        
    Returns:
        dict: Information about identified waves
    """
    try:
        from scipy.signal import find_peaks
        
        # Smooth the series
        smoothed = series.rolling(window=7, center=True).mean().fillna(series)
        
        # Find peaks
        peaks, properties = find_peaks(
            smoothed.values,
            prominence=smoothed.max() * prominence,
            distance=min_distance
        )
        
        waves = []
        for i, peak_idx in enumerate(peaks):
            waves.append({
                'wave_number': i + 1,
                'peak_date': series.index[peak_idx],
                'peak_value': series.iloc[peak_idx],
                'prominence': properties['prominences'][i]
            })
        
        return {
            'num_waves': len(waves),
            'waves': waves,
            'peak_indices': peaks
        }
        
    except ImportError:
        # Fallback without scipy
        return {'num_waves': 0, 'waves': [], 'peak_indices': []}

# Visualization Utilities
def create_color_palette(n_colors):
    """
    Create a color palette for visualizations.
    
    Args:
        n_colors (int): Number of colors needed
        
    Returns:
        list: List of color codes
    """
    if n_colors <= 10:
        return sns.color_palette("tab10", n_colors)
    else:
        return sns.color_palette("husl", n_colors)

def format_large_numbers(value):
    """
    Format large numbers for display.
    
    Args:
        value (float): Number to format
        
    Returns:
        str: Formatted number string
    """
    if value >= 1e9:
        return f"{value/1e9:.1f}B"
    elif value >= 1e6:
        return f"{value/1e6:.1f}M"
    elif value >= 1e3:
        return f"{value/1e3:.1f}K"
    else:
        return f"{value:.0f}"

def create_custom_theme():
    """
    Create custom matplotlib theme for consistent styling.
    
    Returns:
        dict: Matplotlib rcParams
    """
    return {
        'figure.figsize': (12, 8),
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.grid.alpha': 0.3
    }

# Data Validation Utilities
def validate_time_series_data(df):
    """
    Validate time series COVID data for common issues.
    
    Args:
        df (pd.DataFrame): Time series DataFrame
        
    Returns:
        dict: Validation results
    """
    issues = []
    warnings = []
    
    # Check for required columns
    required_cols = ['Country']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
    
    # Check for negative values in date columns
    date_cols = [col for col in df.columns if col not in ['Country', 'Lat', 'Long']]
    for col in date_cols:
        if (df[col] < 0).any():
            issues.append(f"Negative values found in column: {col}")
    
    # Check for non-monotonic cumulative data
    for _, row in df.iterrows():
        country = row['Country']
        values = row[date_cols].values
        if len(values) > 1:
            decreases = np.sum(np.diff(values) < 0)
            if decreases > len(values) * 0.1:  # More than 10% decreases
                warnings.append(f"Non-monotonic data for {country}: {decreases} decreases")
    
    # Check for extreme outliers
    for col in date_cols[-30:]:  # Check last 30 days
        if col in df.columns:
            outliers = detect_outliers(df[col], method='iqr', factor=3)
            if outliers.sum() > 0:
                warnings.append(f"Extreme outliers detected in {col}: {outliers.sum()} countries")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'total_countries': len(df),
        'date_range': len(date_cols),
        'latest_date': date_cols[-1] if date_cols else None
    }

# Export Utilities
def export_to_excel(data_dict, filename):
    """
    Export multiple DataFrames to Excel with different sheets.
    
    Args:
        data_dict (dict): Dictionary of DataFrames
        filename (str): Output Excel filename
    """
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for sheet_name, df in data_dict.items():
            # Clean sheet name (Excel limitations)
            clean_name = sheet_name.replace('/', '_').replace('\\\\', '_')[:31]
            df.to_excel(writer, sheet_name=clean_name, index=False)
    
    print(f"📊 Exported {len(data_dict)} sheets to {filename}")

def create_data_dictionary():
    """
    Create data dictionary for the COVID-19 dataset.
    
    Returns:
        pd.DataFrame: Data dictionary
    """
    data_dict = [
        {
            'Column': 'Country',
            'Type': 'String',
            'Description': 'Country or region name',
            'Example': 'United States'
        },
        {
            'Column': 'Lat',
            'Type': 'Float',
            'Description': 'Latitude coordinate',
            'Example': '37.0902'
        },
        {
            'Column': 'Long',
            'Type': 'Float', 
            'Description': 'Longitude coordinate',
            'Example': '-95.7129'
        },
        {
            'Column': 'MM/DD/YY',
            'Type': 'Integer',
            'Description': 'Cumulative confirmed cases on date',
            'Example': '1000'
        }
    ]
    
    return pd.DataFrame(data_dict)

# Configuration Management
class ProjectConfig:
    """Configuration management for the COVID-19 project."""
    
    def __init__(self):
        self.data_sources = {
            'johns_hopkins': 'https://github.com/CSSEGISandData/COVID-19',
            'oxford_tracker': 'https://github.com/OxCGRT/covid-policy-tracker',
            'our_world_in_data': 'https://github.com/owid/covid-19-data'
        }
        
        self.file_paths = {
            'raw_data': 'data/raw/',
            'processed_data': 'data/processed/',
            'reports': 'reports/',
            'dashboards': 'dashboards/'
        }
        
        self.analysis_settings = {
            'top_countries_count': 10,
            'forecast_days': 21,
            'moving_average_window': 7,
            'test_split_days': 30
        }
    
    def get_file_path(self, file_type):
        """Get file path for specific type."""
        return self.file_paths.get(file_type, '')
    
    def get_analysis_setting(self, setting_name):
        """Get analysis setting value."""
        return self.analysis_settings.get(setting_name)

# Global project configuration instance
CONFIG = ProjectConfig()

if __name__ == "__main__":
    print("COVID-19 Project Utilities")
    print("Available functions:")
    
    functions = [
        'standardize_country_names', 'calculate_rates', 'handle_missing_values',
        'detect_outliers', 'calculate_doubling_time', 'calculate_reproduction_number',
        'identify_waves', 'validate_time_series_data', 'export_to_excel'
    ]
    
    for func in functions:
        print(f"  • {func}")
'''

# Save utils.py
with open('COVID19_Dashboard_Project/src/utils.py', 'w') as f:
    f.write(utils_code)

print("✅ Utilities module created successfully!")