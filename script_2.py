# Create the main data cleaning module
data_cleaning_code = '''"""
COVID-19 Data Cleaning Module
============================
Handles data extraction, cleaning, and preprocessing for COVID-19 analysis.

Author: [Your Name]
Date: September 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

class COVID19DataCleaner:
    """
    A comprehensive data cleaning pipeline for COVID-19 datasets.
    Handles Johns Hopkins data and Oxford policy tracker data.
    """
    
    def __init__(self):
        self.raw_data_path = "data/raw/"
        self.processed_data_path = "data/processed/"
        
    def load_johns_hopkins_data(self):
        """
        Load Johns Hopkins COVID-19 time series data.
        Note: In production, this would fetch from the actual GitHub repository.
        """
        try:
            # Johns Hopkins URLs (archived data)
            base_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
            
            urls = {
                'confirmed': base_url + "time_series_covid19_confirmed_global.csv",
                'deaths': base_url + "time_series_covid19_deaths_global.csv", 
                'recovered': base_url + "time_series_covid19_recovered_global.csv"
            }
            
            data = {}
            for key, url in urls.items():
                try:
                    df = pd.read_csv(url)
                    data[key] = df
                    print(f"✅ Loaded {key} data: {df.shape}")
                except:
                    # Fallback: Create sample data for demonstration
                    print(f"⚠️  Could not fetch {key} data, creating sample data...")
                    data[key] = self._create_sample_data(key)
            
            return data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            # Create sample data for all types
            return {
                'confirmed': self._create_sample_data('confirmed'),
                'deaths': self._create_sample_data('deaths'), 
                'recovered': self._create_sample_data('recovered')
            }
    
    def _create_sample_data(self, data_type):
        """Create sample COVID-19 data for demonstration purposes."""
        countries = ['US', 'Brazil', 'India', 'Russia', 'France', 'Iran', 'Germany', 
                    'Turkey', 'United Kingdom', 'Italy', 'Argentina', 'Ukraine', 
                    'Poland', 'Colombia', 'Mexico', 'South Africa', 'Philippines',
                    'Netherlands', 'Malaysia', 'Iraq', 'Japan', 'Canada', 'Chile',
                    'Bangladesh', 'Belgium', 'Thailand', 'Israel', 'Czech Republic',
                    'Peru', 'Pakistan']
        
        # Generate date range from Jan 2020 to Sep 2025
        start_date = datetime(2020, 1, 22)
        end_date = datetime(2025, 9, 8)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Create base DataFrame structure
        data = {
            'Province/State': [None] * len(countries),
            'Country/Region': countries,
            'Lat': np.random.uniform(-60, 70, len(countries)),
            'Long': np.random.uniform(-180, 180, len(countries))
        }
        
        # Generate realistic COVID data patterns
        base_multipliers = {
            'confirmed': {'US': 100000, 'Brazil': 35000, 'India': 45000, 'Russia': 21000,
                         'France': 15000, 'Iran': 7500, 'Germany': 12000, 'Turkey': 8500,
                         'United Kingdom': 10000, 'Italy': 6000},
            'deaths': {'US': 1100, 'Brazil': 700, 'India': 530, 'Russia': 400,
                      'France': 175, 'Iran': 145, 'Germany': 160, 'Turkey': 100,
                      'United Kingdom': 140, 'Italy': 175},
            'recovered': {'US': 95000, 'Brazil': 32000, 'India': 44000, 'Russia': 20000,
                         'France': 14500, 'Iran': 7200, 'Germany': 11700, 'Turkey': 8200,
                         'United Kingdom': 9700, 'Italy': 5700}
        }
        
        for i, date in enumerate(dates):
            date_str = date.strftime('%m/%d/%y')
            country_values = []
            
            for country in countries:
                # Get base multiplier for this country and data type
                base = base_multipliers[data_type].get(country, np.random.randint(1000, 10000))
                
                # Create realistic growth patterns
                days_since_start = i
                if days_since_start < 50:  # Early period - slow growth
                    value = int(base * (days_since_start / 300) * np.random.uniform(0.1, 1.5))
                elif days_since_start < 200:  # Rapid growth period  
                    value = int(base * (days_since_start / 200) * np.random.uniform(0.8, 2.5))
                elif days_since_start < 400:  # Peak period
                    value = int(base * np.random.uniform(1.5, 3.0))
                elif days_since_start < 800:  # Decline period
                    value = int(base * np.random.uniform(1.0, 2.0))
                else:  # Later waves with variants
                    wave_factor = 1 + 0.5 * np.sin(days_since_start / 100) * np.random.uniform(0.5, 1.5)
                    value = int(base * wave_factor)
                
                # Ensure cumulative nature (except for recovered which can decrease)
                if i > 0 and data_type != 'recovered':
                    prev_value = country_values[-len(countries)] if len(country_values) >= len(countries) else 0
                    value = max(value, prev_value)
                
                # Add some noise
                value = max(0, int(value * np.random.uniform(0.9, 1.1)))
                country_values.append(value)
            
            data[date_str] = country_values
        
        df = pd.DataFrame(data)
        return df
    
    def clean_johns_hopkins_data(self, raw_data):
        """
        Clean and standardize Johns Hopkins COVID-19 data.
        
        Args:
            raw_data (dict): Dictionary containing confirmed, deaths, recovered DataFrames
            
        Returns:
            dict: Cleaned DataFrames
        """
        cleaned_data = {}
        
        for data_type, df in raw_data.items():
            print(f"\\n🧹 Cleaning {data_type} data...")
            
            # Standardize column names
            df = df.rename(columns={
                'Country/Region': 'Country',
                'Province/State': 'Province'
            })
            
            # Handle missing values in geographic data
            df['Province'] = df['Province'].fillna('All')
            df['Lat'] = df['Lat'].fillna(df.groupby('Country')['Lat'].transform('mean'))
            df['Long'] = df['Long'].fillna(df.groupby('Country')['Long'].transform('mean'))
            
            # Standardize country names
            country_mapping = {
                'US': 'United States',
                'Korea, South': 'South Korea',
                'Taiwan*': 'Taiwan',
                'Congo (Kinshasa)': 'Democratic Republic of Congo',
                'Congo (Brazzaville)': 'Republic of Congo',
                'Cote d\\'Ivoire': 'Ivory Coast'
            }
            
            df['Country'] = df['Country'].replace(country_mapping)
            
            # Get date columns (all columns except the first 4)
            date_cols = df.columns[4:]
            
            # Convert date columns to datetime format and handle missing values
            for col in date_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Remove rows where all date values are 0 or NaN
            df = df[df[date_cols].sum(axis=1) > 0]
            
            cleaned_data[data_type] = df
            print(f"   ✅ Shape: {df.shape}, Countries: {df['Country'].nunique()}")
        
        return cleaned_data
    
    def aggregate_by_country(self, cleaned_data):
        """
        Aggregate data by country (sum up provinces/states).
        
        Args:
            cleaned_data (dict): Dictionary of cleaned DataFrames
            
        Returns:
            dict: Country-aggregated DataFrames
        """
        aggregated_data = {}
        
        for data_type, df in cleaned_data.items():
            print(f"\\n📊 Aggregating {data_type} by country...")
            
            # Get date columns
            date_cols = df.columns[4:]
            
            # Group by country and sum the values
            country_df = df.groupby('Country')[date_cols].sum().reset_index()
            
            # Add back geographic info (average lat/long per country)
            geo_info = df.groupby('Country')[['Lat', 'Long']].mean().reset_index()
            country_df = pd.merge(country_df, geo_info, on='Country')
            
            aggregated_data[data_type] = country_df
            print(f"   ✅ Aggregated to {len(country_df)} countries")
        
        return aggregated_data
    
    def create_daily_new_cases(self, aggregated_data):
        """
        Calculate daily new cases from cumulative data.
        
        Args:
            aggregated_data (dict): Country-aggregated DataFrames
            
        Returns:
            dict: DataFrames with daily new cases
        """
        daily_data = {}
        
        for data_type, df in aggregated_data.items():
            print(f"\\n📈 Creating daily new {data_type}...")
            
            df_daily = df.copy()
            date_cols = [col for col in df.columns if col not in ['Country', 'Lat', 'Long']]
            
            # Calculate daily differences (new cases per day)
            for i in range(len(date_cols)-1, 0, -1):
                current_col = date_cols[i]
                prev_col = date_cols[i-1]
                df_daily[current_col] = df_daily[current_col] - df_daily[prev_col]
            
            # Set first day as-is (no previous day to subtract)
            # Ensure no negative values (data quality issues)
            for col in date_cols[1:]:
                df_daily[col] = df_daily[col].clip(lower=0)
            
            daily_data[data_type] = df_daily
            
        return daily_data
    
    def calculate_moving_averages(self, daily_data, window=7):
        """
        Calculate moving averages for smoother trend analysis.
        
        Args:
            daily_data (dict): Daily new cases DataFrames  
            window (int): Rolling window size (default: 7 days)
            
        Returns:
            dict: DataFrames with moving averages
        """
        ma_data = {}
        
        for data_type, df in daily_data.items():
            print(f"\\n📉 Calculating {window}-day moving average for {data_type}...")
            
            df_ma = df.copy()
            date_cols = [col for col in df.columns if col not in ['Country', 'Lat', 'Long']]
            
            # Calculate rolling mean
            for col in date_cols:
                df_ma[f'{col}_MA{window}'] = df[date_cols].rolling(window=window, axis=1).mean().iloc[:, date_cols.index(col)]
            
            ma_data[data_type] = df_ma
        
        return ma_data
    
    def save_processed_data(self, data_dict, filename_prefix):
        """
        Save processed data to CSV files.
        
        Args:
            data_dict (dict): Dictionary of DataFrames to save
            filename_prefix (str): Prefix for output filenames
        """
        for data_type, df in data_dict.items():
            filename = f"{self.processed_data_path}{filename_prefix}_{data_type}.csv"
            df.to_csv(filename, index=False)
            print(f"💾 Saved: {filename}")
    
    def run_complete_pipeline(self):
        """
        Execute the complete data cleaning pipeline.
        
        Returns:
            dict: Dictionary containing all processed datasets
        """
        print("🚀 Starting COVID-19 Data Cleaning Pipeline\\n")
        
        # Step 1: Load raw data
        raw_data = self.load_johns_hopkins_data()
        
        # Step 2: Clean the data
        cleaned_data = self.clean_johns_hopkins_data(raw_data)
        
        # Step 3: Aggregate by country
        aggregated_data = self.aggregate_by_country(cleaned_data)
        
        # Step 4: Calculate daily new cases
        daily_data = self.create_daily_new_cases(aggregated_data)
        
        # Step 5: Calculate moving averages
        ma_data = self.calculate_moving_averages(daily_data)
        
        # Step 6: Save processed data
        self.save_processed_data(aggregated_data, 'country_cumulative')
        self.save_processed_data(daily_data, 'country_daily')
        self.save_processed_data(ma_data, 'country_moving_avg')
        
        print("\\n🎉 Data cleaning pipeline completed successfully!")
        
        return {
            'cumulative': aggregated_data,
            'daily': daily_data,
            'moving_avg': ma_data
        }

# Utility functions
def load_processed_data(data_type='cumulative'):
    """
    Load processed data from CSV files.
    
    Args:
        data_type (str): Type of data to load ('cumulative', 'daily', 'moving_avg')
        
    Returns:
        dict: Dictionary of DataFrames
    """
    data_path = "data/processed/"
    data = {}
    
    for case_type in ['confirmed', 'deaths', 'recovered']:
        filename = f"{data_path}country_{data_type}_{case_type}.csv"
        try:
            df = pd.read_csv(filename)
            data[case_type] = df
            print(f"📖 Loaded: {filename}")
        except FileNotFoundError:
            print(f"⚠️  File not found: {filename}")
    
    return data

def get_top_countries(df, n=10, metric='confirmed'):
    """
    Get top N countries by latest cumulative cases.
    
    Args:
        df (pd.DataFrame): Country data
        n (int): Number of top countries to return
        metric (str): Metric to rank by
        
    Returns:
        list: List of top country names
    """
    # Get latest date column
    date_cols = [col for col in df.columns if col not in ['Country', 'Lat', 'Long']]
    latest_col = date_cols[-1]
    
    # Sort by latest values and get top N
    top_countries = df.nlargest(n, latest_col)['Country'].tolist()
    
    return top_countries

if __name__ == "__main__":
    # Run the data cleaning pipeline
    cleaner = COVID19DataCleaner()
    processed_data = cleaner.run_complete_pipeline()
    
    # Example usage
    print("\\n📋 Sample of processed data:")
    cumulative_confirmed = processed_data['cumulative']['confirmed']
    print(f"Confirmed cases shape: {cumulative_confirmed.shape}")
    print(cumulative_confirmed.head())
    
    # Get top 10 countries
    top_10 = get_top_countries(cumulative_confirmed, n=10)
    print(f"\\nTop 10 countries by confirmed cases: {top_10}")
'''

# Save the data cleaning module
with open('COVID19_Dashboard_Project/src/data_cleaning.py', 'w') as f:
    f.write(data_cleaning_code)

print("✅ Data cleaning module created successfully!")