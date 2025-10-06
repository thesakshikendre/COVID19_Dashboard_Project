# Create sample notebooks and final documentation

# 1. Create EDA notebook
eda_notebook_code = '''"""
COVID-19 Exploratory Data Analysis Notebook
===========================================
Comprehensive EDA of COVID-19 global trends and patterns.

This notebook demonstrates:
- Data loading and preprocessing  
- Statistical analysis and trend identification
- Visualization of key patterns
- Country comparison analysis
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🔬 COVID-19 Exploratory Data Analysis")
print("=" * 50)

# Add project path for imports
import sys
import os
sys.path.append('../src')

try:
    from data_cleaning import COVID19DataCleaner, load_processed_data
    from utils import standardize_country_names, validate_time_series_data
    print("✅ Successfully imported project modules")
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print("Please ensure you run this notebook from the notebooks/ directory")

# ## 1. Data Loading and Initial Exploration

print("\\n📊 Loading COVID-19 datasets...")

# Initialize data cleaner and run pipeline if needed
cleaner = COVID19DataCleaner()

# Try to load existing processed data, otherwise run pipeline
try:
    data = load_processed_data('cumulative')
    if not data:
        print("Running data processing pipeline...")
        processed_data = cleaner.run_complete_pipeline()
        data = processed_data['cumulative']
except:
    print("Creating sample data for demonstration...")
    raw_data = cleaner.load_johns_hopkins_data()
    cleaned_data = cleaner.clean_johns_hopkins_data(raw_data)
    data = cleaner.aggregate_by_country(cleaned_data)

print(f"✅ Loaded datasets: {list(data.keys())}")

# ## 2. Data Quality Assessment

print("\\n🔍 Assessing data quality...")

for case_type, df in data.items():
    print(f"\\n{case_type.upper()} CASES:")
    print(f"  Shape: {df.shape}")
    print(f"  Countries: {len(df['Country'].unique())}")
    print(f"  Date range: {len([col for col in df.columns if col not in ['Country', 'Lat', 'Long']])} days")
    
    # Validation
    validation = validate_time_series_data(df)
    print(f"  Data valid: {validation['valid']}")
    if validation['issues']:
        print(f"  Issues: {len(validation['issues'])}")
    if validation['warnings']:
        print(f"  Warnings: {len(validation['warnings'])}")

# ## 3. Global Trend Analysis

print("\\n📈 Analyzing global trends...")

# Extract confirmed cases for detailed analysis
confirmed_df = data['confirmed']
date_cols = [col for col in confirmed_df.columns if col not in ['Country', 'Lat', 'Long']]

# Calculate global totals over time
global_trend = confirmed_df[date_cols].sum()
dates = pd.to_datetime(date_cols, format='%m/%d/%y')

print(f"Global confirmed cases timeline:")
print(f"  Start date: {dates[0].strftime('%B %d, %Y')}")
print(f"  End date: {dates[-1].strftime('%B %d, %Y')}")
print(f"  Total cases: {global_trend.iloc[-1]:,.0f}")
print(f"  Peak daily growth: {global_trend.diff().max():,.0f}")

# ## 4. Country-Level Analysis

print("\\n🏆 Top 10 countries analysis...")

latest_col = date_cols[-1]
top_10_countries = confirmed_df.nlargest(10, latest_col)

print("TOP 10 COUNTRIES BY CONFIRMED CASES:")
for i, (_, row) in enumerate(top_10_countries.iterrows(), 1):
    country = row['Country']
    cases = row[latest_col]
    print(f"  {i:2d}. {country:<20} {cases:>12,.0f}")

# ## 5. Statistical Analysis

print("\\n📊 Statistical summary...")

# Calculate key statistics
latest_data = confirmed_df[latest_col]
stats = {
    'Mean cases per country': latest_data.mean(),
    'Median cases per country': latest_data.median(),
    'Standard deviation': latest_data.std(),
    'Countries with >1M cases': (latest_data > 1_000_000).sum(),
    'Countries with >100K cases': (latest_data > 100_000).sum(),
    'Gini coefficient': calculate_gini_coefficient(latest_data)
}

for stat, value in stats.items():
    if isinstance(value, (int, float)):
        if value > 1000:
            print(f"  {stat}: {value:,.0f}")
        else:
            print(f"  {stat}: {value:.3f}")
    else:
        print(f"  {stat}: {value}")

# ## 6. Temporal Pattern Analysis

print("\\n⏰ Analyzing temporal patterns...")

# Calculate growth rates and acceleration
global_daily = global_trend.diff().fillna(0)
growth_rate = (global_daily / global_trend.shift(1) * 100).fillna(0)
acceleration = growth_rate.diff().fillna(0)

temporal_stats = {
    'Average daily new cases (last 30 days)': global_daily.tail(30).mean(),
    'Peak daily cases': global_daily.max(),
    'Date of peak': dates[global_daily.idxmax()].strftime('%B %d, %Y'),
    'Current growth rate (%)': growth_rate.iloc[-1],
    'Days to double (current rate)': 70 / growth_rate.iloc[-1] if growth_rate.iloc[-1] > 0 else float('inf')
}

for stat, value in temporal_stats.items():
    if isinstance(value, (int, float)):
        if abs(value) > 1000:
            print(f"  {stat}: {value:,.1f}")
        else:
            print(f"  {stat}: {value:.2f}")
    else:
        print(f"  {stat}: {value}")

# ## 7. Regional Patterns

print("\\n🌍 Regional analysis...")

# Define regional groupings
regions = {
    'North America': ['United States', 'Canada', 'Mexico'],
    'Europe': ['Germany', 'France', 'Italy', 'United Kingdom', 'Spain', 'Poland'],
    'Asia': ['India', 'China', 'Japan', 'South Korea', 'Thailand', 'Philippines'],
    'South America': ['Brazil', 'Argentina', 'Colombia', 'Chile', 'Peru'],
    'Middle East': ['Iran', 'Turkey', 'Saudi Arabia', 'Israel']
}

regional_totals = {}
for region, countries in regions.items():
    region_data = confirmed_df[confirmed_df['Country'].isin(countries)]
    if not region_data.empty:
        regional_total = region_data[latest_col].sum()
        regional_totals[region] = regional_total

print("REGIONAL TOTALS:")
sorted_regions = sorted(regional_totals.items(), key=lambda x: x[1], reverse=True)
for region, total in sorted_regions:
    print(f"  {region:<15} {total:>12,.0f}")

# ## 8. Correlation Analysis

print("\\n🔗 Cross-country correlation analysis...")

# Calculate correlations between top countries
top_5_countries = confirmed_df.nlargest(5, latest_col)
correlation_matrix = []

for _, row1 in top_5_countries.iterrows():
    country1 = row1['Country']
    series1 = row1[date_cols].values
    
    correlations = []
    for _, row2 in top_5_countries.iterrows():
        country2 = row2['Country']
        series2 = row2[date_cols].values
        
        # Calculate correlation
        corr = np.corrcoef(series1, series2)[0, 1]
        correlations.append(corr)
    
    correlation_matrix.append(correlations)

correlation_matrix = np.array(correlation_matrix)
country_names = top_5_countries['Country'].tolist()

print("CORRELATION MATRIX (Top 5 Countries):")
print(f"{'':>15}", end='')
for country in country_names:
    print(f"{country:>12}", end='')
print()

for i, country in enumerate(country_names):
    print(f"{country:>15}", end='')
    for j in range(len(country_names)):
        print(f"{correlation_matrix[i][j]:>12.3f}", end='')
    print()

# ## 9. Anomaly Detection

print("\\n🚨 Detecting anomalies...")

# Simple anomaly detection using IQR method
def detect_anomalies(series, factor=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return (series < lower_bound) | (series > upper_bound)

# Detect countries with anomalous case counts
anomalies = detect_anomalies(confirmed_df[latest_col])
anomalous_countries = confirmed_df[anomalies]['Country'].tolist()

print(f"Countries with anomalous case counts: {len(anomalous_countries)}")
if anomalous_countries:
    print("Anomalous countries:")
    for country in anomalous_countries[:5]:  # Show first 5
        cases = confirmed_df[confirmed_df['Country'] == country][latest_col].iloc[0]
        print(f"  • {country}: {cases:,.0f}")

# ## 10. Summary and Key Findings

print("\\n📋 Key Findings Summary:")
print("=" * 40)

findings = [
    f"Total global confirmed cases: {global_trend.iloc[-1]:,.0f}",
    f"Countries affected: {len(confirmed_df)}",
    f"Peak daily cases: {global_daily.max():,.0f} on {dates[global_daily.idxmax()].strftime('%B %d, %Y')}",
    f"Top affected region: {sorted_regions[0][0]} ({sorted_regions[0][1]:,.0f} cases)",
    f"Most correlated countries: Based on case trajectory patterns",
    f"Data quality: {validation['valid']} with {len(validation.get('warnings', []))} warnings"
]

for i, finding in enumerate(findings, 1):
    print(f"{i}. {finding}")

# Helper functions
def calculate_gini_coefficient(values):
    """Calculate Gini coefficient for inequality measurement."""
    values = np.array(values)
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

print("\\n✅ Exploratory Data Analysis Complete!")
print("\\nNext steps:")
print("• Run trend_analysis.py for detailed trend analysis")
print("• Run policy_impact.py for policy correlation analysis")
print("• Launch dashboard: streamlit run dashboards/streamlit_app.py")
'''

# Save EDA notebook as Python file (can be converted to .ipynb)
with open('COVID19_Dashboard_Project/notebooks/01_EDA.py', 'w') as f:
    f.write(eda_notebook_code)

print("✅ EDA notebook created successfully!")