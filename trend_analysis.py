"""
COVID-19 Trend Analysis Module
=============================
Comprehensive statistical analysis and visualization of COVID-19 trends.

Author: [Your Name]  
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class COVID19TrendAnalyzer:
    """
    Comprehensive trend analysis for COVID-19 data including:
    - Country rankings and comparisons
    - Time series trend analysis
    - Growth rate calculations
    - Correlation analysis
    """

    def __init__(self, data_path="data/processed/"):
        self.data_path = data_path
        self.data = {}
        self.colors = {
            'confirmed': '#FF6B6B',
            'deaths': '#4ECDC4', 
            'recovered': '#45B7D1'
        }

    def load_data(self):
        """Load all processed COVID-19 datasets."""
        data_types = ['cumulative', 'daily', 'moving_avg']
        case_types = ['confirmed', 'deaths', 'recovered']

        for data_type in data_types:
            self.data[data_type] = {}
            for case_type in case_types:
                filename = f"{self.data_path}country_{data_type}_{case_type}.csv"
                try:
                    df = pd.read_csv(filename)
                    self.data[data_type][case_type] = df
                    print(f"📖 Loaded: {case_type} {data_type} data")
                except FileNotFoundError:
                    print(f"⚠️  File not found: {filename}")

    def get_date_columns(self, df):
        """Extract date columns from DataFrame."""
        return [col for col in df.columns if col not in ['Country', 'Lat', 'Long'] 
                and not col.endswith(('_MA7', '_MA14'))]

    def get_top_countries(self, case_type='confirmed', n=10, data_type='cumulative'):
        """
        Get top N countries by total cases.

        Args:
            case_type (str): 'confirmed', 'deaths', or 'recovered'
            n (int): Number of countries to return
            data_type (str): 'cumulative', 'daily', or 'moving_avg'

        Returns:
            pd.DataFrame: Top countries with rankings
        """
        df = self.data[data_type][case_type].copy()
        date_cols = self.get_date_columns(df)
        latest_col = date_cols[-1]

        # Get top countries
        top_df = df.nlargest(n, latest_col)[['Country', latest_col]].copy()
        top_df['Rank'] = range(1, len(top_df) + 1)
        top_df['Total_Cases'] = top_df[latest_col]
        top_df = top_df[['Rank', 'Country', 'Total_Cases']]

        return top_df

    def calculate_growth_rates(self, country, case_type='confirmed', window=7):
        """
        Calculate growth rates for a specific country.

        Args:
            country (str): Country name
            case_type (str): Type of cases
            window (int): Window for calculating growth rate

        Returns:
            pd.Series: Growth rates over time
        """
        df = self.data['cumulative'][case_type]
        country_data = df[df['Country'] == country]

        if country_data.empty:
            return pd.Series([])

        date_cols = self.get_date_columns(df)
        values = country_data[date_cols].iloc[0]

        # Calculate growth rate
        growth_rates = []
        for i in range(window, len(values)):
            if values.iloc[i-window] > 0:
                growth_rate = ((values.iloc[i] / values.iloc[i-window]) ** (1/window) - 1) * 100
                growth_rates.append(growth_rate)
            else:
                growth_rates.append(0)

        return pd.Series(growth_rates, index=date_cols[window:])

    def create_global_summary(self):
        """
        Create global summary statistics.

        Returns:
            dict: Global summary metrics
        """
        summary = {}

        for case_type in ['confirmed', 'deaths', 'recovered']:
            df = self.data['cumulative'][case_type]
            date_cols = self.get_date_columns(df)
            latest_col = date_cols[-1]

            # Global totals
            global_total = df[latest_col].sum()
            summary[f'global_{case_type}'] = global_total

            # Daily new cases (latest)
            if case_type in self.data['daily']:
                daily_df = self.data['daily'][case_type]
                daily_new = daily_df[latest_col].sum()
                summary[f'daily_new_{case_type}'] = daily_new

        # Calculate derived metrics
        if 'global_confirmed' in summary and 'global_deaths' in summary:
            summary['global_death_rate'] = (summary['global_deaths'] / summary['global_confirmed']) * 100

        if 'global_confirmed' in summary and 'global_recovered' in summary:
            summary['global_recovery_rate'] = (summary['global_recovered'] / summary['global_confirmed']) * 100

        return summary

    def plot_country_comparison(self, countries, case_type='confirmed', chart_type='line'):
        """
        Create comparison plots for multiple countries.

        Args:
            countries (list): List of country names
            case_type (str): Type of cases to plot
            chart_type (str): 'line', 'bar', or 'area'

        Returns:
            plotly.graph_objects.Figure: Interactive plot
        """
        df = self.data['cumulative'][case_type]
        date_cols = self.get_date_columns(df)

        fig = go.Figure()

        for country in countries:
            country_data = df[df['Country'] == country]
            if not country_data.empty:
                values = country_data[date_cols].iloc[0]
                dates = pd.to_datetime(date_cols, format='%m/%d/%y')

                if chart_type == 'line':
                    fig.add_trace(go.Scatter(
                        x=dates, y=values,
                        mode='lines+markers',
                        name=country,
                        line=dict(width=3)
                    ))
                elif chart_type == 'area':
                    fig.add_trace(go.Scatter(
                        x=dates, y=values,
                        mode='lines',
                        name=country,
                        fill='tonexty' if country != countries[0] else 'tozeroy'
                    ))

        fig.update_layout(
            title=f'COVID-19 {case_type.title()} Cases - Country Comparison',
            xaxis_title='Date',  
            yaxis_title=f'{case_type.title()} Cases',
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def plot_daily_trends(self, countries, case_type='confirmed', moving_avg=True):
        """
        Plot daily new cases with optional moving average.

        Args:
            countries (list): List of countries
            case_type (str): Type of cases
            moving_avg (bool): Whether to include moving average

        Returns:
            plotly.graph_objects.Figure: Interactive plot
        """
        daily_df = self.data['daily'][case_type]
        date_cols = self.get_date_columns(daily_df)

        fig = go.Figure()

        for country in countries:
            country_data = daily_df[daily_df['Country'] == country]
            if not country_data.empty:
                values = country_data[date_cols].iloc[0]
                dates = pd.to_datetime(date_cols, format='%m/%d/%y')

                # Daily values (bars)
                fig.add_trace(go.Bar(
                    x=dates, y=values,
                    name=f'{country} (Daily)',
                    opacity=0.6
                ))

                # Moving average (line)
                if moving_avg and 'moving_avg' in self.data:
                    ma_df = self.data['moving_avg'][case_type]
                    ma_country_data = ma_df[ma_df['Country'] == country]
                    if not ma_country_data.empty:
                        ma_cols = [col for col in ma_df.columns if col.endswith('_MA7')]
                        if ma_cols:
                            ma_values = ma_country_data[ma_cols].iloc[0]
                            fig.add_trace(go.Scatter(
                                x=dates[-len(ma_values):], y=ma_values,
                                mode='lines',
                                name=f'{country} (7-day MA)',
                                line=dict(width=3)
                            ))

        fig.update_layout(
            title=f'Daily New {case_type.title()} Cases',
            xaxis_title='Date',
            yaxis_title=f'Daily New {case_type.title()} Cases',
            barmode='group',
            template='plotly_white'
        )

        return fig

    def plot_top_countries_bar(self, case_type='confirmed', n=10):
        """
        Create horizontal bar chart of top countries.

        Args:
            case_type (str): Type of cases
            n (int): Number of countries to show

        Returns:
            plotly.graph_objects.Figure: Bar chart
        """
        top_countries = self.get_top_countries(case_type, n)

        fig = go.Figure(go.Bar(
            x=top_countries['Total_Cases'],
            y=top_countries['Country'],
            orientation='h',
            marker_color=self.colors[case_type],
            text=top_countries['Total_Cases'],
            texttemplate='%{text:,.0f}',
            textposition='outside'
        ))

        fig.update_layout(
            title=f'Top {n} Countries by {case_type.title()} Cases',
            xaxis_title=f'Total {case_type.title()} Cases',
            yaxis_title='Country',
            yaxis={'categoryorder': 'total ascending'},
            template='plotly_white'
        )

        return fig

    def create_correlation_matrix(self, countries):
        """
        Create correlation matrix between countries' case trends.

        Args:
            countries (list): List of countries to analyze

        Returns:
            plotly.graph_objects.Figure: Heatmap of correlations
        """
        df = self.data['daily']['confirmed']
        date_cols = self.get_date_columns(df)

        # Create matrix of daily cases for selected countries
        country_matrix = []
        valid_countries = []

        for country in countries:
            country_data = df[df['Country'] == country]
            if not country_data.empty:
                values = country_data[date_cols].iloc[0].values
                country_matrix.append(values)
                valid_countries.append(country)

        if len(country_matrix) < 2:
            return None

        # Calculate correlation matrix
        country_matrix = np.array(country_matrix)
        corr_matrix = np.corrcoef(country_matrix)

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=valid_countries,
            y=valid_countries,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))

        fig.update_layout(
            title='Country Trend Correlation Matrix',
            template='plotly_white'
        )

        return fig

    def analyze_peaks_and_waves(self, country, case_type='confirmed'):
        """
        Identify peaks and waves in country data.

        Args:
            country (str): Country to analyze
            case_type (str): Type of cases

        Returns:
            dict: Peak analysis results
        """
        df = self.data['daily'][case_type]
        country_data = df[df['Country'] == country]

        if country_data.empty:
            return {}

        date_cols = self.get_date_columns(df)
        values = country_data[date_cols].iloc[0]
        dates = pd.to_datetime(date_cols, format='%m/%d/%y')

        # Find peaks (simple peak detection)
        peaks = []
        for i in range(1, len(values)-1):
            if values.iloc[i] > values.iloc[i-1] and values.iloc[i] > values.iloc[i+1]:
                if values.iloc[i] > values.max() * 0.1:  # Only significant peaks
                    peaks.append({
                        'date': dates[i],
                        'value': values.iloc[i],
                        'index': i
                    })

        # Sort peaks by value (descending)
        peaks = sorted(peaks, key=lambda x: x['value'], reverse=True)

        return {
            'country': country,
            'total_peaks': len(peaks),
            'highest_peak': peaks[0] if peaks else None,
            'all_peaks': peaks[:5]  # Top 5 peaks
        }

    def generate_trend_report(self, countries=None):
        """
        Generate comprehensive trend analysis report.

        Args:
            countries (list): Countries to focus on (default: top 10)

        Returns:
            dict: Comprehensive trend report
        """
        if countries is None:
            top_confirmed = self.get_top_countries('confirmed', 10)
            countries = top_confirmed['Country'].tolist()

        report = {
            'global_summary': self.create_global_summary(),
            'top_countries': {},
            'peak_analysis': {},
            'growth_analysis': {}
        }

        # Top countries by each metric
        for case_type in ['confirmed', 'deaths', 'recovered']:
            report['top_countries'][case_type] = self.get_top_countries(case_type, 10)

        # Peak analysis for focus countries
        for country in countries[:5]:  # Limit to top 5 for performance
            report['peak_analysis'][country] = self.analyze_peaks_and_waves(country)

        # Growth rate analysis
        for country in countries[:5]:
            growth_rates = self.calculate_growth_rates(country)
            if not growth_rates.empty:
                report['growth_analysis'][country] = {
                    'avg_growth_rate': growth_rates.mean(),
                    'max_growth_rate': growth_rates.max(),
                    'current_growth_rate': growth_rates.iloc[-1] if len(growth_rates) > 0 else 0
                }

        return report

def create_summary_dashboard(analyzer):
    """
    Create a comprehensive summary dashboard.

    Args:
        analyzer (COVID19TrendAnalyzer): Initialized analyzer

    Returns:
        dict: Dictionary of plotly figures
    """
    # Get top 10 countries
    top_countries = analyzer.get_top_countries('confirmed', 10)['Country'].tolist()

    figures = {}

    # 1. Top countries bar chart
    figures['top_countries_bar'] = analyzer.plot_top_countries_bar('confirmed', 10)

    # 2. Country comparison line chart (top 5)
    figures['country_comparison'] = analyzer.plot_country_comparison(top_countries[:5], 'confirmed')

    # 3. Daily trends (top 3 countries)
    figures['daily_trends'] = analyzer.plot_daily_trends(top_countries[:3], 'confirmed')

    # 4. Correlation matrix
    figures['correlation_matrix'] = analyzer.create_correlation_matrix(top_countries[:8])

    return figures

if __name__ == "__main__":
    # Initialize analyzer and load data
    analyzer = COVID19TrendAnalyzer()
    analyzer.load_data()

    # Generate comprehensive report
    print("\n📊 Generating trend analysis report...")
    report = analyzer.generate_trend_report()

    print("\n🌍 Global Summary:")
    global_summary = report['global_summary']
    for key, value in global_summary.items():
        if isinstance(value, (int, float)):
            print(f"   {key}: {value:,.0f}")

    print("\n🏆 Top 5 Countries by Confirmed Cases:")
    top_confirmed = report['top_countries']['confirmed'].head()
    for _, row in top_confirmed.iterrows():
        print(f"   {row['Rank']}. {row['Country']}: {row['Total_Cases']:,.0f}")

    # Create dashboard figures
    print("\n📈 Creating dashboard visualizations...")
    figures = create_summary_dashboard(analyzer)

    print(f"   ✅ Created {len(figures)} interactive visualizations")
    print("   📊 Available plots:", list(figures.keys()))
