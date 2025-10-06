# Create the Streamlit dashboard application
streamlit_app_code = '''"""
COVID-19 Interactive Dashboard - Streamlit
==========================================
Professional COVID-19 dashboard built with Streamlit for data visualization and analysis.

Author: [Your Name]
Date: September 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from data_cleaning import COVID19DataCleaner, load_processed_data, get_top_countries
    from trend_analysis import COVID19TrendAnalyzer
    from policy_impact import PolicyImpactAnalyzer
    from predictive_models import COVID19Predictor
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.info("Please ensure all required modules are installed and paths are correct.")

# Page configuration
st.set_page_config(
    page_title="COVID-19 Global Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

class COVID19Dashboard:
    """Main dashboard class for Streamlit application."""
    
    def __init__(self):
        self.data_cleaner = None
        self.trend_analyzer = None
        self.policy_analyzer = None
        self.predictor = None
        self.data = {}
        
    def initialize_components(self):
        """Initialize data processing components."""
        with st.spinner("Initializing dashboard components..."):
            try:
                self.data_cleaner = COVID19DataCleaner()
                self.trend_analyzer = COVID19TrendAnalyzer()
                self.policy_analyzer = PolicyImpactAnalyzer()
                self.predictor = COVID19Predictor()
                return True
            except Exception as e:
                st.error(f"Error initializing components: {e}")
                return False
    
    @st.cache_data
    def load_data(_self):
        """Load and cache processed data."""
        try:
            # Try to load existing processed data
            data = load_processed_data('cumulative')
            if not data:
                # Run data pipeline
                st.info("Running data processing pipeline...")
                cleaner = COVID19DataCleaner()
                processed_data = cleaner.run_complete_pipeline()
                data = processed_data['cumulative']
            
            return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    def render_sidebar(self):
        """Render sidebar with navigation and filters."""
        st.sidebar.markdown("## 🦠 COVID-19 Dashboard")
        st.sidebar.markdown("---")
        
        # Navigation
        page = st.sidebar.selectbox(
            "📊 Choose Analysis",
            ["Global Overview", "Country Comparison", "Policy Impact", "Predictions", "Data Explorer"]
        )
        
        st.sidebar.markdown("---")
        
        # Filters
        st.sidebar.markdown("### 🔧 Filters")
        
        # Data type selector
        data_type = st.sidebar.selectbox(
            "Data Type",
            ["confirmed", "deaths", "recovered"]
        )
        
        # Date range selector
        if self.data and 'confirmed' in self.data:
            df = self.data['confirmed']
            date_cols = [col for col in df.columns if col not in ['Country', 'Lat', 'Long']]
            if date_cols:
                start_date = pd.to_datetime(date_cols[0], format='%m/%d/%y')
                end_date = pd.to_datetime(date_cols[-1], format='%m/%d/%y')
                
                selected_dates = st.sidebar.date_input(
                    "Date Range",
                    value=(start_date, end_date),
                    min_value=start_date,
                    max_value=end_date
                )
        
        # Country selector for comparison
        if self.data and 'confirmed' in self.data:
            countries = sorted(self.data['confirmed']['Country'].unique())
            selected_countries = st.sidebar.multiselect(
                "Select Countries",
                countries,
                default=countries[:5] if len(countries) >= 5 else countries
            )
        else:
            selected_countries = []
        
        return page, data_type, selected_countries
    
    def render_global_overview(self, data_type):
        """Render global overview page."""
        st.markdown('<h1 class="main-header">🌍 Global COVID-19 Overview</h1>', unsafe_allow_html=True)
        
        if not self.data or data_type not in self.data:
            st.error("Data not available")
            return
        
        df = self.data[data_type]
        date_cols = [col for col in df.columns if col not in ['Country', 'Lat', 'Long']]
        latest_col = date_cols[-1]
        
        # Global metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_cases = df[latest_col].sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total {data_type.title()}</h3>
                <h2>{total_cases:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            countries_affected = len(df[df[latest_col] > 0])
            st.markdown(f"""
            <div class="metric-card">
                <h3>Countries Affected</h3>
                <h2>{countries_affected}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if len(date_cols) > 1:
                prev_col = date_cols[-2]
                daily_increase = df[latest_col].sum() - df[prev_col].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Daily Increase</h3>
                    <h2>{daily_increase:,.0f}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            avg_per_country = total_cases / countries_affected if countries_affected > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>Average per Country</h3>
                <h2>{avg_per_country:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Top countries chart
        st.markdown("### 🏆 Top 15 Countries")
        top_countries = df.nlargest(15, latest_col)
        
        fig = px.bar(
            x=top_countries[latest_col],
            y=top_countries['Country'],
            orientation='h',
            title=f"Top 15 Countries by {data_type.title()} Cases",
            labels={'x': f'{data_type.title()} Cases', 'y': 'Country'}
        )
        fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Global trend over time
        st.markdown("### 📈 Global Trend Over Time")
        
        # Calculate global daily totals
        global_trend = df[date_cols].sum().reset_index()
        global_trend.columns = ['Date', 'Cases']
        global_trend['Date'] = pd.to_datetime(global_trend['Date'], format='%m/%d/%y')
        
        fig = px.line(
            global_trend, 
            x='Date', 
            y='Cases',
            title=f"Global {data_type.title()} Cases Over Time"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # World map
        st.markdown("### 🗺️ Global Distribution Map")
        
        fig = px.scatter_geo(
            df,
            lat='Lat',
            lon='Long',
            size=latest_col,
            hover_name='Country',
            hover_data={latest_col: ':,.0f'},
            title=f"Global Distribution of {data_type.title()} Cases"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_country_comparison(self, data_type, selected_countries):
        """Render country comparison page."""
        st.markdown('<h1 class="main-header">🔍 Country Comparison</h1>', unsafe_allow_html=True)
        
        if not self.data or data_type not in self.data:
            st.error("Data not available")
            return
        
        if not selected_countries:
            st.warning("Please select countries from the sidebar to compare.")
            return
        
        df = self.data[data_type]
        date_cols = [col for col in df.columns if col not in ['Country', 'Lat', 'Long']]
        
        # Filter data for selected countries
        filtered_df = df[df['Country'].isin(selected_countries)]
        
        # Country comparison table
        st.markdown("### 📊 Country Statistics")
        latest_col = date_cols[-1]
        comparison_data = []
        
        for country in selected_countries:
            country_data = filtered_df[filtered_df['Country'] == country]
            if not country_data.empty:
                latest_cases = country_data[latest_col].iloc[0]
                # Calculate growth rate (last 7 days)
                if len(date_cols) >= 7:
                    week_ago_col = date_cols[-7]
                    week_ago_cases = country_data[week_ago_col].iloc[0]
                    growth_rate = ((latest_cases - week_ago_cases) / week_ago_cases * 100) if week_ago_cases > 0 else 0
                else:
                    growth_rate = 0
                
                comparison_data.append({
                    'Country': country,
                    f'Total {data_type.title()}': f"{latest_cases:,.0f}",
                    '7-Day Growth Rate': f"{growth_rate:.2f}%"
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
        
        # Time series comparison
        st.markdown("### 📈 Trend Comparison")
        
        fig = go.Figure()
        
        for country in selected_countries:
            country_data = filtered_df[filtered_df['Country'] == country]
            if not country_data.empty:
                values = country_data[date_cols].iloc[0]
                dates = pd.to_datetime(date_cols, format='%m/%d/%y')
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=values,
                    mode='lines+markers',
                    name=country,
                    line=dict(width=3)
                ))
        
        fig.update_layout(
            title=f"{data_type.title()} Cases Comparison",
            xaxis_title="Date",
            yaxis_title=f"{data_type.title()} Cases",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Growth rate comparison
        st.markdown("### 📊 Growth Rate Analysis")
        
        # Calculate daily growth rates
        growth_data = []
        for country in selected_countries:
            country_data = filtered_df[filtered_df['Country'] == country]
            if not country_data.empty:
                values = country_data[date_cols].iloc[0]
                growth_rates = []
                for i in range(1, len(values)):
                    if values.iloc[i-1] > 0:
                        growth_rate = (values.iloc[i] - values.iloc[i-1]) / values.iloc[i-1] * 100
                        growth_rates.append(growth_rate)
                    else:
                        growth_rates.append(0)
                
                # Take last 30 days of growth rates
                recent_growth = growth_rates[-30:] if len(growth_rates) >= 30 else growth_rates
                avg_growth = np.mean(recent_growth) if recent_growth else 0
                
                growth_data.append({
                    'Country': country,
                    'Average Daily Growth Rate (Last 30 days)': f"{avg_growth:.2f}%"
                })
        
        if growth_data:
            growth_df = pd.DataFrame(growth_data)
            st.dataframe(growth_df, use_container_width=True)
    
    def render_policy_impact(self):
        """Render policy impact analysis page."""
        st.markdown('<h1 class="main-header">🏛️ Policy Impact Analysis</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
            <h4>📋 Policy Impact Analysis</h4>
            <p>This section analyzes the correlation between government policies and COVID-19 case trends using the Oxford Government Response Tracker data.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Run policy analysis
        with st.spinner("Running policy impact analysis..."):
            try:
                if not hasattr(self, 'policy_results'):
                    analyzer = PolicyImpactAnalyzer()
                    self.policy_results = analyzer.run_complete_analysis()
                
                if self.policy_results:
                    # Display key insights
                    insights = self.policy_results['insights']
                    
                    st.markdown("### 🏆 Most Effective Policies")
                    for i, policy in enumerate(insights['most_effective_policies'][:3], 1):
                        correlation = policy['correlation']
                        effectiveness = "High" if correlation < -0.3 else "Moderate" if correlation < -0.1 else "Low"
                        st.markdown(f"{i}. **{policy['policy']}** - Correlation: {correlation:.3f} ({effectiveness} effectiveness)")
                    
                    st.markdown("### 💡 Key Recommendations")
                    for rec in insights['recommendations']:
                        st.markdown(f"• {rec}")
                    
                    # Policy stringency vs cases chart
                    st.markdown("### 📊 Policy Stringency vs Case Trends")
                    
                    # Create sample visualization
                    fig = go.Figure()
                    
                    # Sample data for demonstration
                    dates = pd.date_range('2020-03-01', '2021-12-31', freq='W')
                    stringency = 50 + 30 * np.sin(np.arange(len(dates)) * 0.1) + np.random.normal(0, 5, len(dates))
                    cases = 1000 * (100 - stringency) / 100 + np.random.normal(0, 100, len(dates))
                    cases = np.maximum(cases, 0)  # Ensure non-negative
                    
                    fig.add_trace(go.Scatter(
                        x=dates, y=stringency,
                        name='Policy Stringency Index',
                        yaxis='y',
                        line=dict(color='red')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=dates, y=cases,
                        name='Daily New Cases',
                        yaxis='y2',
                        line=dict(color='blue')
                    ))
                    
                    fig.update_layout(
                        title='Policy Stringency vs Daily Cases (Sample)',
                        xaxis_title='Date',
                        yaxis=dict(title='Stringency Index', side='left'),
                        yaxis2=dict(title='Daily New Cases', side='right', overlaying='y'),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error running policy analysis: {e}")
                st.info("Policy analysis requires additional data processing. This is a demonstration version.")
    
    def render_predictions(self, data_type, selected_countries):
        """Render predictions page."""
        st.markdown('<h1 class="main-header">🔮 COVID-19 Predictions</h1>', unsafe_allow_html=True)
        
        if not selected_countries:
            st.warning("Please select countries from the sidebar for predictions.")
            return
        
        # Prediction settings
        col1, col2 = st.columns(2)
        with col1:
            forecast_days = st.slider("Forecast Days", 7, 60, 21)
        with col2:
            model_type = st.selectbox("Model Type", ["ARIMA", "Prophet", "Both"])
        
        country = st.selectbox("Select Country for Prediction", selected_countries)
        
        if st.button("Generate Predictions"):
            with st.spinner(f"Generating {forecast_days}-day forecast for {country}..."):
                try:
                    predictor = COVID19Predictor()
                    results = predictor.run_complete_modeling_pipeline(
                        country=country,
                        case_type=data_type,
                        forecast_days=forecast_days
                    )
                    
                    if results and 'models' in results:
                        st.success("Predictions generated successfully!")
                        
                        # Display model performance
                        if 'comparison' in results and results['comparison']:
                            st.markdown("### 📊 Model Performance Comparison")
                            
                            metrics_df = []
                            for model_name, model_data in results['models'].items():
                                if 'metrics' in model_data and model_data['metrics']:
                                    metrics = model_data['metrics']
                                    metrics_df.append({
                                        'Model': model_name.upper(),
                                        'MAE': f"{metrics.get('MAE', 0):.2f}",
                                        'RMSE': f"{metrics.get('RMSE', 0):.2f}",
                                        'MAPE': f"{metrics.get('MAPE', 0):.2f}%",
                                        'R²': f"{metrics.get('R2', 0):.3f}"
                                    })
                            
                            if metrics_df:
                                st.dataframe(pd.DataFrame(metrics_df), use_container_width=True)
                            
                            best_model = results['comparison'].get('overall_best', 'arima')
                            st.info(f"🏆 Best performing model: {best_model.upper()}")
                        
                        # Display predictions chart
                        st.markdown("### 📈 Prediction Visualization")
                        
                        fig = go.Figure()
                        
                        # Historical data
                        if 'data' in results:
                            historical = results['data']
                            fig.add_trace(go.Scatter(
                                x=historical['date'],
                                y=historical['cases'],
                                mode='lines',
                                name='Historical Data',
                                line=dict(color='blue')
                            ))
                        
                        # Predictions
                        colors = {'arima': 'red', 'prophet': 'green'}
                        for model_name, model_data in results['models'].items():
                            if 'predictions' in model_data:
                                pred = model_data['predictions']
                                
                                # Prediction line
                                fig.add_trace(go.Scatter(
                                    x=pred['date'],
                                    y=pred['predicted'],
                                    mode='lines',
                                    name=f'{model_name.upper()} Prediction',
                                    line=dict(color=colors.get(model_name, 'orange'))
                                ))
                                
                                # Confidence interval
                                fig.add_trace(go.Scatter(
                                    x=list(pred['date']) + list(pred['date'][::-1]),
                                    y=list(pred['upper_ci']) + list(pred['lower_ci'][::-1]),
                                    fill='toself',
                                    fillcolor=f"rgba{colors.get(model_name, 'orange')}",
                                    line=dict(color='rgba(255,255,255,0)'),
                                    name=f'{model_name.upper()} Confidence Interval',
                                    opacity=0.3
                                ))
                        
                        fig.update_layout(
                            title=f'{data_type.title()} Cases Prediction for {country}',
                            xaxis_title='Date',
                            yaxis_title=f'{data_type.title()} Cases',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.error("Failed to generate predictions. Please try a different country or settings.")
                        
                except Exception as e:
                    st.error(f"Error generating predictions: {e}")
                    st.info("Prediction functionality requires additional model setup.")
    
    def render_data_explorer(self, data_type, selected_countries):
        """Render data explorer page."""
        st.markdown('<h1 class="main-header">🔍 Data Explorer</h1>', unsafe_allow_html=True)
        
        if not self.data or data_type not in self.data:
            st.error("Data not available")
            return
        
        df = self.data[data_type]
        
        # Data overview
        st.markdown("### 📊 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Countries", len(df))
        with col2:
            date_cols = [col for col in df.columns if col not in ['Country', 'Lat', 'Long']]
            st.metric("Time Points", len(date_cols))
        with col3:
            st.metric("Data Type", data_type.title())
        
        # Raw data view
        st.markdown("### 📋 Raw Data")
        
        if selected_countries:
            filtered_df = df[df['Country'].isin(selected_countries)]
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.dataframe(df.head(20), use_container_width=True)
        
        # Download options
        st.markdown("### 💾 Download Data")
        
        if st.button("Generate CSV Download"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f'covid19_{data_type}_data.csv',
                mime='text/csv'
            )
        
        # Data statistics
        st.markdown("### 📈 Data Statistics")
        
        if selected_countries:
            stats_df = filtered_df[date_cols[-10:]].describe()  # Last 10 time points
            st.dataframe(stats_df, use_container_width=True)

    def run(self):
        """Main application runner."""
        # Initialize components
        if not hasattr(self, 'data'):
            if self.initialize_components():
                self.data = self.load_data()
        
        # Render sidebar and get selections
        page, data_type, selected_countries = self.render_sidebar()
        
        # Render selected page
        if page == "Global Overview":
            self.render_global_overview(data_type)
        elif page == "Country Comparison":
            self.render_country_comparison(data_type, selected_countries)
        elif page == "Policy Impact":
            self.render_policy_impact()
        elif page == "Predictions":
            self.render_predictions(data_type, selected_countries)
        elif page == "Data Explorer":
            self.render_data_explorer(data_type, selected_countries)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>🔬 Built with Streamlit • 📊 Data from Johns Hopkins University • 🏛️ Policy data from Oxford COVID-19 Government Response Tracker</p>
            <p>💻 Created by [Your Name] • 📧 Contact: your.email@example.com</p>
        </div>
        """, unsafe_allow_html=True)

# Run the dashboard
if __name__ == "__main__":
    dashboard = COVID19Dashboard()
    dashboard.run()
'''

# Save the Streamlit dashboard
with open('COVID19_Dashboard_Project/dashboards/streamlit_app.py', 'w') as f:
    f.write(streamlit_app_code)

print("✅ Streamlit dashboard created successfully!")