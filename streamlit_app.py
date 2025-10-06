"""
COVID-19 Interactive Dashboard - Streamlit
==========================================
Professional COVID-19 dashboard built with Streamlit for data visualization and analysis.

Author: [Sakshi Kendre]
Date: September 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

# ---------------------------
# Load processed data
# ---------------------------
def load_processed_data():
    base_path = os.path.join(os.path.dirname(__file__), "data", "processed")

    data = {
        "confirmed": pd.read_csv(os.path.join(base_path, "country_cumulative_confirmed.csv")),
        "deaths": pd.read_csv(os.path.join(base_path, "country_cumulative_deaths.csv")),
        "recovered": pd.read_csv(os.path.join(base_path, "country_cumulative_recovered.csv")),
    }

    return data

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(
    page_title="COVID-19 Global Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Dashboard class
# ---------------------------
class COVID19Dashboard:
    def __init__(self):
        self.data = {}

    @st.cache_data
    def load_data(_self):
        return load_processed_data()

    def render_sidebar(self):
        st.sidebar.markdown("## 🦠 COVID-19 Dashboard")
        st.sidebar.markdown("---")

        # Navigation
        page = st.sidebar.selectbox(
            "📊 Choose Analysis",
            ["Global Overview", "Country Comparison", "Data Explorer"]
        )

        st.sidebar.markdown("---")

        # Filters
        data_type = st.sidebar.selectbox("Data Type", ["confirmed", "deaths", "recovered"])

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
        st.markdown(f"# 🌍 Global COVID-19 {data_type.title()} Overview")

        df = self.data[data_type]
        date_cols = [col for col in df.columns if col not in ['Country', 'Lat', 'Long']]
        latest_col = date_cols[-1]

        total_cases = df[latest_col].sum()
        countries_affected = len(df[df[latest_col] > 0])

        col1, col2 = st.columns(2)
        col1.metric(f"Total {data_type.title()}", f"{total_cases:,.0f}")
        col2.metric("Countries Affected", f"{countries_affected}")

        # Top 15 countries
        st.markdown("### 🏆 Top 15 Countries")
        top_countries = df.nlargest(15, latest_col)
        fig = px.bar(
            top_countries,
            x=latest_col,
            y="Country",
            orientation="h",
            title=f"Top 15 Countries by {data_type.title()} Cases"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Global trend
        st.markdown("### 📈 Global Trend Over Time")
        global_trend = df[date_cols].sum().reset_index()
        global_trend.columns = ["Date", "Cases"]
        global_trend["Date"] = pd.to_datetime(global_trend["Date"], format="%m/%d/%y")
        fig = px.line(global_trend, x="Date", y="Cases", title=f"Global {data_type.title()} Cases Over Time")
        st.plotly_chart(fig, use_container_width=True)

    def render_country_comparison(self, data_type, selected_countries):
        st.markdown(f"# 🔍 {data_type.title()} Cases - Country Comparison")

        if not selected_countries:
            st.warning("Please select countries from the sidebar to compare.")
            return

        df = self.data[data_type]
        date_cols = [col for col in df.columns if col not in ['Country', 'Lat', 'Long']]
        filtered_df = df[df['Country'].isin(selected_countries)]

        # Line chart
        fig = go.Figure()
        for country in selected_countries:
            country_data = filtered_df[filtered_df['Country'] == country]
            if not country_data.empty:
                values = country_data[date_cols].iloc[0]
                dates = pd.to_datetime(date_cols, format="%m/%d/%y")
                fig.add_trace(go.Scatter(x=dates, y=values, mode="lines", name=country))

        fig.update_layout(title=f"{data_type.title()} Cases Comparison", height=500)
        st.plotly_chart(fig, use_container_width=True)

    def render_data_explorer(self, data_type, selected_countries):
        st.markdown(f"# 🔍 Data Explorer - {data_type.title()} Cases")

        df = self.data[data_type]
        st.markdown("### 📊 Raw Data")
        if selected_countries:
            st.dataframe(df[df['Country'].isin(selected_countries)], use_container_width=True)
        else:
            st.dataframe(df.head(20), use_container_width=True)

    def run(self):
        self.data = self.load_data()
        page, data_type, selected_countries = self.render_sidebar()

        if page == "Global Overview":
            self.render_global_overview(data_type)
        elif page == "Country Comparison":
            self.render_country_comparison(data_type, selected_countries)
        elif page == "Data Explorer":
            self.render_data_explorer(data_type, selected_countries)

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center;color:#666;'>"
            "<p>🔬 Built with Streamlit • 📊 Data from Johns Hopkins University</p>"
            "<p>💻 Created by Sakshi Kendre</p>"
            "</div>",
            unsafe_allow_html=True
        )


# ---------------------------
# Run the dashboard
# ---------------------------
if __name__ == "__main__":
    dashboard = COVID19Dashboard()
    dashboard.run()

