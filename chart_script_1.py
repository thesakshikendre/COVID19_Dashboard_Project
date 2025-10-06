import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# Data from the provided JSON
dates = ["2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01", "2021-01-01", "2021-04-01", "2021-07-01", "2021-10-01", "2022-01-01", "2022-04-01", "2022-07-01", "2022-10-01", "2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01"]

countries_data = {
    "United States": [0, 0.2, 4.3, 8.1, 20.2, 32.1, 34.5, 44.2, 62.3, 80.1, 89.2, 95.7, 101.2, 103.8, 106.1, 107.8],
    "India": [0, 0.001, 1.1, 7.4, 10.7, 19.2, 31.4, 34.1, 38.2, 43.1, 44.2, 44.6, 44.9, 45.0, 45.1, 45.2],
    "Brazil": [0, 0.03, 2.4, 5.0, 8.7, 14.1, 19.3, 21.6, 23.1, 30.4, 33.2, 34.5, 36.8, 37.1, 37.4, 37.6],
    "Russia": [0, 0.004, 0.8, 1.4, 3.4, 4.9, 6.2, 8.0, 11.5, 17.8, 19.0, 20.4, 21.1, 21.4, 21.6, 21.8],
    "France": [0, 0.09, 0.17, 0.73, 3.5, 5.6, 6.2, 7.1, 15.8, 24.2, 31.2, 35.8, 38.2, 39.1, 39.5, 39.7]
}

# Convert dates to datetime objects
date_objects = [datetime.strptime(date, "%Y-%m-%d") for date in dates]

# Brand colors in specified order
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C']

# Create the figure
fig = go.Figure()

# Add a line for each country
for i, (country, cases) in enumerate(countries_data.items()):
    fig.add_trace(go.Scatter(
        x=date_objects,
        y=cases,
        mode='lines+markers',
        name=country,
        line=dict(color=colors[i], width=3),
        marker=dict(size=6)
    ))

# Update layout
fig.update_layout(
    title="COVID-19 Global Cumulative Cases",
    xaxis_title="Date",
    yaxis_title="Cases (millions)",
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.05,
        xanchor='center',
        x=0.5
    )
)

# Update axes
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

# Update traces
fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image("covid_global_trends.png")