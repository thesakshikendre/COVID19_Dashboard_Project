import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Create the data from the provided JSON
data = {
    "stringency": [15, 22, 28, 35, 41, 48, 52, 58, 63, 68, 72, 76, 81, 85, 89, 93, 18, 25, 32, 39, 44, 56, 61, 67, 74, 79, 87, 91],
    "growth_rate": [45.2, 38.7, 42.1, 28.4, 25.6, 22.8, 18.3, 19.7, 12.4, 8.9, 15.2, 6.7, 4.3, 3.1, 2.8, 1.9, 51.3, 44.8, 31.2, 29.6, 24.1, 16.8, 14.2, 10.5, 7.8, 5.4, 2.6, 1.5],
    "category": ["Low", "Low", "Low", "Moderate", "Moderate", "Moderate", "Moderate", "Moderate", "High", "High", "High", "High", "High", "High", "High", "High", "Low", "Low", "Moderate", "Moderate", "Moderate", "Moderate", "High", "High", "High", "High", "High", "High"]
}

df = pd.DataFrame(data)

# Define colors for categories
color_map = {
    "Low": "#1FB8CD",      # Strong cyan
    "Moderate": "#DB4545",  # Bright red
    "High": "#2E8B57"      # Sea green
}

# Create scatter plot
fig = px.scatter(df, 
                 x="stringency", 
                 y="growth_rate", 
                 color="category",
                 color_discrete_map=color_map,
                 title="Policy Impact")

# Add trend line
X = df["stringency"].values.reshape(-1, 1)
y = df["growth_rate"].values
reg = LinearRegression().fit(X, y)
trend_x = np.linspace(df["stringency"].min(), df["stringency"].max(), 100)
trend_y = reg.predict(trend_x.reshape(-1, 1))

fig.add_trace(go.Scatter(
    x=trend_x,
    y=trend_y,
    mode='lines',
    name='Trend',
    line=dict(color='black', width=2, dash='dash'),
    showlegend=False
))

# Update layout
fig.update_layout(
    xaxis_title="Pol Stringency",
    yaxis_title="Case Growth %",
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update traces
fig.update_traces(cliponaxis=False)

# Update axes
fig.update_xaxes(range=[0, 100])
fig.update_yaxes(range=[0, 60])

# Save the chart
fig.write_image("policy_impact_analysis.png")