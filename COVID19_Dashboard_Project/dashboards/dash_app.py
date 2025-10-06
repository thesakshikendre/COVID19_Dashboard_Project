"""
dash_app.py - Plotly Dash dashboard starter
"""
from dash import Dash, html
app = Dash(__name__)
app.layout = html.Div([html.H1("COVID-19 Dashboard (Dash)")])

if __name__ == "__main__":
    app.run_server(debug=True)