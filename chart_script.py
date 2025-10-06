import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Define the enhanced data with better positioning for left-to-right flow
components = [
    # Data Sources (leftmost column)
    {"name": "JH COVID Data", "type": "data_source", "position": {"x": 1, "y": 3}},
    {"name": "Oxford Policy", "type": "data_source", "position": {"x": 1, "y": 2}},
    {"name": "Demographics", "type": "data_source", "position": {"x": 1, "y": 1}},
    
    # Processing Pipeline (middle columns)
    {"name": "Data Cleaning", "type": "processing", "position": {"x": 3, "y": 2}},
    {"name": "Trend Analysis", "type": "processing", "position": {"x": 5, "y": 3}},
    {"name": "Policy Impact", "type": "processing", "position": {"x": 5, "y": 2}},
    {"name": "Predict Models", "type": "processing", "position": {"x": 5, "y": 1}},
    
    # Outputs (rightmost columns)
    {"name": "Dashboard", "type": "output", "position": {"x": 7, "y": 2}},
    {"name": "Reports", "type": "output", "position": {"x": 7, "y": 1}},
    {"name": "Data Exports", "type": "output", "position": {"x": 9, "y": 1.5}},
]

connections = [
    {"from": "JH COVID Data", "to": "Data Cleaning"},
    {"from": "Oxford Policy", "to": "Data Cleaning"},
    {"from": "Demographics", "to": "Data Cleaning"},
    {"from": "Data Cleaning", "to": "Trend Analysis"},
    {"from": "Data Cleaning", "to": "Policy Impact"},
    {"from": "Data Cleaning", "to": "Predict Models"},
    {"from": "Trend Analysis", "to": "Dashboard"},
    {"from": "Policy Impact", "to": "Dashboard"},
    {"from": "Predict Models", "to": "Dashboard"},
    {"from": "Policy Impact", "to": "Reports"},
    {"from": "Predict Models", "to": "Reports"},
    {"from": "Dashboard", "to": "Data Exports"},
    {"from": "Reports", "to": "Data Exports"},
]

# Define colors for different component types
colors = {
    "data_source": "#1FB8CD",  # Strong cyan
    "processing": "#DB4545",   # Bright red
    "output": "#2E8B57",       # Sea green
}

# Create position and type mappings
pos_mapping = {comp["name"]: comp["position"] for comp in components}
type_mapping = {comp["name"]: comp["type"] for comp in components}

# Create figure
fig = go.Figure()

# Add connections with improved arrows
for conn in connections:
    from_pos = pos_mapping[conn["from"]]
    to_pos = pos_mapping[conn["to"]]
    
    # Calculate arrow positioning
    dx = to_pos["x"] - from_pos["x"]
    dy = to_pos["y"] - from_pos["y"]
    
    # Adjust start and end points to account for node size
    offset = 0.3
    if dx > 0:  # Moving right
        start_x = from_pos["x"] + offset
        end_x = to_pos["x"] - offset
    else:
        start_x = from_pos["x"] - offset
        end_x = to_pos["x"] + offset
    
    if dy > 0:  # Moving up
        start_y = from_pos["y"] + offset/2
        end_y = to_pos["y"] - offset/2
    elif dy < 0:  # Moving down
        start_y = from_pos["y"] - offset/2
        end_y = to_pos["y"] + offset/2
    else:  # Same level
        start_y = from_pos["y"]
        end_y = to_pos["y"]
    
    # Add connecting line
    fig.add_trace(go.Scatter(
        x=[start_x, end_x],
        y=[start_y, end_y],
        mode='lines',
        line=dict(color='#666666', width=2),
        showlegend=False,
        hoverinfo='none'
    ))
    
    # Add arrowhead
    arrow_size = 0.15
    angle = np.arctan2(end_y - start_y, end_x - start_x)
    
    arrow_x = [
        end_x - arrow_size * np.cos(angle - np.pi/6),
        end_x,
        end_x - arrow_size * np.cos(angle + np.pi/6)
    ]
    arrow_y = [
        end_y - arrow_size * np.sin(angle - np.pi/6),
        end_y,
        end_y - arrow_size * np.sin(angle + np.pi/6)
    ]
    
    fig.add_trace(go.Scatter(
        x=arrow_x,
        y=arrow_y,
        mode='lines',
        line=dict(color='#666666', width=2),
        fill='toself',
        fillcolor='#666666',
        showlegend=False,
        hoverinfo='none'
    ))

# Add rectangular nodes by component type
for comp_type in ["data_source", "processing", "output"]:
    x_vals = []
    y_vals = []
    texts = []
    
    for comp in components:
        if comp["type"] == comp_type:
            x_vals.append(comp["position"]["x"])
            y_vals.append(comp["position"]["y"])
            texts.append(comp["name"])
    
    if not x_vals:  # Skip if no components of this type
        continue
    
    # Create legend names
    legend_names = {
        "data_source": "Data Sources",
        "processing": "Processing",
        "output": "Outputs"
    }
    
    # Add rectangular boxes as markers with larger size
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='markers+text',
        marker=dict(
            color=colors[comp_type],
            size=60,  # Increased size
            symbol='square',
            line=dict(color='white', width=2)
        ),
        text=texts,
        textposition="middle center",
        textfont=dict(color='white', size=9, family='Arial Bold'),  # Reduced font size
        name=legend_names[comp_type],
        hovertemplate='%{text}<extra></extra>'
    ))

# Add stage labels at the top
stage_labels = [
    {"text": "Data Sources", "x": 1, "y": 3.8},
    {"text": "ETL & Clean", "x": 3, "y": 3.8},
    {"text": "Analysis", "x": 5, "y": 3.8},
    {"text": "User Interface", "x": 7, "y": 3.8},
    {"text": "Export", "x": 9, "y": 3.8}
]

for label in stage_labels:
    fig.add_trace(go.Scatter(
        x=[label["x"]],
        y=[label["y"]],
        mode='text',
        text=[label["text"]],
        textfont=dict(size=10, color='#333333', family='Arial Bold'),
        showlegend=False,
        hoverinfo='none'
    ))

# Add process annotations
process_annotations = [
    {"text": "Validation", "x": 4, "y": 2.5},
    {"text": "Modeling", "x": 6, "y": 2}
]

for annotation in process_annotations:
    fig.add_trace(go.Scatter(
        x=[annotation["x"]],
        y=[annotation["y"]],
        mode='text',
        text=[annotation["text"]],
        textfont=dict(size=8, color='#666666', family='Arial'),
        showlegend=False,
        hoverinfo='none'
    ))

# Update layout
fig.update_layout(
    title="COVID-19 Dashboard Architecture",
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(range=[0, 10], showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(range=[0.3, 4.2], showgrid=False, showticklabels=False, zeroline=False)
)

# Update traces
fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image("covid_dashboard_architecture.png")