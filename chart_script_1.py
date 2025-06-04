import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

# Eye landmark data
landmarks = {
    "P1": {"x": 0, "y": 0, "label": "P1"},
    "P2": {"x": 1, "y": -0.5, "label": "P2"},
    "P3": {"x": 2, "y": -0.5, "label": "P3"},
    "P4": {"x": 3, "y": 0, "label": "P4"},
    "P5": {"x": 2, "y": 0.5, "label": "P5"},
    "P6": {"x": 1, "y": 0.5, "label": "P6"}
}

# Create figure
fig = go.Figure()

# Extract coordinates
x_coords = [landmarks[p]["x"] for p in landmarks.keys()]
y_coords = [landmarks[p]["y"] for p in landmarks.keys()]
labels = [landmarks[p]["label"] for p in landmarks.keys()]

# Add landmark points with larger text
fig.add_trace(go.Scatter(
    x=x_coords,
    y=y_coords,
    mode='markers+text',
    marker=dict(size=15, color='#1FB8CD'),
    text=labels,
    textposition='top center',
    textfont=dict(size=14, color='black'),
    name='Eye Points',
    showlegend=False
))

# Eye outline (connecting P1->P2->P3->P4->P5->P6->P1)
eye_outline_x = [0, 1, 2, 3, 2, 1, 0]
eye_outline_y = [0, -0.5, -0.5, 0, 0.5, 0.5, 0]

fig.add_trace(go.Scatter(
    x=eye_outline_x,
    y=eye_outline_y,
    mode='lines',
    line=dict(color='#5D878F', width=3),
    name='Eye Shape',
    showlegend=False
))

# Vertical distance lines for EAR calculation
# P2 to P6 vertical line
fig.add_trace(go.Scatter(
    x=[1, 1],
    y=[-0.5, 0.5],
    mode='lines',
    line=dict(color='#FFC185', width=4, dash='dot'),
    name='P2-P6 Distance',
    showlegend=True
))

# P3 to P5 vertical line  
fig.add_trace(go.Scatter(
    x=[2, 2],
    y=[-0.5, 0.5],
    mode='lines',
    line=dict(color='#ECEBD5', width=4, dash='dot'),
    name='P3-P5 Distance',
    showlegend=True
))

# Horizontal distance line P1 to P4
fig.add_trace(go.Scatter(
    x=[0, 3],
    y=[0, 0],
    mode='lines',
    line=dict(color='#D2BA4C', width=4, dash='dashdot'),
    name='P1-P4 Distance',
    showlegend=True
))

# Update layout
fig.update_layout(
    title='Eye Aspect Ratio Calculation',
    xaxis_title='X Coord',
    yaxis_title='Y Coord',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

fig.update_xaxes(range=[-0.5, 3.5], showgrid=True)
fig.update_yaxes(range=[-0.8, 0.8], showgrid=True)

# Save the chart
fig.write_image("ear_diagram.png")