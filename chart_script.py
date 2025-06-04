import plotly.graph_objects as go
import json

# Data from the provided JSON
data = {
  "metrics": ["FPS Performance", "Accuracy", "Resource Efficiency", "Setup Ease", "Cross-platform Support", "Real-time Performance"],
  "dlib_scores": [3, 5, 3, 2, 3, 3],
  "mediapipe_scores": [5, 4, 5, 5, 5, 5],
  "max_score": 5
}

# Abbreviate metrics to fit 15 character limit
abbreviated_metrics = [
    "FPS Perf",
    "Accuracy", 
    "Resource Eff",
    "Setup Ease",
    "Cross-platform",
    "Real-time Perf"
]

# Create radar chart
fig = go.Figure()

# Add Dlib trace
fig.add_trace(go.Scatterpolar(
    r=data["dlib_scores"] + [data["dlib_scores"][0]],  # Close the polygon
    theta=abbreviated_metrics + [abbreviated_metrics[0]],  # Close the polygon
    fill='toself',
    name='Dlib',
    line_color='#1FB8CD',
    fillcolor='rgba(31, 184, 205, 0.3)'
))

# Add MediaPipe trace
fig.add_trace(go.Scatterpolar(
    r=data["mediapipe_scores"] + [data["mediapipe_scores"][0]],  # Close the polygon
    theta=abbreviated_metrics + [abbreviated_metrics[0]],  # Close the polygon
    fill='toself',
    name='MediaPipe',
    line_color='#FFC185',
    fillcolor='rgba(255, 193, 133, 0.3)'
))

# Update layout
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, data["max_score"]],
            tickvals=[1, 2, 3, 4, 5]
        )
    ),
    title="Dlib vs MediaPipe Performance",
    legend=dict(
        orientation='h', 
        yanchor='bottom', 
        y=1.05, 
        xanchor='center', 
        x=0.5
    )
)

# Save the chart
fig.write_image("drowsiness_detection_comparison.png")

print("Chart saved successfully!")