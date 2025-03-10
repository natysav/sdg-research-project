import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ---------------------------
# Data Preparation
# ---------------------------

# Generate 60 monthly dates over 5 years
dates = pd.date_range(start='2018-01-01', periods=60, freq='MS')
dates_str = dates.strftime('%Y-%m').tolist()

# Simulated revenue data (replace with your actual revenue)
np.random.seed(1)
revenue = np.linspace(50, 200, 60) + np.random.normal(0, 5, 60)

# Generate sample sustainability scores for each SDG (17 lines)
num_sdgs = 17
sdg_scores = []
for i in range(num_sdgs):
    base = np.linspace(70, 90, 60)
    offset = i * 0.5  # slight offset per SDG
    noise = np.random.normal(0, 1.5, 60)
    sdg_scores.append(base + offset + noise)

# Compute an aggregated sustainability score (average of the 17 SDG scores)
agg_score = np.mean(sdg_scores, axis=0)

# UN SDG Colors (official colors)
sdg_colors = {
    1: "#E5243B",   2: "#DDA63A",  3: "#4C9F38",  4: "#C5192D",  5: "#FF3A21",
    6: "#26BDE2",   7: "#FCC30B",  8: "#A21942",  9: "#FD6925", 10: "#DD1367",
   11: "#FD9D24",  12: "#BF8B2E", 13: "#3F7E44", 14: "#0A97D9", 15: "#56C02B",
   16: "#00689D", 17: "#19486A"
}

# ---------------------------
# Create Traces
# ---------------------------

# Trace 0: Revenue vs. Date (2D) view – flat graph (z=0 for all points)
trace_2d = go.Scatter3d(
    x=dates_str,
    y=revenue,
    z=[0]*len(dates_str),
    mode='lines+markers',
    marker=dict(size=5, color='blue'),
    line=dict(width=4, color='blue'),
    name='Revenue vs. Date (2D)'
)

# Trace 1: Revenue vs. Date vs. Aggregated Sustainability Score (3D)
trace_agg = go.Scatter3d(
    x=dates_str,
    y=revenue,
    z=agg_score,
    mode='lines+markers',
    marker=dict(size=5, color='#005187'),
    line=dict(width=4, color='#005187'),
    name='Aggregated Sustainability Score'
)

# Traces 2 to 18: One trace per SDG (17 traces)
sdg_traces = []
for sdg in range(1, num_sdgs+1):
    trace = go.Scatter3d(
        x=dates_str,
        y=revenue,
        z=sdg_scores[sdg-1],
        mode='lines+markers',
        marker=dict(size=5, color=sdg_colors.get(sdg, '#000000')),
        line=dict(width=3, color=sdg_colors.get(sdg, '#000000')),
        name=f'SDG {sdg}'
    )
    sdg_traces.append(trace)

# Combine all traces.
# Order: index 0 = 2D view, index 1 = aggregated 3D, indices 2-18 = 17 SDG 3D traces.
data = [trace_2d, trace_agg] + sdg_traces

fig = go.Figure(data=data)

# ---------------------------
# Set Up Update Menus
# ---------------------------
# Define visibility arrays for the three modes.
visibility_2d  = [True] + [False]*(1 + num_sdgs)
visibility_agg = [False, True] + [False]*num_sdgs
visibility_sdg = [False, False] + [True]*num_sdgs

# Common camera settings to fix the vertical direction.
common_camera = {
    "up": {"x": 0, "y": 0, "z": 1}
}

# Update menu buttons:
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            x=0.5,
            y=1.15,
            buttons=[
                dict(
                    label="Revenue vs. Date (2D)",
                    method="update",
                    args=[
                        {"visible": visibility_2d},
                        {"title": "Revenue vs. Date (2D)",
                         "scene": {
                             "camera": {
                                 "eye": {"x": 0, "y": 0, "z": 2},
                                 "projection": {"type": "orthographic"},
                                 **common_camera
                             },
                             "zaxis": {"visible": False}
                         }
                        }
                    ]
                ),
                dict(
                    label="Revenue vs. Date vs. Aggregated Score (3D)",
                    method="update",
                    args=[
                        {"visible": visibility_agg},
                        {"title": "Revenue vs. Date vs. Aggregated Sustainability Score",
                         "scene": {
                             "camera": {
                                 "eye": {"x": 1.25, "y": 1.25, "z": 1.25},
                                 "projection": {"type": "perspective"},
                                 **common_camera
                             },
                             "zaxis": {"visible": True}
                         }
                        }
                    ]
                ),
                dict(
                    label="Revenue vs. Date vs. 17 SDG (3D)",
                    method="update",
                    args=[
                        {"visible": visibility_sdg},
                        {"title": "Revenue vs. Date vs. 17 SDG",
                         "scene": {
                             "camera": {
                                 "eye": {"x": 1.25, "y": 1.25, "z": 1.25},
                                 "projection": {"type": "perspective"},
                                 **common_camera
                             },
                             "zaxis": {"visible": True}
                         }
                        }
                    ]
                )
            ],
            pad={"r": 10, "t": 10},
            showactive=True
        )
    ],
    scene=dict(
        xaxis=dict(title="Date"),
        yaxis=dict(title="Revelue"),
        zaxis=dict(title="SDG Score", visible=True),
        aspectmode="manual",
        aspectratio=dict(x=4, y=1, z=1)
    ),
    margin=dict(l=0, r=0, b=0, t=50)
)

# Set initial view title.
fig.update_layout(title="Revenue vs. Date (2D)")

# ---------------------------
# Display the Figure
# ---------------------------
fig.show()