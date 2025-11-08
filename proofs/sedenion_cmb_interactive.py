import numpy as np
import plotly.graph_objects as go

# Sedenion 15-mode CMB modulation
l = np.arange(2, 3000)
omega = 2 * np.pi * 43  # 43 Hz

# 15 sedenion imaginary units
e_sed = np.linspace(1.0, 0.1, 15)

# Baseline CMB
Cl = 1e-3 * l * (l + 1) * np.exp(-l / 1000.0)

# Sedenion modulation
Cl_sed = np.zeros_like(l, dtype=float)
for i in range(15):
    phase = omega * l * (i + 1) / 1000.0
    Cl_sed += e_sed[i] * Cl * np.sin(phase)**2

# Aladin v∞ damping + peaks
Cl_aladin = Cl_sed * np.exp(-l / 800.0) * np.sin(l / 220.0)**2

# Interactive plot
fig = go.Figure()

fig.add_trace(go.Scatter(x=l, y=Cl, mode='lines', name='Baseline CMB',
                         line=dict(color='gray', dash='dash')))
fig.add_trace(go.Scatter(x=l, y=Cl_sed, mode='lines', name='Sedenion 15 Modes',
                         line=dict(color='orange')))
fig.add_trace(go.Scatter(x=l, y=Cl_aladin, mode='lines', name='Aladin v∞ + Sedenion',
                         line=dict(color='gold', width=3)))

fig.update_layout(
    title="Sedenion 15-Mode CMB Modulation — Interactive",
    xaxis_title="Multipole l",
    yaxis_title="Cₗ [μK²]",
    yaxis=dict(type='log'),
    hovermode="x unified",
    template="plotly_dark",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

fig.write_html("sedenion_cmb_interactive.html")
print("Interactive plot saved: sedenion_cmb_interactive.html")
