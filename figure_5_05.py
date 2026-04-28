#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:30:25 2026

@author: lolaramseyer
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plot styling
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "figure.figsize": (8, 5.5),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.8,
    "lines.linewidth": 2.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Core simulation model
def simulate(
    T=30,
    dt=1.0,

    # Initial stocks
    J0=2000, M0=800, S0=250, L0=80,

    # Entry flow
    entry_per_year=300,

    # Base promotion rates
    pJM=0.12, pMS=0.10, pSL=0.08,

    # Attrition rates
    aJ=0.08, aM=0.06, aS=0.05, aL=0.04,

    # Mentorship settings
    alpha=0.3,         # mentorship strength
    kappa=400.0,       # saturation constant

    # Promotion bias (held constant here)
    bias_JM=0.15,
    bias_MS=0.15,
    bias_SL=0.15
):
    n_steps = int(T / dt) + 1
    time = np.arange(0, T + dt, dt)

    J = np.zeros(n_steps)
    M = np.zeros(n_steps)
    S = np.zeros(n_steps)
    L = np.zeros(n_steps)
    mentorship = np.zeros(n_steps)

    J[0], M[0], S[0], L[0] = J0, M0, S0, L0

    for t in range(1, n_steps):
        # mentorship availability rises with senior + leadership presence
        upper_levels = S[t-1] + L[t-1]
        mentorship[t-1] = alpha * (upper_levels / (upper_levels + kappa))

        # promotion flows
        promote_JM = pJM * (1 - bias_JM) * (1 + mentorship[t-1]) * J[t-1]
        promote_MS = pMS * (1 - bias_MS) * (1 + mentorship[t-1]) * M[t-1]
        promote_SL = pSL * (1 - bias_SL) * (1 + mentorship[t-1]) * S[t-1]

        # attrition flows
        attrit_J = aJ * J[t-1]
        attrit_M = aM * M[t-1]
        attrit_S = aS * S[t-1]
        attrit_L = aL * L[t-1]

        # stock updates
        J[t] = J[t-1] + (entry_per_year - promote_JM - attrit_J) * dt
        M[t] = M[t-1] + (promote_JM - promote_MS - attrit_M) * dt
        S[t] = S[t-1] + (promote_MS - promote_SL - attrit_S) * dt
        L[t] = L[t-1] + (promote_SL - attrit_L) * dt

        # prevent negative values
        J[t] = max(J[t], 0)
        M[t] = max(M[t], 0)
        S[t] = max(S[t], 0)
        L[t] = max(L[t], 0)

    # final mentorship value
    upper_levels = S[-1] + L[-1]
    mentorship[-1] = alpha * (upper_levels / (upper_levels + kappa))

    df = pd.DataFrame({
        "Year": time,
        "Junior": J,
        "Mid": M,
        "Senior": S,
        "Leadership": L,
        "Mentorship": mentorship
    })

    return df

# Figure 5.5 Leadership at year 30 by mentorship level
alpha_values = [0.1, 0.3, 0.6, 0.9]
labels = ["Low", "Moderate", "High", "Very high"]

leadership_year30 = []

for alpha in alpha_values:
    df = simulate(alpha=alpha)
    leadership_year30.append(df["Leadership"].iloc[-1])

# Plot bar chart
fig, ax = plt.subplots(figsize=(8, 5.5))
bars = ax.bar(labels, leadership_year30)

# Add value labels
for bar, value in zip(bars, leadership_year30):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(leadership_year30) * 0.015,
        f"{value:.0f}",
        ha="center",
        va="bottom",
        fontsize=11
    )

ax.set_title("Leadership Representation at Year 30 by Mentorship Level")
ax.set_xlabel("Mentorship level")
ax.set_ylabel("Women in leadership at year 30")

plt.tight_layout()
plt.show()

# Print exact values and gains
results = pd.DataFrame({
    "Mentorship level": labels,
    "Alpha": alpha_values,
    "Leadership at year 30": leadership_year30
})

results["Step gain"] = results["Leadership at year 30"].diff()

print(results)
