#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:55:16 2026

@author: lolaramseyer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Plot styling
# -----------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "figure.figsize": (9, 5.5),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.8,
    "lines.linewidth": 2.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# -----------------------------
# Core simulation model
# -----------------------------
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

    # Mentorship strength
    alpha=0.3,
    kappa=400.0,

    # Promotion bias
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
        # Mentorship availability depends on senior + leadership presence
        upper_levels = S[t-1] + L[t-1]
        mentorship[t-1] = alpha * (upper_levels / (upper_levels + kappa))

        # Promotion flows
        promote_JM = pJM * (1 - bias_JM) * (1 + mentorship[t-1]) * J[t-1]
        promote_MS = pMS * (1 - bias_MS) * (1 + mentorship[t-1]) * M[t-1]
        promote_SL = pSL * (1 - bias_SL) * (1 + mentorship[t-1]) * S[t-1]

        # Attrition flows
        attrit_J = aJ * J[t-1]
        attrit_M = aM * M[t-1]
        attrit_S = aS * S[t-1]
        attrit_L = aL * L[t-1]

        # Stock updates
        J[t] = J[t-1] + (entry_per_year - promote_JM - attrit_J) * dt
        M[t] = M[t-1] + (promote_JM - promote_MS - attrit_M) * dt
        S[t] = S[t-1] + (promote_MS - promote_SL - attrit_S) * dt
        L[t] = L[t-1] + (promote_SL - attrit_L) * dt

        # Prevent negatives
        J[t] = max(J[t], 0)
        M[t] = max(M[t], 0)
        S[t] = max(S[t], 0)
        L[t] = max(L[t], 0)

    # Final mentorship value
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

# -----------------------------
# Figure 5.6
# Mentorship availability over time
# -----------------------------
alpha_values = [0.1, 0.3, 0.6, 0.9]
labels = ["Low (α = 0.1)", "Moderate (α = 0.3)", "High (α = 0.6)", "Very High (α = 0.9)"]

fig, ax = plt.subplots(figsize=(9, 5.5))

for alpha, label in zip(alpha_values, labels):
    df = simulate(alpha=alpha)
    ax.plot(df["Year"], df["Mentorship"], label=label)

ax.set_title("Mentorship Availability Over Time under Different Mentorship Strenght Scenarios")
ax.set_xlabel("Year")
ax.set_ylabel("Mentorship availability index ")
ax.set_xlim(0, 30)
ax.set_ylim(bottom=0)
ax.legend(frameon=False)

plt.tight_layout()
plt.show()
