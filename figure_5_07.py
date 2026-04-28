#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:00:09 2026

@author: lolaramseyer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------
# Plot styling
# ---------------------------------
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

# ---------------------------------
# Core simulation model
# ---------------------------------
def simulate(
    T=30,
    dt=1.0,
    J0=2000, M0=800, S0=250, L0=80,
    entry_per_year=300,
    pJM=0.12, pMS=0.10, pSL=0.08,
    aJ=0.08, aM=0.06, aS=0.05, aL=0.04,
    alpha=0.3,
    bias_JM=0.15, bias_MS=0.15, bias_SL=0.15,
    kappa=0.002
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
        total_prev = J[t-1] + M[t-1] + S[t-1] + L[t-1]

        # Mentorship availability rises with senior + leadership representation
        mentorship[t-1] = alpha * (1 - np.exp(-kappa * (S[t-1] + L[t-1])))

        # Effective promotion rates
        eff_pJM = pJM * (1 - bias_JM) * (1 + mentorship[t-1])
        eff_pMS = pMS * (1 - bias_MS) * (1 + mentorship[t-1])
        eff_pSL = pSL * (1 - bias_SL) * (1 + mentorship[t-1])

        # Keep rates within sensible bounds
        eff_pJM = min(eff_pJM, 1.0)
        eff_pMS = min(eff_pMS, 1.0)
        eff_pSL = min(eff_pSL, 1.0)

        # Flows
        entry = entry_per_year * dt

        prom_JM = eff_pJM * J[t-1] * dt
        prom_MS = eff_pMS * M[t-1] * dt
        prom_SL = eff_pSL * S[t-1] * dt

        attr_J = aJ * J[t-1] * dt
        attr_M = aM * M[t-1] * dt
        attr_S = aS * S[t-1] * dt
        attr_L = aL * L[t-1] * dt

        # Stock updates
        J[t] = J[t-1] + entry - prom_JM - attr_J
        M[t] = M[t-1] + prom_JM - prom_MS - attr_M
        S[t] = S[t-1] + prom_MS - prom_SL - attr_S
        L[t] = L[t-1] + prom_SL - attr_L

        # Prevent negative values
        J[t] = max(J[t], 0)
        M[t] = max(M[t], 0)
        S[t] = max(S[t], 0)
        L[t] = max(L[t], 0)

    # Final mentorship value
    mentorship[-1] = alpha * (1 - np.exp(-kappa * (S[-1] + L[-1])))

    df = pd.DataFrame({
        "Time": time,
        "Junior": J,
        "Mid": M,
        "Senior": S,
        "Leadership": L,
        "Mentorship": mentorship
    })

    total = df["Junior"] + df["Mid"] + df["Senior"] + df["Leadership"]
    df["Senior_Leadership_share"] = 100 * (df["Senior"] + df["Leadership"]) / total

    return df

# ---------------------------------
# Promotion bias sensitivity analysis
# ---------------------------------
bias_values = [0.00, 0.10, 0.20, 0.30]
results = {}

for bias in bias_values:
    results[bias] = simulate(
        T=30,
        dt=1.0,
        J0=2000, M0=800, S0=250, L0=80,
        entry_per_year=300,
        pJM=0.12, pMS=0.10, pSL=0.08,
        aJ=0.08, aM=0.06, aS=0.05, aL=0.04,
        alpha=0.3,          # hold mentorship constant
        bias_JM=bias,
        bias_MS=bias,
        bias_SL=bias,
        kappa=0.002
    )

# ---------------------------------
# Plot Figure 5.7
# ---------------------------------
fig, ax = plt.subplots()

for bias, df in results.items():
    ax.plot(df["Time"], df["Senior_Leadership_share"], label=f"{int(bias*100)}% bias")

ax.set_title("Senior + Leadership Share Over Time Under Different Promotion Bias Levels")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Senior + Leadership share (%)")
ax.set_xlim(0, 30)
ax.legend(title="Promotion bias")

plt.tight_layout()
plt.savefig("figure_5_7_bias_sensitivity.png", bbox_inches="tight")
plt.show()
