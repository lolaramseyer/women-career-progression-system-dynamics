#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:42:46 2026

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

# -----------------------------
# Core simulation model
# -----------------------------
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
        mentorship[t-1] = alpha * (1 - np.exp(-kappa * (S[t-1] + L[t-1])))

        eff_pJM = pJM * (1 - bias_JM) * (1 + mentorship[t-1])
        eff_pMS = pMS * (1 - bias_MS) * (1 + mentorship[t-1])
        eff_pSL = pSL * (1 - bias_SL) * (1 + mentorship[t-1])

        eff_pJM = min(eff_pJM, 1.0)
        eff_pMS = min(eff_pMS, 1.0)
        eff_pSL = min(eff_pSL, 1.0)

        entry = entry_per_year * dt

        prom_JM = eff_pJM * J[t-1] * dt
        prom_MS = eff_pMS * M[t-1] * dt
        prom_SL = eff_pSL * S[t-1] * dt

        attr_J = aJ * J[t-1] * dt
        attr_M = aM * M[t-1] * dt
        attr_S = aS * S[t-1] * dt
        attr_L = aL * L[t-1] * dt

        J[t] = J[t-1] + entry - prom_JM - attr_J
        M[t] = M[t-1] + prom_JM - prom_MS - attr_M
        S[t] = S[t-1] + prom_MS - prom_SL - attr_S
        L[t] = L[t-1] + prom_SL - attr_L

        J[t] = max(J[t], 0)
        M[t] = max(M[t], 0)
        S[t] = max(S[t], 0)
        L[t] = max(L[t], 0)

    mentorship[-1] = alpha * (1 - np.exp(-kappa * (S[-1] + L[-1])))

    df = pd.DataFrame({
        "Time": time,
        "Junior": J,
        "Mid": M,
        "Senior": S,
        "Leadership": L,
        "Mentorship": mentorship
    })

    return df

# -----------------------------
# Figure 5.8
# Leadership at year 30 by bias level
# -----------------------------
bias_values = [0.00, 0.10, 0.20, 0.30]
labels = ["0%", "10%", "20%", "30%"]

leadership_year30 = []

for bias in bias_values:
    df = simulate(
        alpha=0.3,          # hold mentorship constant
        bias_JM=bias,
        bias_MS=bias,
        bias_SL=bias
    )
    leadership_year30.append(df["Leadership"].iloc[-1])

# -----------------------------
# Plot bar chart
# -----------------------------
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

ax.set_title("Leadership Representation at Year 30 by Promotion Bias Level")
ax.set_xlabel("Promotion bias level")
ax.set_ylabel("Women in leadership at year 30")

plt.tight_layout()
plt.show()

# -----------------------------
# Print exact values
# -----------------------------
results = pd.DataFrame({
    "Promotion bias level": labels,
    "Bias value": bias_values,
    "Leadership at year 30": leadership_year30
})

print(results)import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Plot styling
# -----------------------------
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

# -----------------------------
# Core simulation model
# -----------------------------
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
        mentorship[t-1] = alpha * (1 - np.exp(-kappa * (S[t-1] + L[t-1])))

        eff_pJM = pJM * (1 - bias_JM) * (1 + mentorship[t-1])
        eff_pMS = pMS * (1 - bias_MS) * (1 + mentorship[t-1])
        eff_pSL = pSL * (1 - bias_SL) * (1 + mentorship[t-1])

        eff_pJM = min(eff_pJM, 1.0)
        eff_pMS = min(eff_pMS, 1.0)
        eff_pSL = min(eff_pSL, 1.0)

        entry = entry_per_year * dt

        prom_JM = eff_pJM * J[t-1] * dt
        prom_MS = eff_pMS * M[t-1] * dt
        prom_SL = eff_pSL * S[t-1] * dt

        attr_J = aJ * J[t-1] * dt
        attr_M = aM * M[t-1] * dt
        attr_S = aS * S[t-1] * dt
        attr_L = aL * L[t-1] * dt

        J[t] = J[t-1] + entry - prom_JM - attr_J
        M[t] = M[t-1] + prom_JM - prom_MS - attr_M
        S[t] = S[t-1] + prom_MS - prom_SL - attr_S
        L[t] = L[t-1] + prom_SL - attr_L

        J[t] = max(J[t], 0)
        M[t] = max(M[t], 0)
        S[t] = max(S[t], 0)
        L[t] = max(L[t], 0)

    mentorship[-1] = alpha * (1 - np.exp(-kappa * (S[-1] + L[-1])))

    df = pd.DataFrame({
        "Time": time,
        "Junior": J,
        "Mid": M,
        "Senior": S,
        "Leadership": L,
        "Mentorship": mentorship
    })

    return df

# -----------------------------
# Figure 5.8
# Leadership at year 30 by bias level
# -----------------------------
bias_values = [0.00, 0.10, 0.20, 0.30]
labels = ["0%", "10%", "20%", "30%"]

leadership_year30 = []

for bias in bias_values:
    df = simulate(
        alpha=0.3,          # hold mentorship constant
        bias_JM=bias,
        bias_MS=bias,
        bias_SL=bias
    )
    leadership_year30.append(df["Leadership"].iloc[-1])

# -----------------------------
# Plot bar chart
# -----------------------------
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

ax.set_title("Leadership Representation at Year 30 by Promotion Bias Level")
ax.set_xlabel("Promotion bias level")
ax.set_ylabel("Women in leadership at year 30")

plt.tight_layout()
plt.show()

# -----------------------------
# Print exact values
# -----------------------------
results = pd.DataFrame({
    "Promotion bias level": labels,
    "Bias value": bias_values,
    "Leadership at year 30": leadership_year30
})

print(results)