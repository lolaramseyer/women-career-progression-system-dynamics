#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:33:48 2026

@author: lolaramseyer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plot styling
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "figure.figsize": (8.5, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Core simulation model
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

    total = df["Junior"] + df["Mid"] + df["Senior"] + df["Leadership"]
    df["Leadership_share"] = 100 * df["Leadership"] / total
    df["Senior_Leadership_share"] = 100 * (df["Senior"] + df["Leadership"]) / total

    return df

# Interaction matrix settings
alpha_values = [0.1, 0.3, 0.6, 0.9]
bias_values = [0.0, 0.1, 0.2, 0.3]

alpha_labels = ["0.1", "0.3", "0.6", "0.9"]
bias_labels = ["0%", "10%", "20%", "30%"]

# Build heatmap matrix
heatmap_data = np.zeros((len(bias_values), len(alpha_values)))

for i, bias in enumerate(bias_values):
    for j, alpha in enumerate(alpha_values):
        df = simulate(
            alpha=alpha,
            bias_JM=bias,
            bias_MS=bias,
            bias_SL=bias,
            kappa=0.002
        )
        heatmap_data[i, j] = df["Leadership_share"].iloc[-1]

# Plot heatmap
fig, ax = plt.subplots(figsize=(8.5, 6))
im = ax.imshow(heatmap_data, cmap="Blues", aspect="auto")

# Axis labels and ticks
ax.set_xticks(np.arange(len(alpha_labels)))
ax.set_yticks(np.arange(len(bias_labels)))
ax.set_xticklabels(alpha_labels)
ax.set_yticklabels(bias_labels)

ax.set_xlabel("Mentorship level (α)")
ax.set_ylabel("Promotion bias level")
ax.set_title("Interaction Effects of Mentorship and Promotion Bias on Leadership Representation at Year 30")

# Add cell values
for i in range(len(bias_values)):
    for j in range(len(alpha_values)):
        value = heatmap_data[i, j]
        ax.text(
            j, i, f"{value:.1f}",
            ha="center", va="center",
            color="black", fontsize=11
        )

# Color bar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Leadership share at year 30 (%)")

plt.tight_layout()
plt.show()

# Print table of values
interaction_table = pd.DataFrame(
    heatmap_data,
    index=[f"Bias {b}" for b in bias_labels],
    columns=[f"α = {a}" for a in alpha_labels]
)

print("\nLeadership share at year 30 (%)")
print(interaction_table.round(2))
