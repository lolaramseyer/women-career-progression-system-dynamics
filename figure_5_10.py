
"""
Created on Fri Mar 13 15:09:18 2026

@author: lolaramseyer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plot styling
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

# Core simulation model
def simulate(
    T=30,
    dt=1.0,
    J0=2000,
    M0=800,
    S0=250,
    L0=80,
    entry_per_year=300,

    pJM=0.10,
    pMS=0.045,
    pSL=0.020,

    aJ=0.08,
    aM=0.06,
    aS=0.05,
    aL=0.04,

    alpha=0.1,
    mentor_scale=3000,

    bias_JM=0.15,
    bias_MS=0.25,
    bias_SL=0.35,

    use_stagnation=True,
    gamma=0.4,
    cap_effective_promo=True
):
    steps = int(T / dt)
    years = np.arange(0, T + dt, dt)

    J = np.zeros(steps + 1)
    M = np.zeros(steps + 1)
    S = np.zeros(steps + 1)
    L = np.zeros(steps + 1)
    mentorship = np.zeros(steps + 1)

    J[0], M[0], S[0], L[0] = J0, M0, S0, L0

    for t in range(steps):

        mentorship[t] = 1.0 - np.exp(-(S[t] + L[t]) / mentor_scale)
        mentorship[t] = np.clip(mentorship[t], 0.0, 1.0)

        f_mentorship = 1.0 + alpha * mentorship[t]

        if use_stagnation:
            g_stagnation = 1.0 + gamma * (1.0 - mentorship[t])
        else:
            g_stagnation = 1.0

        eff_pJM = pJM * f_mentorship * (1 - bias_JM)
        eff_pMS = pMS * f_mentorship * (1 - bias_MS)
        eff_pSL = pSL * f_mentorship * (1 - bias_SL)

        if cap_effective_promo:
            eff_pJM = np.clip(eff_pJM, 0.0, 1.0)
            eff_pMS = np.clip(eff_pMS, 0.0, 1.0)
            eff_pSL = np.clip(eff_pSL, 0.0, 1.0)

        prom_JM = J[t] * eff_pJM
        prom_MS = M[t] * eff_pMS
        prom_SL = S[t] * eff_pSL

        attr_J = J[t] * aJ * g_stagnation
        attr_M = M[t] * aM * g_stagnation
        attr_S = S[t] * aS * g_stagnation
        attr_L = L[t] * aL * g_stagnation

        J[t + 1] = max(J[t] + dt * (entry_per_year - prom_JM - attr_J), 0.0)
        M[t + 1] = max(M[t] + dt * (prom_JM - prom_MS - attr_M), 0.0)
        S[t + 1] = max(S[t] + dt * (prom_MS - prom_SL - attr_S), 0.0)
        L[t + 1] = max(L[t] + dt * (prom_SL - attr_L), 0.0)

    df = pd.DataFrame({
        "Year": years,
        "Junior": J,
        "Mid-level": M,
        "Senior": S,
        "Leadership": L
    })

    total = df["Junior"] + df["Mid-level"] + df["Senior"] + df["Leadership"]
    df["Senior_Leadership_share"] = 100 * (df["Senior"] + df["Leadership"]) / total

    return df

# Baseline scenario low mentorship + baseline stage-specific bias
baseline = simulate(
    alpha=0.1,
    bias_JM=0.15,
    bias_MS=0.25,
    bias_SL=0.35
)

# Selected scenario comparison
scenarios = [
    {
        "label": "Low mentorship + 30% bias",
        "alpha": 0.1,
        "bias_JM": 0.30,
        "bias_MS": 0.30,
        "bias_SL": 0.30
    },
    {
        "label": "Moderate mentorship + 10% bias",
        "alpha": 0.3,
        "bias_JM": 0.10,
        "bias_MS": 0.10,
        "bias_SL": 0.10
    },
    {
        "label": "High mentorship + 10% bias",
        "alpha": 0.6,
        "bias_JM": 0.10,
        "bias_MS": 0.10,
        "bias_SL": 0.10
    },
    {
        "label": "Very high mentorship + 0% bias",
        "alpha": 0.9,
        "bias_JM": 0.00,
        "bias_MS": 0.00,
        "bias_SL": 0.00
    },
]

fig, ax = plt.subplots(figsize=(9, 5.5))

for scenario in scenarios:
    df = simulate(
        alpha=scenario["alpha"],
        bias_JM=scenario["bias_JM"],
        bias_MS=scenario["bias_MS"],
        bias_SL=scenario["bias_SL"]
    )

    diff = df["Senior_Leadership_share"] - baseline["Senior_Leadership_share"]

    ax.plot(
        df["Year"],
        diff,
        label=scenario["label"]
    )

ax.axhline(0, linewidth=1.5, linestyle="--")
ax.set_title("Selected Scenario Comparison Relative to Baseline")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Difference in senior + leadership share\nfrom baseline (percentage points)")
ax.set_xlim(0, 30)
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig("figure_5_10_scenario_comparison_relative_to_baseline.png", bbox_inches="tight")
plt.show()

