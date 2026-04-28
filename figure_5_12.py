
"""
@author: lolaramseyer

Figure 5.12: Leadership representation at year 30 under intervention scenarios
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "figure.figsize": (9, 5.5),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

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

    J[0], M[0], S[0], L[0] = J0, M0, S0, L0

    for t in range(steps):
        mentorship = 1.0 - np.exp(-(S[t] + L[t]) / mentor_scale)
        mentorship = np.clip(mentorship, 0.0, 1.0)

        f_mentorship = 1.0 + alpha * mentorship

        if use_stagnation:
            g_stagnation = 1.0 + gamma * (1.0 - mentorship)
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
    df["Leadership_share"] = 100 * df["Leadership"] / total
    df["Senior_Leadership_share"] = 100 * (df["Senior"] + df["Leadership"]) / total

    return df

# Intervention scenarios
scenarios = [
    {
        "label": "Baseline",
        "alpha": 0.1,
        "bias_JM": 0.15,
        "bias_MS": 0.25,
        "bias_SL": 0.35
    },
    {
        "label": "Mentorship only",
        "alpha": 0.6,
        "bias_JM": 0.15,
        "bias_MS": 0.25,
        "bias_SL": 0.35
    },
    {
        "label": "Bias reduction only",
        "alpha": 0.1,
        "bias_JM": 0.05,
        "bias_MS": 0.10,
        "bias_SL": 0.15
    },
    {
        "label": "Combined",
        "alpha": 0.6,
        "bias_JM": 0.05,
        "bias_MS": 0.10,
        "bias_SL": 0.15
    },
]

labels = []
leadership_share_year30 = []
raw_rows = []

for scenario in scenarios:
    df = simulate(
        alpha=scenario["alpha"],
        bias_JM=scenario["bias_JM"],
        bias_MS=scenario["bias_MS"],
        bias_SL=scenario["bias_SL"]
    )

    final = df.iloc[-1]

    labels.append(scenario["label"])
    leadership_share_year30.append(final["Leadership_share"])

    raw_rows.append({
        "Scenario": scenario["label"],
        "Junior": round(final["Junior"], 2),
        "Mid-level": round(final["Mid-level"], 2),
        "Senior": round(final["Senior"], 2),
        "Leadership": round(final["Leadership"], 2),
        "Total": round(
            final["Junior"] + final["Mid-level"] + final["Senior"] + final["Leadership"],
            2
        )
    })

# Plot Figure 5.12
fig, ax = plt.subplots(figsize=(9, 5.5))

bars = ax.bar(labels, leadership_share_year30)

for bar, value in zip(bars, leadership_share_year30):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(leadership_share_year30) * 0.02,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=11
    )

ax.set_title("Final Leadership Representation Under Intervention Scenarios")
ax.set_xlabel("Intervention scenario")
ax.set_ylabel("Leadership share at year 30 (%)")
ax.set_ylim(0, max(leadership_share_year30) * 1.2)

plt.tight_layout()
plt.savefig("figure_5_12_intervention_bar_chart.png", bbox_inches="tight")
plt.show()

# Print raw year-30 values
raw_outputs = pd.DataFrame(raw_rows)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

print("\nRaw year 30 outputs for manual Table 5.5 calculations:\n")
print(raw_outputs.to_string(index=False))
