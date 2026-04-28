
"""
Created on Tue Mar 10 15:55:16 2026

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
        # Mentorship availability
        mentorship[t-1] = alpha * (1 - np.exp(-kappa * (S[t-1] + L[t-1])))

        # Effective promotion rates
        eff_pJM = pJM * (1 - bias_JM) * (1 + mentorship[t-1])
        eff_pMS = pMS * (1 - bias_MS) * (1 + mentorship[t-1])
        eff_pSL = pSL * (1 - bias_SL) * (1 + mentorship[t-1])

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

        # Prevent negatives
        J[t] = max(J[t], 0)
        M[t] = max(M[t], 0)
        S[t] = max(S[t], 0)
        L[t] = max(L[t], 0)

    mentorship[-1] = alpha * (1 - np.exp(-kappa * (S[-1] + L[-1])))

    df = pd.DataFrame({
        "Year": time,
        "Junior": J,
        "Mid": M,
        "Senior": S,
        "Leadership": L,
        "Mentorship": mentorship
    })

    return df

# Figure 5.6 Mentorship availability over time
alpha_values = [0.1, 0.3, 0.6, 0.9]
labels = [
    "Low (α = 0.1)",
    "Moderate (α = 0.3)",
    "High (α = 0.6)",
    "Very high (α = 0.9)"
]

fig, ax = plt.subplots()

summary_rows = []

for alpha, label in zip(alpha_values, labels):
    df = simulate(alpha=alpha)
    ax.plot(df["Year"], df["Mentorship"], label=label)

    summary_rows.append({
        "Mentorship level": label,
        "Mentorship at year 0": round(df["Mentorship"].iloc[0], 4),
        "Mentorship at year 30": round(df["Mentorship"].iloc[-1], 4)
    })

ax.set_title("Mentorship Availability Over Time under Different Mentorship Strength Scenarios")
ax.set_xlabel("Year")
ax.set_ylabel("Mentorship availability index")
ax.set_xlim(0, 30)
ax.set_ylim(bottom=0)
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig("figure_5_6_mentorship_availability_over_time.png", bbox_inches="tight")
plt.show()

# Print simple check values
summary_df = pd.DataFrame(summary_rows)

print("\nFigure 5.6 mentorship values:\n")
print(summary_df.to_string(index=False))
