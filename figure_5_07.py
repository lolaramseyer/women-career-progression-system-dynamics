
"""
Created on Thu Mar 12 14:00:09 2026

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

# Core simulation model Same as corrected baseline
def simulate(
    T=30,
    dt=1.0,
    J0=2000,
    M0=800,
    S0=250,
    L0=80,
    entry_per_year=300,

    # corrected baseline progression rates
    pJM=0.10,
    pMS=0.045,
    pSL=0.020,

    # attrition rates
    aJ=0.08,
    aM=0.06,
    aS=0.05,
    aL=0.04,

    # corrected baseline mentorship
    alpha=0.1,
    mentor_scale=3000,

    # bias parameters
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
    readiness = np.zeros(steps + 1)

    J[0], M[0], S[0], L[0] = J0, M0, S0, L0

    for t in range(steps):

        mentorship[t] = 1.0 - np.exp(-(S[t] + L[t]) / mentor_scale)
        mentorship[t] = np.clip(mentorship[t], 0.0, 1.0)

        readiness[t] = mentorship[t]
        f_mentorship = 1.0 + alpha * mentorship[t]

        f_bias_JM = 1.0 - bias_JM
        f_bias_MS = 1.0 - bias_MS
        f_bias_SL = 1.0 - bias_SL

        if use_stagnation:
            g_stagnation = 1.0 + gamma * (1.0 - readiness[t])
        else:
            g_stagnation = 1.0

        eff_pJM = pJM * f_mentorship * f_bias_JM
        eff_pMS = pMS * f_mentorship * f_bias_MS
        eff_pSL = pSL * f_mentorship * f_bias_SL

        if cap_effective_promo:
            eff_pJM = np.clip(eff_pJM, 0.0, 1.0)
            eff_pMS = np.clip(eff_pMS, 0.0, 1.0)
            eff_pSL = np.clip(eff_pSL, 0.0, 1.0)

        promote_JM = J[t] * eff_pJM
        promote_MS = M[t] * eff_pMS
        promote_SL = S[t] * eff_pSL

        exit_J = J[t] * aJ * g_stagnation
        exit_M = M[t] * aM * g_stagnation
        exit_S = S[t] * aS * g_stagnation
        exit_L = L[t] * aL * g_stagnation

        J[t + 1] = max(J[t] + dt * (entry_per_year - promote_JM - exit_J), 0.0)
        M[t + 1] = max(M[t] + dt * (promote_JM - promote_MS - exit_M), 0.0)
        S[t + 1] = max(S[t] + dt * (promote_MS - promote_SL - exit_S), 0.0)
        L[t + 1] = max(L[t] + dt * (promote_SL - exit_L), 0.0)

    mentorship[steps] = 1.0 - np.exp(-(S[steps] + L[steps]) / mentor_scale)
    mentorship[steps] = np.clip(mentorship[steps], 0.0, 1.0)

    df = pd.DataFrame({
        "Year": years,
        "Junior": J,
        "Mid-level": M,
        "Senior": S,
        "Leadership": L,
        "MentorshipAvailability": mentorship
    })

    df["Total"] = df["Junior"] + df["Mid-level"] + df["Senior"] + df["Leadership"]
    df["Senior_Leadership_share"] = (
        (df["Senior"] + df["Leadership"]) / df["Total"] * 100
    )

    return df

# Promotion bias sensitivity analysis, Here the same bias value is applied at all transition stages
bias_values = [0.00, 0.10, 0.20, 0.30]
results = {}

for bias in bias_values:
    results[bias] = simulate(
        bias_JM=bias,
        bias_MS=bias,
        bias_SL=bias
    )

# Plot Figure 5.7
fig, ax = plt.subplots(figsize=(9, 5.5))

for bias, df in results.items():
    ax.plot(
        df["Year"],
        df["Senior_Leadership_share"],
        label=f"{int(bias * 100)}% bias"
    )

ax.set_title("Senior + Leadership Share Over Time Under Different Promotion Bias Levels")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Senior + Leadership share (%)")
ax.set_xlim(0, 30)
ax.legend(title="Promotion bias", frameon=False)

plt.tight_layout()
plt.savefig("figure_5_7_bias_sensitivity.png", bbox_inches="tight")
plt.show()

# Print core year-30 values for checking
print("\nFigure 5.7 year-30 senior + leadership share values:\n")

for bias, df in results.items():
    final = df.iloc[-1]
    print(f"{int(bias * 100)}% bias: {final['Senior_Leadership_share']:.2f}%")
