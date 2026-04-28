
"""
Created on Tue Mar 10 10:59:40 2026

@author: lolaramseyer
graph 5.1: Women in senior + Leadership Over Time (Baseline)
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

    # baseline progression rates
    pJM=0.10,
    pMS=0.045,
    pSL=0.020,

    # attrition rates
    aJ=0.08,
    aM=0.06,
    aS=0.05,
    aL=0.04,

    # mentorship assumptions
    alpha=0.1,
    mentor_scale=3000,

    # promotion bias assumptions
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

    J[0] = J0
    M[0] = M0
    S[0] = S0
    L[0] = L0

    for t in range(steps):

        # Mentorship availability depends on current senior + leadership stock
        mentorship[t] = 1.0 - np.exp(-(S[t] + L[t]) / mentor_scale)
        mentorship[t] = np.clip(mentorship[t], 0.0, 1.0)

        # Promotion readiness is assumed to follow mentorship availability
        readiness[t] = mentorship[t]

        # Mentorship effect on promotion
        f_mentorship = 1.0 + alpha * mentorship[t]

        # Bias reduces promotion probability
        f_bias_JM = 1.0 - bias_JM
        f_bias_MS = 1.0 - bias_MS
        f_bias_SL = 1.0 - bias_SL

        # Stagnation increases attrition when readiness is low
        if use_stagnation:
            g_stagnation = 1.0 + gamma * (1.0 - readiness[t])
        else:
            g_stagnation = 1.0

        # Effective promotion probabilities
        eff_pJM = pJM * f_mentorship * f_bias_JM
        eff_pMS = pMS * f_mentorship * f_bias_MS
        eff_pSL = pSL * f_mentorship * f_bias_SL

        if cap_effective_promo:
            eff_pJM = np.clip(eff_pJM, 0.0, 1.0)
            eff_pMS = np.clip(eff_pMS, 0.0, 1.0)
            eff_pSL = np.clip(eff_pSL, 0.0, 1.0)

        # Promotion flows
        promote_JM = J[t] * eff_pJM
        promote_MS = M[t] * eff_pMS
        promote_SL = S[t] * eff_pSL

        # Attrition flows
        exit_J = J[t] * aJ * g_stagnation
        exit_M = M[t] * aM * g_stagnation
        exit_S = S[t] * aS * g_stagnation
        exit_L = L[t] * aL * g_stagnation

        # Stock updates
        J_next = J[t] + dt * (entry_per_year - promote_JM - exit_J)
        M_next = M[t] + dt * (promote_JM - promote_MS - exit_M)
        S_next = S[t] + dt * (promote_MS - promote_SL - exit_S)
        L_next = L[t] + dt * (promote_SL - exit_L)

        J[t + 1] = max(J_next, 0.0)
        M[t + 1] = max(M_next, 0.0)
        S[t + 1] = max(S_next, 0.0)
        L[t + 1] = max(L_next, 0.0)

    # Final mentorship/readiness values
    mentorship[steps] = 1.0 - np.exp(-(S[steps] + L[steps]) / mentor_scale)
    mentorship[steps] = np.clip(mentorship[steps], 0.0, 1.0)
    readiness[steps] = mentorship[steps]

    df = pd.DataFrame({
        "Year": years,
        "Junior": J,
        "Mid": M,
        "Senior": S,
        "Leadership": L,
        "MentorshipAvailability": mentorship,
        "PromotionReadiness": readiness
    })

    df["Total"] = df["Junior"] + df["Mid"] + df["Senior"] + df["Leadership"]
    df["SeniorPlusLeadership"] = df["Senior"] + df["Leadership"]
    df["SplusL_Share"] = df["SeniorPlusLeadership"] / df["Total"]
    df["Leadership_Share"] = df["Leadership"] / df["Total"]

    return df

# Run baseline scenario
baseline = simulate()

# Figure 5.1 Senior + Leadership share over time under baseline
plt.figure()

plt.plot(
    baseline["Year"],
    baseline["SplusL_Share"] * 100,
    label="Baseline"
)

plt.xlabel("Year")
plt.ylabel("Senior + leadership share (%)")
plt.title("Senior and Leadership Representation Over Time Under Baseline Conditions")
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig("figure_5_1_baseline_senior_leadership_share.png", bbox_inches="tight")
plt.show()

# Print raw year 30 values for checking
final = baseline.iloc[-1]

print("\nFigure 5.1 baseline year 30 values:\n")
print(f"Junior stock:                 {final['Junior']:.2f}")
print(f"Mid-level stock:              {final['Mid']:.2f}")
print(f"Senior stock:                 {final['Senior']:.2f}")
print(f"Leadership stock:             {final['Leadership']:.2f}")
print(f"Total stock:                  {final['Total']:.2f}")
print(f"Senior + Leadership stock:    {final['SeniorPlusLeadership']:.2f}")
print(f"Senior + Leadership share:    {final['SplusL_Share'] * 100:.2f}%")
print(f"Leadership share:             {final['Leadership_Share'] * 100:.2f}%")
print(f"Mentorship availability:      {final['MentorshipAvailability']:.4f}")
