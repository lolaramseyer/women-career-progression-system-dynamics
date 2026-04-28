
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

# Core simulation model Same as Figure 5.1 
def simulate(
    T=30,
    dt=1.0,
    J0=2000,
    M0=800,
    S0=250,
    L0=80,
    entry_per_year=300,

    # New baseline progression rates
    pJM=0.10,
    pMS=0.045,
    pSL=0.020,

    # Attrition rates
    aJ=0.08,
    aM=0.06,
    aS=0.05,
    aL=0.04,

    # Mentorship settings
    alpha=0.1,
    mentor_scale=3000,

    # New baseline bias assumptions
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
    readiness[steps] = mentorship[steps]

    df = pd.DataFrame({
        "Year": years,
        "Junior": J,
        "Mid-level": M,
        "Senior": S,
        "Leadership": L,
        "MentorshipAvailability": mentorship,
        "PromotionReadiness": readiness
    })

    return df


# Figure 5.5, Leadership stock at year 30 by mentorship level
alpha_values = [0.1, 0.3, 0.6, 0.9]
labels = ["Low", "Moderate", "High", "Very high"]

leadership_year30 = []

for alpha in alpha_values:
    df = simulate(alpha=alpha)
    leadership_year30.append(df["Leadership"].iloc[-1])

# Plot bar chart
fig, ax = plt.subplots(figsize=(8, 5.5))
bars = ax.bar(labels, leadership_year30)

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
plt.savefig("figure_5_5_leadership_by_mentorship.png", bbox_inches="tight")
plt.show()

print("\nFigure 5.5 year-30 leadership stock values:\n")

for label, alpha, leadership in zip(labels, alpha_values, leadership_year30):
    print(f"{label} mentorship, alpha = {alpha}: Leadership stock = {leadership:.2f}")
