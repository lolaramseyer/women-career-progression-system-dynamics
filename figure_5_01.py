#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:59:40 2026

@author: lolaramseyer
graph 5.1: Women in senior + Leadership Over Time (Baseline)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Plot styling ---
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
    J0=2000, M0=800, S0=250, L0=80,
    entry_per_year=300,
    pJM=0.12, pMS=0.10, pSL=0.08,
    aJ=0.08, aM=0.06, aS=0.05, aL=0.04,
    alpha=0.6,
    mentor_scale=500.0,
    bias_JM=0.05,
    bias_MS=0.10,
    bias_SL=0.20,
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

    J[0], M[0], S[0], L[0] = float(J0), float(M0), float(S0), float(L0)

    for t in range(steps):
        mentorship[t] = 1.0 - np.exp(-(S[t] + L[t]) / float(mentor_scale))
        mentorship[t] = float(np.clip(mentorship[t], 0.0, 1.0))

        readiness[t] = float(np.clip(mentorship[t], 0.0, 1.0))

        f_mentorship = 1.0 + float(alpha) * mentorship[t]

        f_bias_JM = 1.0 - float(bias_JM)
        f_bias_MS = 1.0 - float(bias_MS)
        f_bias_SL = 1.0 - float(bias_SL)

        if use_stagnation:
            g_stagnation = 1.0 + float(gamma) * (1.0 - readiness[t])
        else:
            g_stagnation = 1.0

        eff_pJM = float(pJM) * f_mentorship * f_bias_JM
        eff_pMS = float(pMS) * f_mentorship * f_bias_MS
        eff_pSL = float(pSL) * f_mentorship * f_bias_SL

        if cap_effective_promo:
            eff_pJM = float(np.clip(eff_pJM, 0.0, 1.0))
            eff_pMS = float(np.clip(eff_pMS, 0.0, 1.0))
            eff_pSL = float(np.clip(eff_pSL, 0.0, 1.0))

        promote_JM = J[t] * eff_pJM
        promote_MS = M[t] * eff_pMS
        promote_SL = S[t] * eff_pSL

        exit_J = J[t] * float(aJ) * g_stagnation
        exit_M = M[t] * float(aM) * g_stagnation
        exit_S = S[t] * float(aS) * g_stagnation
        exit_L = L[t] * float(aL) * g_stagnation

        J_next = J[t] + dt * (float(entry_per_year) - promote_JM - exit_J)
        M_next = M[t] + dt * (promote_JM - promote_MS - exit_M)
        S_next = S[t] + dt * (promote_MS - promote_SL - exit_S)
        L_next = L[t] + dt * (promote_SL - exit_L)

        J[t + 1] = max(J_next, 0.0)
        M[t + 1] = max(M_next, 0.0)
        S[t + 1] = max(S_next, 0.0)
        L[t + 1] = max(L_next, 0.0)

    mentorship[steps] = 1.0 - np.exp(-(S[steps] + L[steps]) / float(mentor_scale))
    mentorship[steps] = float(np.clip(mentorship[steps], 0.0, 1.0))
    readiness[steps] = float(np.clip(mentorship[steps], 0.0, 1.0))

    df = pd.DataFrame({
        "year": years,
        "Junior": J,
        "Mid": M,
        "Senior": S,
        "Leadership": L,
        "MentorshipAvailability": mentorship,
        "PromotionReadiness": readiness
    })

    df["Total"] = df["Junior"] + df["Mid"] + df["Senior"] + df["Leadership"]
    df["SeniorPlusLeadership"] = df["Senior"] + df["Leadership"]
    df["SplusL_Share"] = np.where(
        df["Total"] > 0,
        df["SeniorPlusLeadership"] / df["Total"],
        0.0
    )

    return df

# -----------------------------
# Run baseline
# -----------------------------
baseline = simulate(
    T=30,
    dt=1,
    J0=2000, M0=800, S0=250, L0=80,
    entry_per_year=300,
    pJM=0.12, pMS=0.10, pSL=0.08,
    aJ=0.08, aM=0.06, aS=0.05, aL=0.04,
    alpha=0.6,
    mentor_scale=500,
    bias_JM=0.05,
    bias_MS=0.10,
    bias_SL=0.20,
    use_stagnation=True,
    gamma=0.4,
    cap_effective_promo=True
)

# -----------------------------
# Figure 5.1
# Women in Senior + Leadership over time (baseline)
# -----------------------------
plt.figure()
plt.plot(baseline["year"], baseline["SplusL_Share"], label="Baseline")
plt.xlabel("Year")
plt.ylabel("Share of women in senior + leadership roles")
plt.title("Women in Senior and Leadership Roles Over Time (Baseline Scenario)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("figure_5_1_baseline_senior_leadership_share.png", bbox_inches="tight")
plt.show()
