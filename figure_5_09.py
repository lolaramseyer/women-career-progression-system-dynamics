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

# Core simulation model same corrected model as previous figures
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

    return df

# Function to get year 30 values
def get_year30_values(alpha, bias):
    df = simulate(
        alpha=alpha,
        bias_JM=bias,
        bias_MS=bias,
        bias_SL=bias
    )

    final = df.iloc[-1]

    junior = final["Junior"]
    mid = final["Mid-level"]
    senior = final["Senior"]
    leadership = final["Leadership"]
    total = junior + mid + senior + leadership

    leadership_share = leadership / total * 100
    senior_leadership_share = (senior + leadership) / total * 100

    return {
        "Junior": junior,
        "Mid-level": mid,
        "Senior": senior,
        "Leadership": leadership,
        "Total": total,
        "Leadership share (%)": leadership_share,
        "Senior + Leadership share (%)": senior_leadership_share
    }

# Heatmap settings
alpha_values = [0.1, 0.3, 0.6, 0.9]
bias_values = [0.0, 0.1, 0.2, 0.3]

alpha_labels = ["0.1", "0.3", "0.6", "0.9"]
bias_labels = ["0%", "10%", "20%", "30%"]

heatmap_data = np.zeros((len(bias_values), len(alpha_values)))

for i, bias in enumerate(bias_values):
    for j, alpha in enumerate(alpha_values):
        values = get_year30_values(alpha, bias)
        heatmap_data[i, j] = values["Leadership share (%)"]

# Plot Figure 5.9
fig, ax = plt.subplots(figsize=(8.5, 6))

im = ax.imshow(heatmap_data, cmap="Blues", aspect="auto")

ax.set_xticks(np.arange(len(alpha_labels)))
ax.set_yticks(np.arange(len(bias_labels)))

ax.set_xticklabels(alpha_labels)
ax.set_yticklabels(bias_labels)

ax.set_xlabel("Mentorship level (α)")
ax.set_ylabel("Promotion bias level")
ax.set_title(
    "Interaction Effects of Mentorship and Promotion Bias\n"
    "on Leadership Representation at Year 30"
)

for i in range(len(bias_values)):
    for j in range(len(alpha_values)):
        ax.text(
            j,
            i,
            f"{heatmap_data[i, j]:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=11
        )

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Leadership share at year 30 (%)")

plt.tight_layout()
plt.savefig("figure_5_9_interaction_heatmap.png", bbox_inches="tight")
plt.show()

# Print heatmap values
heatmap_table = pd.DataFrame(
    heatmap_data,
    index=bias_labels,
    columns=[f"α = {a}" for a in alpha_labels]
)

print("\nLeadership share at year 30 (%) for Figure 5.9:\n")
print(heatmap_table.round(2).to_string())

# Raw outputs for manual Table 5.4 calculations
selected_scenarios = [
    ("Low (α = 0.1)", 0.1, 0.0),
    ("Low (α = 0.1)", 0.1, 0.1),
    ("Low (α = 0.1)", 0.1, 0.3),

    ("Moderate (α = 0.3)", 0.3, 0.0),
    ("Moderate (α = 0.3)", 0.3, 0.1),
    ("Moderate (α = 0.3)", 0.3, 0.3),

    ("Very high (α = 0.9)", 0.9, 0.0),
    ("Very high (α = 0.9)", 0.9, 0.1),
    ("Very high (α = 0.9)", 0.9, 0.3),
]

raw_rows = []

for label, alpha, bias in selected_scenarios:
    values = get_year30_values(alpha, bias)

    raw_rows.append({
        "Mentorship level": label,
        "Bias level": f"{int(bias * 100)}%",
        "Junior": round(values["Junior"], 2),
        "Mid-level": round(values["Mid-level"], 2),
        "Senior": round(values["Senior"], 2),
        "Leadership": round(values["Leadership"], 2),
        "Total": round(values["Total"], 2)
    })

raw_outputs = pd.DataFrame(raw_rows)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

print("\nRaw year 30 outputs for manual Table 5.4 calculations:\n")
print(raw_outputs.to_string(index=False))

raw_outputs.to_csv("table_5_4_raw_outputs.csv", index=False)v", index=False)
