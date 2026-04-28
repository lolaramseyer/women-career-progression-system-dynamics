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
    J0=2000,
    M0=800,
    S0=250,
    L0=80,
    entry_per_year=300,
    pJM=0.12,
    pMS=0.10,
    pSL=0.08,
    aJ=0.08,
    aM=0.06,
    aS=0.05,
    aL=0.04,
    alpha=0.3,
    bias=0.15,
    kappa=0.002
):
    """
    Simulates women's movement through four career stocks:
    Junior, Mid-level, Senior and Leadership.
    """

    n_steps = int(T / dt) + 1
    time = np.arange(0, T + dt, dt)

    J = np.zeros(n_steps)
    M = np.zeros(n_steps)
    S = np.zeros(n_steps)
    L = np.zeros(n_steps)
    mentorship = np.zeros(n_steps)

    J[0] = J0
    M[0] = M0
    S[0] = S0
    L[0] = L0

    for t in range(1, n_steps):

        # Mentorship availability depends on senior + leadership representation
        mentorship[t - 1] = alpha * (1 - np.exp(-kappa * (S[t - 1] + L[t - 1])))

        # Effective promotion rates after mentorship and promotion bias
        eff_pJM = pJM * (1 - bias) * (1 + mentorship[t - 1])
        eff_pMS = pMS * (1 - bias) * (1 + mentorship[t - 1])
        eff_pSL = pSL * (1 - bias) * (1 + mentorship[t - 1])

        # Cap promotion rates at 1
        eff_pJM = min(eff_pJM, 1.0)
        eff_pMS = min(eff_pMS, 1.0)
        eff_pSL = min(eff_pSL, 1.0)

        # Promotion flows
        prom_JM = eff_pJM * J[t - 1] * dt
        prom_MS = eff_pMS * M[t - 1] * dt
        prom_SL = eff_pSL * S[t - 1] * dt

        # Attrition flows
        attr_J = aJ * J[t - 1] * dt
        attr_M = aM * M[t - 1] * dt
        attr_S = aS * S[t - 1] * dt
        attr_L = aL * L[t - 1] * dt

        # Entry flow
        entry = entry_per_year * dt

        # Stock updates
        J[t] = J[t - 1] + entry - prom_JM - attr_J
        M[t] = M[t - 1] + prom_JM - prom_MS - attr_M
        S[t] = S[t - 1] + prom_MS - prom_SL - attr_S
        L[t] = L[t - 1] + prom_SL - attr_L

        # Prevent negative stock values
        J[t] = max(J[t], 0)
        M[t] = max(M[t], 0)
        S[t] = max(S[t], 0)
        L[t] = max(L[t], 0)

    # Final mentorship value
    mentorship[-1] = alpha * (1 - np.exp(-kappa * (S[-1] + L[-1])))

    results = pd.DataFrame({
        "Time": time,
        "Junior": J,
        "Mid": M,
        "Senior": S,
        "Leadership": L,
        "Mentorship": mentorship
    })

    return results

# Function to get year 30 summary values
def get_year30_values(alpha, bias):
    """
    Runs one scenario and returns raw stocks and calculated shares at year 30.
    """

    df = simulate(alpha=alpha, bias=bias)
    final = df.iloc[-1]

    junior = final["Junior"]
    mid = final["Mid"]
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

# Settings for Figure 5.9 heatmap
alpha_values = [0.1, 0.3, 0.6, 0.9]
bias_values = [0.0, 0.1, 0.2, 0.3]

alpha_labels = ["0.1", "0.3", "0.6", "0.9"]
bias_labels = ["0%", "10%", "20%", "30%"]

# Build heatmap values
heatmap_data = np.zeros((len(bias_values), len(alpha_values)))

for i, bias in enumerate(bias_values):
    for j, alpha in enumerate(alpha_values):
        values = get_year30_values(alpha, bias)
        heatmap_data[i, j] = values["Leadership share (%)"]

# Plot Figure 5.9 heatmap
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

# Add numbers inside heatmap cells
for i in range(len(bias_values)):
    for j in range(len(alpha_values)):
        ax.text(
            j,
            i,
            f"{heatmap_data[i, j]:.1f}",
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

# Raw outputs for Table 5.4
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

raw_outputs.to_csv("table_5_4_raw_outputs.csv", index=False)
