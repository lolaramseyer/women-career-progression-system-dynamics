#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:38:29 2026

@author: lolaramseyer
figure 5.3 workforce distribution by level over time (Junior, Mid, Senior, Leadership)
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Global plotting style
# -----------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "figure.figsize": (9, 5.5),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# -----------------------------
# Time settings
# -----------------------------
T = 30
dt = 1.0
time = np.arange(0, T + dt, dt)

# -----------------------------
# Example baseline workforce stocks
# Replace these with your actual model outputs if needed
# -----------------------------
junior = 2000 + 40*time - 1.2*(time**2)
mid = 800 + 22*time - 0.4*(time**2)
senior = 250 + 10*time + 0.12*(time**2)
leadership = 80 + 3*time + 0.05*(time**2)

# -----------------------------
# Plot: Figure 5.3
# -----------------------------
fig, ax = plt.subplots()

ax.plot(time, junior, linewidth=2.5, label="Junior")
ax.plot(time, mid, linewidth=2.5, label="Mid-level")
ax.plot(time, senior, linewidth=2.5, label="Senior")
ax.plot(time, leadership, linewidth=2.5, label="Leadership")

ax.set_xlabel("Time (years)")
ax.set_ylabel("Number of women")
ax.set_title("Workforce distribution by level over time under baseline conditions", pad=12)

ax.set_xlim(0, T)
ax.grid(True, linestyle="--", alpha=0.35)

ax.legend(frameon=False)

plt.tight_layout()
plt.show()
