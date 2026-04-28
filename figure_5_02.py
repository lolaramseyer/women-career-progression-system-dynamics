
"""
Created on Tue Mar 10 11:18:15 2026

@author: lolaramseyer
5,2 figure: mentorship availiability over time (baseline)
"""

import numpy as np
import matplotlib.pyplot as plt

# baseline simulation
T = 30                  # years
dt = 1.0
time = np.arange(0, T + dt, dt)

# Example women stocks at Senior and Leadership level
# Replace these with your actual model outputs if you already have them
senior = 250 + 8*time + 0.15*(time**2)
leadership = 80 + 2*time + 0.05*(time**2)


# Mentorship function alpha = mentorship scaling parameter
alpha = 0.3

# total senior representation feeding mentorship capacity
senior_pool = senior + leadership

# mentorship availability (saturating function)
mentorship = 1 - np.exp(-alpha * senior_pool / 1000)

# Plot: Figure 5.2
plt.figure(figsize=(9, 5.5))
plt.plot(time, mentorship, linewidth=2.5, label="Mentorship availability")

plt.xlabel("Time (years)")
plt.ylabel("Mentorship availability")
plt.title("Mentorship availability over time (baseline)")
plt.xlim(0, T)
plt.ylim(0, max(mentorship) * 1.1)
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
