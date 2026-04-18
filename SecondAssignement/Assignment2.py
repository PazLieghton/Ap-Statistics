# -*- coding: utf-8 -*-
"""
Im going to be honest with myself for once and actually invest time for the
@author: pazli
"""

import numpy as np

x = np.array([2, 4, 6, 5, 7, 3, 8, 5, 6, 4, 7, 3])
y = np.array([8, 15, 22, 18, 27, 10, 30, 17, 21, 13, 25, 11])
z = np.array([8, 7, 5, 6, 4, 7, 4, 6, 5, 7, 4, 8])

data = np.array([x, y, z])
n = len(x)

mean_x = np.mean(x)
mean_y = np.mean(y)
mean_z = np.mean(z)

#Its gotta be 3.16 11.91 -2.5
#Its gotta be 11.91 45.5 -9.4
#Its gotta be -2.5 -9.4 2.07
# The upper right triangle

E_xy = np.mean(x * y)
E_xz = np.mean(x * z)
E_yz = np.mean(y * z)
E_x2 = np.mean(x * x)           # E[X^2]
E_y2 = np.mean(y * y)
E_z2 = np.mean(z * z)

# Formula: Cov(X,Y) = E[XY] - E[X]E[Y]
cov_pop_xx = E_x2 - mean_x * mean_x        
cov_pop_yy = E_y2 - mean_y * mean_y
cov_pop_zz = E_z2 - mean_z * mean_z
cov_pop_xy = E_xy - mean_x * mean_y
cov_pop_xz = E_xz - mean_x * mean_z
cov_pop_yz = E_yz - mean_y * mean_z

#gotta use the division fucking factor
scale = n / (n - 1)
cov_sample_xx = cov_pop_xx * scale
cov_sample_yy = cov_pop_yy * scale
cov_sample_zz = cov_pop_zz * scale
cov_sample_xy = cov_pop_xy * scale
cov_sample_xz = cov_pop_xz * scale
cov_sample_yz = cov_pop_yz * scale

cov_manual = np.array([
    [cov_sample_xx, cov_sample_xy, cov_sample_xz],
    [cov_sample_xy, cov_sample_yy, cov_sample_yz],
    [cov_sample_xz, cov_sample_yz, cov_sample_zz]
])


std_x = np.std(x, ddof=1)      # sample std (ddof=1) to match np.corrcoef
std_y = np.std(y, ddof=1)
std_z = np.std(z, ddof=1)

corr_manual = np.array([
    [1.0,                     cov_sample_xy/(std_x*std_y), cov_sample_xz/(std_x*std_z)],
    [cov_sample_xy/(std_x*std_y), 1.0,                     cov_sample_yz/(std_y*std_z)],
    [cov_sample_xz/(std_x*std_z), cov_sample_yz/(std_y*std_z), 1.0]
])
# ---- Compare with NumPy built‑ins ----
cov_np = np.cov(data, bias=False)     # sample covariance
corr_np = np.corrcoef(data)

print("Manual covariance matrix:\n", cov_manual)
print("\nNumPy covariance matrix:\n", cov_np)
print("\nDifference (should be ~0):\n", cov_manual - cov_np)

print("\nManual correlation matrix:\n", corr_manual)
print("\nNumPy correlation matrix:\n", corr_np)
print("\nDifference:\n", corr_manual - corr_np)


#RE FUCKING DO EVERYTHING AND BE CAREFUL WHEN YOU GET A BSOD, SAVE THINGS SO YOU DONT HAVE TO RELY ON DEEPSEEK YOU ASSHOLE
# ... (your existing code above) ...

# ------------------- 3D Visualization -------------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot: color by sleep hours (z)
sc = ax.scatter(x, y, z, c=z, cmap='coolwarm', s=80, alpha=0.9, edgecolors='k')

# Axis labels
ax.set_xlabel('Hours Studied', fontsize=12, labelpad=10)
ax.set_ylabel('Problems Solved', fontsize=12, labelpad=10)
ax.set_zlabel('Hours Slept', fontsize=12, labelpad=10)
ax.set_title('Student Study Data: Hours, Problems, Sleep', fontsize=14, pad=20)

# Colorbar to decode sleep hours
cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Hours Slept', fontsize=11)

# Optional: Fit a plane (y ~ x + z) and plot it
# Plane equation: y = a*x + b*z + c
A = np.vstack([x, z, np.ones_like(x)]).T
coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
a, b, c = coeff

# Create meshgrid for the plane
xx, zz = np.meshgrid(np.linspace(min(x), max(x), 10),
                     np.linspace(min(z), max(z), 10))
yy = a * xx + b * zz + c

ax.plot_surface(xx, yy, zz, alpha=0.3, color='green', edgecolor='none')

# Adjust view angle for better perspective
ax.view_init(elev=20, azim=120)

plt.tight_layout()
plt.show()