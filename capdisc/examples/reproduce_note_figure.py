"""Reproduces (and extends) the "Tetrahedra family vs. 3-point and 2-point
discrepancies" figure from the original 2016 talk, using sphcap's exact
enumerative discrepancy computation instead of a hand symbolic solve.
"""
import numpy as np
import matplotlib.pyplot as plt

from sphcap import configs, discrepancy

ts = np.linspace(0.02, 0.98, 400)
vals = np.array([discrepancy(configs.disphenoid_family(t), method="exact").value for t in ts])

t_star = np.sqrt((13 - np.sqrt(105)) / 32)
t_reg = 1 / np.sqrt(3)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(ts, vals, color="#1f77b4", lw=2, label=r"$D(X(t))$ (exact)")
ax.axhline(3 / 8, color="#999999", ls="--", lw=1, label=r"$3/8$")
ax.axvline(t_star, color="#d62728", ls=":", lw=1.5,
           label=rf"$t^\ast\approx{t_star:.4f}$ (optimum)")
ax.axvline(t_reg, color="#2ca02c", ls=":", lw=1.5,
           label=rf"$t_{{\rm reg}}=1/\sqrt{{3}}\approx{t_reg:.4f}$ (regular tet.)")
ax.plot([t_star], [3 / 8], "o", color="#d62728", zorder=5)
ax.set_xlabel(r"disphenoid parameter $t$")
ax.set_ylabel(r"spherical cap discrepancy $D$")
ax.set_title("Disphenoid family: exact discrepancy vs. squish parameter")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, 0.55)
fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/disphenoid_family_discrepancy.png", dpi=160)
print("min at t=", ts[np.argmin(vals)], "value=", vals.min(), "target 3/8=", 3 / 8)
