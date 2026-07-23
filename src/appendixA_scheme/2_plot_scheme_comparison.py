# Finite-difference scheme dependence of extracted hydrodynamic coefficients
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os as _os
def _data(name):
    return _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'data', name)

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 6.5, "savefig.dpi": 600, "figure.constrained_layout.use": True,
    "lines.linewidth": 1.1, "lines.markersize": 3.6,
})

df = pd.read_csv(_data("scheme_comparison_results.csv"))
schemes = [
    ("forward",  "tab:red",   "o", "-",  "forward diff."),
    ("backward", "tab:blue",  "s", "-",  "backward diff."),
    ("central",  "tab:green", "^", "-",  "central diff."),
]

fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.45))

for col, ax, ylab in [("gamma", axes[0], r"Friction $\gamma$"),
                      ("nu",    axes[1], r"Viscosity $\nu$"),
                      ("c2",    axes[2], r"Elasticity $c^2$")]:
    for s, c, m, ls, lbl in schemes:
        sub = df[df["scheme"] == s].sort_values("delta_t_cg")
        ax.plot(sub["delta_t_cg"], sub[col], marker=m, ls=ls, color=c, label=lbl)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel(r"$\Delta t_{\rm cg}$")
    ax.set_ylabel(ylab)

# exact-derivative reference (from 11_dict_scan, D4 full dict, same window)
axes[0].axhline(0.1409, color="tab:purple", lw=1.2, ls="-.",
                label="exact derivative")
axes[1].axhline(-0.0085, color="tab:purple", lw=1.2, ls="-.")
axes[2].axhline(4.1924, color="tab:purple", lw=1.2, ls="-.")

h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=4, framealpha=0.95,
           columnspacing=1.0, handlelength=1.9, handletextpad=0.4)
axes[2].set_ylim(2.8, 4.5)
for ax, tag in zip(axes, "abc"):
    ax.text(0.03, 0.05, f"({tag})", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold")

fig.savefig("FigS_Scheme_Comparison.png", dpi=300)
fig.savefig("FigS_Scheme_Comparison.pdf")
print("saved FigS_Scheme_Comparison.{png,pdf}")
