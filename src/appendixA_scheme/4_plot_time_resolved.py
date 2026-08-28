# ----------------------------------------------------------------------
# Part of the reproduction code for:
#   Seiki Saito, "Temporal coarse-graining as the origin of macroscopic
#   friction in quantum spin chains via data-driven Liouvillian extraction",
#   Phys. Rev. Research (2026). DOI: 10.1103/41m6-x2m9
# If you use this code, or a modified version of it, in your work,
# please cite the paper above. (MIT License; see LICENSE.)
# ----------------------------------------------------------------------
# Time-resolved gamma(t): exact / central vs forward difference
import pandas as pd
import matplotlib.pyplot as plt
import os as _os
def _data(name):
    return _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'data', name)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 5.8, "savefig.dpi": 600, "figure.constrained_layout.use": True,
    "lines.linewidth": 0.9,
})

df = pd.read_csv(_data("time_resolved_schemes.csv"))

fig, ax = plt.subplots(figsize=(3.42, 2.75))
styles = {
    "exact": ("tab:purple", "-",  1.1, "exact derivative"),
    "ctr20": ("tab:green",  "--", 0.9, r"central diff. ($\Delta t_{\rm cg}=0.04$)"),
    "fwd20": ("tab:red",    "-",  1.3, r"forward diff. ($\Delta t_{\rm cg}=0.04$)"),
    "ctr50": ("0.55",       ":",  0.9, r"central diff. ($\Delta t_{\rm cg}=0.20$)"),
}
for s in ("exact", "ctr20", "ctr50", "fwd20"):
    c, ls, lw, lbl = styles[s]
    sub = df[df["scheme"] == s].sort_values("t_center")
    ax.plot(sub["t_center"], sub["gamma"], ls, color=c, lw=lw, label=lbl)

ax.axhline(0, color="gray", lw=0.9, ls=":")
ax.axhspan(-0.0353 - 0.1846, -0.0353 + 0.1846, color="tab:purple", alpha=0.07)
ax.set_xlabel(r"Window center $t$")
ax.set_ylabel(r"Friction $\gamma(t)$  (bulk median)")
ax.set_xlim(df["t_center"].min(), df["t_center"].max())
ax.legend(loc="upper right", framealpha=0.95, ncol=2)
fig.savefig("FigS_Time_Resolved_Gamma.png", dpi=300)
fig.savefig("FigS_Time_Resolved_Gamma.pdf")
print("saved FigS_Time_Resolved_Gamma.{png,pdf}")
