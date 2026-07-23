# Fig.11 candidate: dictionary-size convergence & forward-prediction capability
# (referee comment #1). Data: dict_scan_results.csv / dict_scan_prediction.npz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os as _os
def _data(name):
    return _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'data', name)

# Final physical size: figure* width = 7.05 in (revtex4-2 \textwidth),
# so the fonts below are the sizes actually seen in print (referee comment 2).
plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 6.5,
    "lines.linewidth": 1.1, "lines.markersize": 4,
    "axes.linewidth": 0.7, "savefig.dpi": 600,
    "figure.constrained_layout.use": False,
})

df = pd.read_csv(_data("dict_scan_results.csv"))
npz = np.load(_data("dict_scan_prediction.npz"))

fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.9))
fig.subplots_adjust(left=0.09, right=0.975, bottom=0.17, top=0.78, wspace=0.72)

# ---------------- (a) coefficients vs dictionary size ----------------
ax = axes[0]
sub_cg = df[(df["mode"] == "cg") & (df["n_obs"] >= 39)].sort_values("n_obs")
sub_ex = df[(df["mode"] == "exact") & (df["n_obs"] >= 39)].sort_values("n_obs")
n = sub_cg["n_obs"].values

h1, = ax.plot(n, sub_cg["gamma"], "o-", color="tab:blue",
              label=r"$\gamma$ (cg)")
h2, = ax.plot(n, sub_cg["nu"], "s-", color="tab:green",
              label=r"$\nu$ (cg)")
h3, = ax.plot(n, sub_ex["gamma"], "o--", mfc="none", color="tab:blue",
              label=r"$\gamma$ (exact)")
h4, = ax.plot(n, sub_ex["nu"], "s--", mfc="none", color="tab:green",
              label=r"$\nu$ (exact)")
ax.axhline(0.0, color="gray", lw=0.8, ls=":")
ax.set_xlabel("Dictionary size $N_{\\rm dict}$")
ax.set_ylabel(r"$\gamma,\ \nu$")
ax.set_xticks(n)
ax.set_ylim(-0.45, 0.95)

ax2 = ax.twinx()
h5, = ax2.plot(n, sub_cg["c2"], "^-", color="tab:red", label=r"$c^2$ (cg)")
h6, = ax2.plot(n, sub_ex["c2"], "^--", mfc="none", color="tab:red",
               label=r"$c^2$ (exact)")
ax2.set_ylabel(r"$c^2$", color="tab:red", labelpad=1)
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.set_ylim(3.0, 4.6)

ax.legend(handles=[h1, h2, h3, h4, h5, h6], loc="lower left",
          bbox_to_anchor=(-0.02, 1.02), ncol=3, framealpha=1.0,
          handlelength=1.6, columnspacing=0.8, handletextpad=0.4,
          borderaxespad=0.0)
ax.text(0.03, 0.94, "(a)", transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="top")

# ---------------- (b) prediction error vs time ----------------
ax = axes[1]
t = npz["D4_full149_cg__t"]
curves = [
    ("D1_Z20_cg__err",     r"$\{Z\}$ (20, cg)",          "tab:cyan",  ":"),
    ("D2_ZJ39_cg__err",    r"$\{Z,J\}$ (39, cg)",        "tab:blue",  "-"),
    ("D4_full149_cg__err", r"full hydro. (149, cg)",            "tab:red",   "-"),
    ("D4_full149_exact__err", r"full hydro. (149, exact deriv.)", "tab:purple",    "--"),
]
for key, lbl, c, ls in curves:
    ax.plot(t, npz[key], ls, color=c, label=lbl)
ax.set_xlabel("Time $t$")
ax.set_ylabel("Median rel. error of $\\langle Z_i(t)\\rangle$")
ax.set_xlim(1.0, 3.0)
ax.set_ylim(0, 0.36)
ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.02), ncol=2,
          framealpha=1.0, handlelength=1.6, columnspacing=0.8,
          handletextpad=0.4, borderaxespad=0.0)
ax.text(0.03, 0.94, "(b)", transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="top")

# ---------------- (c) single-sample trajectory ----------------
ax = axes[2]
ex = npz["D4_full149_cg__traj_exact"] * 1e3
p_cg = npz["D4_full149_cg__traj_pred"] * 1e3
p_ed = npz["D4_full149_exact__traj_pred"] * 1e3
m = t <= 2.5
ax.plot(t[m], ex[m], "-", color="0.45", lw=1.4, label="exact")
ax.plot(t[m], p_cg[m], "-", color="tab:red", lw=1.4,
        label=r"full hydro. (149, cg)")
ax.plot(t[m], p_ed[m], "--", color="tab:purple", lw=1.1,
        label=r"full hydro. (149, exact deriv.)")
ax.axhline(0.0, color="gray", lw=0.8, ls=":")
ax.set_xlabel("Time $t$")
ax.set_ylabel(r"$\langle Z_{10}(t)\rangle \times 10^{3}$")
ax.set_xlim(1.0, 2.5)
ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.02), ncol=1,
          framealpha=1.0, handlelength=1.5, columnspacing=0.7,
          handletextpad=0.35, borderaxespad=0.0)
ax.text(0.965, 0.94, "(c)", transform=ax.transAxes, fontsize=9.5,
        fontweight="bold", va="top", ha="right")

fig.savefig("Fig11_Dictionary_Prediction.png", dpi=300)
fig.savefig("Fig11_Dictionary_Prediction.pdf")
print("saved Fig11_Dictionary_Prediction.{png,pdf}")
