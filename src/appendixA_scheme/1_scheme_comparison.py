# ----------------------------------------------------------------------
# Part of the reproduction code for:
#   Seiki Saito, "Temporal coarse-graining as the origin of macroscopic
#   friction in quantum spin chains via data-driven Liouvillian extraction",
#   Phys. Rev. Research (2026). DOI: 10.1103/41m6-x2m9
# If you use this code, or a modified version of it, in your work,
# please cite the paper above. (MIT License; see LICENSE.)
# ----------------------------------------------------------------------
# ======================================================================
# 12_scheme_comparison.py
# 有限差分スキーム依存性の検証: 前方/後方/中央差分 + (1/Δ)log K
# 同一データ・同一学習窓 (t in [1.0, 2.0], 500 samples) で
# 流体係数 (c^2, gamma, nu) の Δt_cg 依存性をスキーム別に抽出する。
# X_win が全スキーム・全 cg で共通なので pinv(X_win) は 1 回だけ計算。
# ======================================================================
import glob
import time
import numpy as np
import scipy.linalg as la
import pandas as pd

print("=== [Scheme check] forward vs backward vs central vs logK ===")

dt = 0.002
N_Z, N_J = 20, 19
N1 = N_Z + N_J
train_start, train_end = 500, 1000
rank_tol = 1e-5

cg_list = [1, 3, 5, 11, 21, 31, 41, 51, 71, 101]

file_list = sorted(glob.glob("gedmd_current_len3_sample_*.npz"))
print(f"サンプル数: {len(file_list)}")
t0 = time.time()
all_X = [np.load(f)["X_data"] for f in file_list]
print(f"読み込み完了 {time.time()-t0:.1f}s")

X_win = np.hstack([X[:, train_start:train_end] for X in all_X])
print("X_win:", X_win.shape)

t0 = time.time()
U, S, Vh = la.svd(X_win, full_matrices=False, lapack_driver="gesvd")
r = int(np.sum(S > rank_tol))
Pinv = Vh[:r].conj().T @ np.diag(1.0 / S[:r]) @ U[:, :r].conj().T  # (cols, 149)
print(f"SVD done (rank {r}/{len(S)}) {time.time()-t0:.1f}s")

def shifted(shift):
    return np.hstack([X[:, train_start + shift:train_end + shift] for X in all_X])

def coeffs(L):
    """Schur 補元で bath を畳み込み、流体係数を抽出 (script 10 と同一)"""
    try:
        L_open = np.real(
            L[:N1, :N1]
            - L[:N1, N1:] @ np.linalg.pinv(L[N1:, N1:], rcond=1e-3) @ L[N1:, :N1]
        )
    except np.linalg.LinAlgError:
        return (np.nan,) * 3
    c2 = np.zeros(N_J); gam = np.zeros(N_J); nu = np.zeros(N_J)
    for k in range(1, N_J - 1):
        c2[k] = (L_open[N_Z + k, k] - L_open[N_Z + k, k + 1]) / 2.0
        gam[k] = -L_open[N_Z + k, N_Z + k]
        nu[k] = (L_open[N_Z + k, N_Z + k - 1] + L_open[N_Z + k, N_Z + k + 1]) / 2.0
    return (np.median(c2[2:-2]), np.median(gam[2:-2]), np.median(nu[2:-2]))

records = []
for cg in cg_list:
    d = cg * dt
    Xp = shifted(+cg)
    Xm = shifted(-cg)
    Ls = {
        "forward":  ((Xp - X_win) / d) @ Pinv,
        "backward": ((X_win - Xm) / d) @ Pinv,
        "central":  ((Xp - Xm) / (2 * d)) @ Pinv,
    }
    # (1/Δ) log K  (主枝; 固有値が負実軸に近いと不定 → 例外は NaN)
    K = Xp @ Pinv
    try:
        Llog = np.real(la.logm(K)) / d
        Ls["logK"] = Llog
    except Exception as e:
        print(f"  logm failed at cg={cg}: {e}")
    for scheme, L in Ls.items():
        c2, gam, nu = coeffs(L)
        maxre = float(np.max(np.real(la.eigvals(L))))
        records.append(dict(cg=cg, delta_t_cg=d, scheme=scheme,
                            c2=c2, gamma=gam, nu=nu,
                            D=c2 / gam if abs(gam) > 1e-6 else np.nan,
                            max_Re_eig=maxre))
        print(f"cg={cg:4d} Δ={d:.3f} [{scheme:8s}] "
              f"c2={c2:+.4f} gamma={gam:+.4f} nu={nu:+.4f}")

df = pd.DataFrame(records)
df.to_csv("scheme_comparison_results.csv", index=False)
print("\n=> scheme_comparison_results.csv 保存完了")
