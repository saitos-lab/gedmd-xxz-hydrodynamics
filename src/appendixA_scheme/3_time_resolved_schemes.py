# ----------------------------------------------------------------------
# Part of the reproduction code for:
#   Seiki Saito, "Temporal coarse-graining as the origin of macroscopic
#   friction in quantum spin chains via data-driven Liouvillian extraction",
#   Phys. Rev. Research (2026). DOI: 10.1103/41m6-x2m9
# If you use this code, or a modified version of it, in your work,
# please cite the paper above. (MIT License; see LICENSE.)
# ----------------------------------------------------------------------
# ======================================================================
# 13_time_resolved_schemes.py
# 時間分解の係数抽出: exact / central / forward をスライディング窓で比較
# 目的: Fig.8 の「exact γ(t) はゼロ周りで振動」の主張を検証
#       (ゼロ周りか、小さな正の値 ~0.14 周りか)
# 正規方程式 (G = X X^T) 使用で高速化。窓 τ=1.0, stride 0.02。
# ======================================================================
import glob
import time
import numpy as np
import pandas as pd

print("=== [Time-resolved] exact vs central vs forward ===")

dt = 0.002
N_Z, N_J = 20, 19
N1 = N_Z + N_J
win = 500            # τ = 1.0
stride = 10          # 0.02
CG_A = 20            # Δt_cg = 0.04
CG_B = 50            # Δt_cg = 0.20

file_list = sorted(glob.glob("gedmd_current_len3_sample_*.npz"))
print(f"サンプル数: {len(file_list)}")
t0 = time.time()
all_X, all_dX = [], []
for f in file_list:
    d = np.load(f)
    all_X.append(d["X_data"])
    all_dX.append(d["dX_data"])
print(f"読み込み完了 {time.time()-t0:.1f}s")

# 3次元配列化 (samples, 149, 2000) -> 窓抽出を軸スライスで高速に
Xall = np.stack(all_X)    # (500, 149, 2000)
dXall = np.stack(all_dX)
del all_X, all_dX

def win_mat(arr, s):
    """窓 [s, s+win) を全サンプル横結合した (149, 500*win) 行列"""
    return np.concatenate(arr[:, :, s:s + win], axis=1)

def coeffs(L):
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

starts = list(range(CG_B, 2000 - win - CG_B, stride))
print(f"フレーム数: {len(starts)}")

records = []
t0 = time.time()
for i, s in enumerate(starts):
    Xw = win_mat(Xall, s)
    G = Xw @ Xw.T
    G += np.eye(149) * (1e-10 * np.trace(G) / 149)
    Ginv = np.linalg.inv(G)

    Ys = {
        "exact": win_mat(dXall, s),
        f"fwd{CG_A}": (win_mat(Xall, s + CG_A) - Xw) / (CG_A * dt),
        f"ctr{CG_A}": (win_mat(Xall, s + CG_A) - win_mat(Xall, s - CG_A)) / (2 * CG_A * dt),
        f"ctr{CG_B}": (win_mat(Xall, s + CG_B) - win_mat(Xall, s - CG_B)) / (2 * CG_B * dt),
    }
    t_center = (s + win / 2) * dt
    for scheme, Y in Ys.items():
        L = (Y @ Xw.T) @ Ginv
        c2, gam, nu = coeffs(L)
        records.append(dict(t_center=t_center, scheme=scheme,
                            c2=c2, gamma=gam, nu=nu))
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(starts)} frames ({time.time()-t0:.0f}s)")

df = pd.DataFrame(records)
df.to_csv("time_resolved_schemes.csv", index=False)

# 安定窓 [1.0, 2.5] の統計
print("\n--- 時間統計 (t_center in [1.0, 2.5]) ---")
m = (df["t_center"] >= 1.0) & (df["t_center"] <= 2.5)
for scheme in df["scheme"].unique():
    sub = df[m & (df["scheme"] == scheme)]
    print(f"[{scheme:6s}] gamma: mean={sub['gamma'].mean():+.4f} "
          f"std={sub['gamma'].std():.4f} min={sub['gamma'].min():+.4f} "
          f"max={sub['gamma'].max():+.4f} | "
          f"c2 mean={sub['c2'].mean():.4f} | nu mean={sub['nu'].mean():+.4f}")
print("\n=> time_resolved_schemes.csv 保存完了")
