# ----------------------------------------------------------------------
# Part of the reproduction code for:
#   Seiki Saito, "Temporal coarse-graining as the origin of macroscopic
#   friction in quantum spin chains via data-driven Liouvillian extraction",
#   Phys. Rev. Research (2026). DOI: 10.1103/41m6-x2m9
# If you use this code, or a modified version of it, in your work,
# please cite the paper above. (MIT License; see LICENSE.)
# ----------------------------------------------------------------------
# ======================================================================
# 11_dict_scan_prediction.py
# 査読コメント#1対応: 辞書サイズ依存性 + 前方予測能力の評価
#  (a) 部分辞書 {Z}, {Z,J}, {Z,J,ZZ,K}, {full} で L を再抽出し
#      流体係数 (c^2, gamma, nu) の辞書サイズ収束を評価
#  (b) 各辞書の L による前方予測 e^{Lt}X(t0) と厳密値の誤差を評価
#      (exact-derivative L vs coarse-grained L, Δt_cg=0.04)
#  既存の gedmd_current_len3_sample_*.npz のみ使用（新規時間発展なし）
# ======================================================================
import glob
import time
import numpy as np
import scipy.linalg as la
import pandas as pd

print("=== [Referee #1] Dictionary-size scan & forward prediction ===")

dt = 0.002
N_Z, N_J = 20, 19

# 部分辞書の行インデックス (1_gen_data_current.py の順序に基づく)
DICTS = {
    "D1_Z20":    np.arange(0, 20),     # Z のみ
    "D2_ZJ39":   np.arange(0, 39),     # Z + J
    "D3_len2_77": np.arange(0, 77),    # Z + J + ZZ + K
    "D4_full149": np.arange(0, 149),   # 全て (len-3 含む)
}

# 学習窓: t in [1.0, 2.0] (安定流体窓), 粗視化 cg=20 (Δt_cg=0.04)
train_start, train_end = 500, 1000
CG = 20
# 予測区間: t0=1.0 から t=3.0 まで, 5ステップ(0.01)刻み
pred_t0_idx = 500
pred_end_idx = 1500
pred_stride = 5
dt_pred = pred_stride * dt

# ----------------------------------------------------------------------
file_list = sorted(glob.glob("gedmd_current_len3_sample_*.npz"))
print(f"サンプル数: {len(file_list)}")
t0 = time.time()
all_X = []
all_dX = []
for f in file_list:
    d = np.load(f)
    all_X.append(d["X_data"])
    all_dX.append(d["dX_data"])
print(f"読み込み完了 {time.time()-t0:.1f}s")

rank_tol = 1e-5

def fit_L(rows, mode):
    """学習窓のアンサンブル連結データから L を回帰"""
    Xw = np.hstack([X[rows][:, train_start:train_end] for X in all_X])
    if mode == "exact":
        Y = np.hstack([dX[rows][:, train_start:train_end] for dX in all_dX])
    else:  # coarse-grained forward difference
        Xn = np.hstack([X[rows][:, train_start + CG:train_end + CG] for X in all_X])
        Y = (Xn - Xw) / (CG * dt)
    U, S, Vh = la.svd(Xw, full_matrices=False, lapack_driver="gesvd")
    r = int(np.sum(S > rank_tol))
    L = Y @ Vh[:r].conj().T @ np.diag(1.0 / S[:r]) @ U[:, :r].conj().T
    return L

def extract_coeffs(L, n_rows):
    """L から流体係数を抽出 (J 行が無い D1 は不可)"""
    if n_rows < 39:
        return None
    N1 = N_Z + N_J
    if n_rows > N1:  # bath を Schur 補元で畳み込み
        try:
            L_open = np.real(
                L[:N1, :N1]
                - L[:N1, N1:] @ np.linalg.pinv(L[N1:, N1:], rcond=1e-3) @ L[N1:, :N1]
            )
        except np.linalg.LinAlgError:
            return None
    else:
        L_open = np.real(L)
    c2 = np.zeros(N_J); gam = np.zeros(N_J); nu = np.zeros(N_J)
    for k in range(1, N_J - 1):
        c2[k] = (L_open[N_Z + k, k] - L_open[N_Z + k, k + 1]) / 2.0
        gam[k] = -L_open[N_Z + k, N_Z + k]
        nu[k] = (L_open[N_Z + k, N_Z + k - 1] + L_open[N_Z + k, N_Z + k + 1]) / 2.0
    return (np.median(c2[2:-2]), np.median(gam[2:-2]), np.median(nu[2:-2]))

def prediction_error(L, rows):
    """e^{L t} X(t0) による Z ブロック予測誤差 (サンプル毎→中央値)"""
    P = la.expm(L * dt_pred)
    idxs = np.arange(pred_t0_idx, pred_end_idx + 1, pred_stride)
    n_t = len(idxs)
    zsl = slice(0, N_Z)  # rows の先頭 20 要素は常に Z
    errs = np.zeros((len(all_X), n_t))
    traj_ex, traj_pr = None, None
    for s, X in enumerate(all_X):
        x = X[rows][:, pred_t0_idx].copy()
        Zex_all = X[rows][zsl][:, idxs]
        norm = la.norm(Zex_all) / np.sqrt(n_t)  # 典型振幅
        for n, ti in enumerate(idxs):
            errs[s, n] = la.norm(x[zsl] - X[rows][zsl, ti]) / (norm * np.sqrt(N_Z) + 1e-300)
            x = P @ x
        if s == 0:
            # 代表サンプルの bulk site 10 トラジェクトリを保存
            traj_ex = X[10, idxs].copy()
            x = X[rows][:, pred_t0_idx].copy()
            tp = np.zeros(n_t)
            for n in range(n_t):
                tp[n] = x[10]
                x = P @ x
            traj_pr = tp
    med_err = np.median(errs, axis=0)
    return idxs * dt, med_err, traj_ex, traj_pr

# ----------------------------------------------------------------------
records = []
pred_store = {}
for name, rows in DICTS.items():
    for mode in ("exact", "cg"):
        t1 = time.time()
        L = fit_L(rows, mode)
        co = extract_coeffs(L, len(rows))
        tarr, med_err, tr_ex, tr_pr = prediction_error(L, rows)
        pred_store[f"{name}_{mode}"] = dict(t=tarr, err=med_err,
                                            traj_exact=tr_ex, traj_pred=tr_pr)
        rec = dict(dict_name=name, n_obs=len(rows), mode=mode,
                   c2=co[0] if co else np.nan,
                   gamma=co[1] if co else np.nan,
                   nu=co[2] if co else np.nan,
                   D=(co[0] / co[1]) if (co and abs(co[1]) > 1e-6) else np.nan,
                   err_t15=med_err[np.argmin(np.abs(tarr - 1.5))],
                   err_t20=med_err[np.argmin(np.abs(tarr - 2.0))],
                   err_t30=med_err[np.argmin(np.abs(tarr - 3.0))])
        records.append(rec)
        print(f"[{name} / {mode}] n={len(rows)} "
              f"c2={rec['c2']:.4f} gamma={rec['gamma']:.4f} nu={rec['nu']:.4f} "
              f"D={rec['D']:.4f} err(t=2)={rec['err_t20']:.3e}  ({time.time()-t1:.1f}s)")

df = pd.DataFrame(records)
df.to_csv("dict_scan_results.csv", index=False)
np.savez_compressed("dict_scan_prediction.npz",
                    **{f"{k}__{kk}": vv for k, v in pred_store.items()
                       for kk, vv in v.items() if vv is not None})
print("\n=> dict_scan_results.csv / dict_scan_prediction.npz 保存完了")
print(df.to_string())
