# ----------------------------------------------------------------------
# Part of the reproduction code for:
#   Seiki Saito, "Temporal coarse-graining as the origin of macroscopic
#   friction in quantum spin chains via data-driven Liouvillian extraction",
#   Phys. Rev. Research (2026). DOI: 10.1103/41m6-x2m9
# If you use this code, or a modified version of it, in your work,
# please cite the paper above. (MIT License; see LICENSE.)
# ----------------------------------------------------------------------
# ======================================================================
# 役割: 粗視化時間 Δt_cg を細かく変化させ、マクロ係数を計算してCSVに保存する
# ======================================================================
import glob
import numpy as np
import scipy.linalg as la
import pandas as pd
import time

print("=== [Markovian Plateau] 粗視化時間依存性のデータ計算と保存 ===")

# ==========================================
# 1. パラメータ設定
# ==========================================
dt = 0.002
tau = 1.0
t_step = 0.01
t_max = 4.0

window_steps = int(round(tau / dt))
stride_steps = int(round(t_step / dt))
num_frames = int(round((t_max - tau) / t_step)) + 1

# 計算時間を節約するため、係数が安定している「1.0秒 〜 2.5秒」の区間で時間平均をとる
eval_frames = []
for frame in range(num_frames):
    t_center = frame * t_step + tau / 2.0
    if 1.0 <= t_center <= 2.5:
        eval_frames.append(frame)

print(f"評価対象フレーム数: {len(eval_frames)} (t = 1.0 ~ 2.5s の平均を計算)")

# 検証する cg_steps のリスト (1 から 100 まで細かく振る)
cg_list = ['exact'] + list(range(1, 102, 2))
delta_t_list = [0.0 if cg == 'exact' else cg * dt for cg in cg_list]

# ==========================================
# 2. データの読み込み
# ==========================================
file_list = sorted(glob.glob("gedmd_current_len3_sample_*.npz"))
if not file_list:
    print("エラー: サンプルデータが見つかりません。")
    exit()

print(f"全 {len(file_list)} サンプルを読み込み中...")
all_X_data = [np.load(f)['X_data'] for f in file_list]
all_dX_data = [np.load(f)['dX_data'] for f in file_list] # 厳密微分 ('exact') 用

N_Z = 20; N_J = 19; N1 = N_Z + N_J
rank_tol = 1e-5

# ==========================================
# 3. 各 cg_steps での係数抽出 (時間平均)
# ==========================================
mean_c2_list = []
mean_gamma_list = []
mean_nu_list = []

start_time = time.time()

for idx, cg in enumerate(cg_list):
    c2_frames = []
    gamma_frames = []
    nu_frames = []
    
    delta_t_cg = delta_t_list[idx]
    
    for frame in eval_frames:
        t_start = frame * t_step
        start_idx = int(round(t_start / dt))
        end_idx = start_idx + window_steps
        
        X_win = np.hstack([X[:, start_idx:end_idx] for X in all_X_data])
        
        if cg == 'exact':
            if end_idx > all_X_data[0].shape[1]:
                continue
            dX_coarse = np.hstack([dX[:, start_idx:end_idx] for dX in all_dX_data])
        else:
            if end_idx + cg >= all_X_data[0].shape[1]:
                continue
            X_next = np.hstack([X[:, start_idx+cg : end_idx+cg] for X in all_X_data])
            dX_coarse = (X_next - X_win) / delta_t_cg

        U, S, Vh = la.svd(X_win, full_matrices=False, lapack_driver='gesvd')
        r = np.sum(S > rank_tol)
        
        if r > 0:
            L_L = dX_coarse @ Vh[:r, :].conj().T @ np.diag(1/S[:r]) @ U[:, :r].conj().T
            try:
                L_eff = L_L[:N1, :N1] - L_L[:N1, N1:] @ np.linalg.pinv(L_L[N1:, N1:], rcond=1e-3) @ L_L[N1:, :N1]
                L_open = np.real(L_eff)
            except np.linalg.LinAlgError:
                L_open = np.zeros((N1, N1))
        else:
            L_open = np.zeros((N1, N1))
            
        temp_c2 = np.zeros(N_J); temp_gamma = np.zeros(N_J); temp_nu = np.zeros(N_J)
        for k in range(1, N_J - 1):
            temp_c2[k] = (L_open[N_Z+k, k] - L_open[N_Z+k, k+1]) / 2.0
            temp_gamma[k] = -L_open[N_Z+k, N_Z+k]
            temp_nu[k] = (L_open[N_Z+k, N_Z+k-1] + L_open[N_Z+k, N_Z+k+1]) / 2.0
            
        c2_frames.append(np.median(temp_c2[2:-2]))
        gamma_frames.append(np.median(temp_gamma[2:-2]))
        nu_frames.append(np.median(temp_nu[2:-2]))

    # 時間平均を保存
    mean_c2_list.append(np.mean(c2_frames) if c2_frames else 0.0)
    mean_gamma_list.append(np.mean(gamma_frames) if gamma_frames else 0.0)
    mean_nu_list.append(np.mean(nu_frames) if nu_frames else 0.0)
    
    if (idx + 1) % 10 == 0:
        print(f"  進行状況: {idx + 1}/{len(cg_list)} (cg={cg}) 完了...")

print(f"計算完了: {time.time() - start_time:.2f} 秒")

# 拡散係数 D = c^2 / gamma の計算
D_list = []
for c2, gamma in zip(mean_c2_list, mean_gamma_list):
    if abs(gamma) > 1e-5:
        D_list.append(c2 / gamma)
    else:
        D_list.append(np.nan)

# ==========================================
# 4. CSVファイルへの書き出し
# ==========================================
df_results = pd.DataFrame({
    'delta_t_cg': delta_t_list,
    'c2': mean_c2_list,
    'gamma': mean_gamma_list,
    'nu': mean_nu_list,
    'D': D_list
})

output_csv = "markovian_plateau_data.csv"
df_results.to_csv(output_csv, index=False)
print(f"\n=> 抽出したデータを '{output_csv}' に保存しました。")