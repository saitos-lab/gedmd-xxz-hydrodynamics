# ----------------------------------------------------------------------
# Part of the reproduction code for:
#   Seiki Saito, "Temporal coarse-graining as the origin of macroscopic
#   friction in quantum spin chains via data-driven Liouvillian extraction",
#   Phys. Rev. Research (2026). DOI: 10.1103/41m6-x2m9
# If you use this code, or a modified version of it, in your work,
# please cite the paper above. (MIT License; see LICENSE.)
# ----------------------------------------------------------------------
# ======================================================================
# データ生成・保存: 様々な粗視化時間幅と厳密微分での係数抽出 (対称化なし)
# ======================================================================
import glob
import numpy as np
import scipy.linalg as la
import os

print("=== [Data Generation] 各粗視化ステップでのマクロ係数抽出と保存 ===")

dt = 0.002
tau = 1.0
t_step = 0.01
t_max = 4.0

window_steps = int(round(tau / dt))
stride_steps = int(round(t_step / dt))
num_frames = int(round((t_max - tau) / t_step)) + 1

file_list = sorted(glob.glob("gedmd_current_len3_sample_*.npz"))
if not file_list:
    print("エラー: サンプルデータが見つかりません。")
    exit()

print(f"全 {len(file_list)} サンプルを読み込み中...")
all_X_data = [np.load(f)['X_data'] for f in file_list]
all_dX_data = [np.load(f)['dX_data'] for f in file_list] # 厳密微分用にdXも読み込む

N_Z = 20; N_J = 19; N1 = N_Z + N_J
rank_tol = 1e-5 # 情報は切り捨てない (フルランクで解く)

def moving_average(arr, window=5):
    pad_width = window // 2
    padded_arr = np.pad(arr, pad_width, mode='edge')
    return np.convolve(padded_arr, np.ones(window)/window, mode='valid')

# 比較する設定のリスト ('exact' は厳密な時間微分 dX を使用)
# 論文 Fig. 8 はこの設定 (Δt_cg = 0.004, 0.01, 0.02, 0.04, 0.10) を使用
cg_settings = ['exact', 2, 5, 10, 20, 50]

for cg in cg_settings:
    if cg == 'exact':
        print("\n--- 処理中: 厳密微分 (Exact Commutator) ---")
    else:
        print(f"\n--- 処理中: 有限差分 cg_steps = {cg} (Δt_cg = {cg*dt:.3f}) ---")
        
    time_axis = []
    raw_c2, raw_gamma, raw_nu, raw_D_Z = [], [], [], []

    for frame in range(num_frames):
        t_start = frame * t_step
        start_idx = int(round(t_start / dt))
        end_idx = start_idx + window_steps
        
        # データの終端判定
        if cg == 'exact':
            if end_idx > all_X_data[0].shape[1]:
                break
        else:
            if end_idx + cg >= all_X_data[0].shape[1]:
                break
            
        time_axis.append(t_start + tau / 2.0)
        
        # 現在のウィンドウ
        X_win = np.hstack([X[:, start_idx:end_idx] for X in all_X_data])
        
        # 厳密微分か有限差分かの切り替え
        if cg == 'exact':
            dX_eff = np.hstack([dX[:, start_idx:end_idx] for dX in all_dX_data])
        else:
            X_next = np.hstack([X[:, start_idx+cg : end_idx+cg] for X in all_X_data])
            delta_t_cg = cg * dt
            dX_eff = (X_next - X_win) / delta_t_cg
        
        # gEDMD実行
        U, S, Vh = la.svd(X_win, full_matrices=False, lapack_driver='gesvd')
        r = np.sum(S > rank_tol)
        
        if r > 0:
            L_L = dX_eff @ Vh[:r, :].conj().T @ np.diag(1/S[:r]) @ U[:, :r].conj().T
            try:
                L_eff = L_L[:N1, :N1] - L_L[:N1, N1:] @ np.linalg.pinv(L_L[N1:, N1:], rcond=1e-3) @ L_L[N1:, :N1]
                L_open = np.real(L_eff)
            except np.linalg.LinAlgError:
                L_open = np.zeros((N1, N1))
        else:
            L_open = np.zeros((N1, N1))
            
        # パリティ対称化は行わない (生の L_open をそのまま使用)

        # 係数抽出
        temp_c2 = np.zeros(N_J); temp_gamma = np.zeros(N_J)
        temp_nu = np.zeros(N_J); temp_DZ = np.zeros(N_Z)
        
        for k in range(1, N_J - 1):
            temp_c2[k] = (L_open[N_Z+k, k] - L_open[N_Z+k, k+1]) / 2.0
            temp_gamma[k] = -L_open[N_Z+k, N_Z+k]
            temp_nu[k] = (L_open[N_Z+k, N_Z+k-1] + L_open[N_Z+k, N_Z+k+1]) / 2.0
        for i in range(1, N_Z - 1):
            temp_DZ[i] = -L_open[i, i] / 2.0

        raw_c2.append(np.median(temp_c2[2:-2]))
        raw_gamma.append(np.median(temp_gamma[2:-2]))
        raw_nu.append(np.median(temp_nu[2:-2]))
        raw_D_Z.append(np.median(temp_DZ[2:-2]))

    # スムージングとDの計算
    t_array = np.array(time_axis)
    smooth_c2 = moving_average(np.array(raw_c2), 5)
    smooth_gamma = moving_average(np.array(raw_gamma), 5)
    smooth_nu = moving_average(np.array(raw_nu), 5)
    smooth_D_Z = moving_average(np.array(raw_D_Z), 5)

    smooth_D = np.zeros(len(t_array))
    for i in range(len(t_array)):
        if abs(smooth_gamma[i]) > 1e-5: 
            smooth_D[i] = smooth_c2[i] / smooth_gamma[i]

    # 結果をファイルに保存
    save_name = f"cg_data_{cg}.npz"
    np.savez(save_name, 
             time=t_array, 
             c2=smooth_c2, 
             gamma=smooth_gamma, 
             nu=smooth_nu, 
             D=smooth_D, 
             Dz=smooth_D_Z)
    print(f"=> '{save_name}' を保存しました。")

print("\nすべてのデータ生成と保存が完了しました。")