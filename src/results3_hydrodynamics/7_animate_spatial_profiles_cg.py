# ======================================================================
# Sliding-window animation of the bulk spatial profiles (c^2, gamma, nu, D)
# under temporal coarse-graining, forward difference with cg_steps = 20
# (Δt_cg = 0.04) (Supplementary Video 5).
# Fig. 10(b) of the paper is the frame whose window is t = 0.5 ~ 1.5.
# No parity symmetrization is applied: the raw L_open is used as-is,
# matching the published figure and video.
# ======================================================================
import glob
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as animation

print("=== [Spatial Profiles] 有限差分 (cg=20) によるマクロ流体アニメーション ===")

# ==========================================
# 1. パラメータ設定
# ==========================================
dt = 0.002       # シミュレーションの時間刻み
tau = 1.0        # 時間窓のサイズ
t_step = 0.01    # スライドさせる幅
t_max = 4.0      # 最大シミュレーション時間

# 時間粗視化パラメータ (cg_steps * dt = Δt_cg)
cg_steps = 20
delta_t_cg = cg_steps * dt
print(f"粗視化時間幅: Δt_cg = {delta_t_cg:.3f} (cg_steps = {cg_steps})")

window_steps = int(round(tau / dt))
stride_steps = int(round(t_step / dt))
num_frames = int(round((t_max - tau) / t_step)) + 1

# ==========================================
# 2. データの事前読み込み
# ==========================================
file_list = sorted(glob.glob("gedmd_current_len3_sample_*.npz"))
if not file_list:
    print("エラー: サンプルデータが見つかりません。")
    exit()

print(f"{len(file_list)} 個のサンプルデータをメモリに読み込んでいます...")
all_X_data = [np.load(f)['X_data'] for f in file_list]
# 有限差分を使用するため、厳密微分(dX_data)は読み込み不要です

N_Z = 20; N_J = 19; N1 = N_Z + N_J
rank_tol = 1e-4

# ==========================================
# 3. アニメーション設定・更新関数の定義
# ==========================================
fig, axes = plt.subplots(4, 1, figsize=(10, 9))
ax1, ax2, ax3, ax4 = axes

def moving_average(arr, window=4):
    valid_arr = arr[1:-1]
    if len(valid_arr) < window: return valid_arr
    return np.convolve(valid_arr, np.ones(window)/window, mode='valid')

def update(frame):
    t_start = frame * t_step
    t_end = t_start + tau

    start_idx = int(round(t_start / dt))
    end_idx = start_idx + window_steps

    # データの終端チェック (未来のデータが取れない場合はアニメーションを停止)
    if end_idx + cg_steps >= all_X_data[0].shape[1]:
        return

    X_win = np.hstack([X[:, start_idx:end_idx] for X in all_X_data])

    # 有限差分で近似微分を計算
    X_next = np.hstack([X[:, start_idx+cg_steps : end_idx+cg_steps] for X in all_X_data])
    dX_coarse = (X_next - X_win) / delta_t_cg

    # gEDMD と Mori-Zwanzig 抽出
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

    # パリティ対称化は行わない (生の L_open をそのまま使用)

    # 係数の抽出
    c2_array = np.zeros(N_J)
    gamma_array = np.zeros(N_J)
    nu_array = np.zeros(N_J)
    D_Z_array = np.zeros(N_Z)

    for k in range(1, N_J - 1):
        c2_array[k] = (L_open[N_Z+k, k] - L_open[N_Z+k, k+1]) / 2.0
        gamma_array[k] = -L_open[N_Z+k, N_Z+k]
        nu_array[k] = (L_open[N_Z+k, N_Z+k-1] + L_open[N_Z+k, N_Z+k+1]) / 2.0

    for i in range(1, N_Z - 1):
        D_Z_array[i] = -L_open[i, i] / 2.0

    # 空間移動平均
    c2_smooth = moving_average(c2_array)
    gamma_smooth = moving_average(gamma_array)
    nu_smooth = moving_average(nu_array)
    D_Z_smooth = moving_average(D_Z_array)

    # D = c^2 / \gamma の計算
    D_macro = np.zeros_like(c2_smooth)
    for idx in range(len(c2_smooth)):
        if abs(gamma_smooth[idx]) > 1e-10:
            D_macro[idx] = c2_smooth[idx] / gamma_smooth[idx]

    # === 描画のクリアと再設定 ===
    for ax in axes: ax.clear()
    x_axis = np.arange(len(c2_smooth))

    fig.suptitle(f"Emergent Hydrodynamics (Coarse-grained: $\\Delta t_{{cg}}={delta_t_cg}$)\n(Window: $\\tau={tau}$, Time: $t = {t_start:.2f} \\sim {t_end:.2f}$)", fontsize=15)

    ax1.plot(x_axis, c2_smooth, 's-', color='teal', linewidth=2)
    ax1.set_title("Pressure Gradient $c^2$ (Elasticity)", fontsize=12)
    ax1.axhline(0, color='black', linestyle='--'); ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-2, 6])

    ax2.plot(x_axis, gamma_smooth, 'D-', color='purple', linewidth=2)
    ax2.set_title(r"Local Friction $\gamma$ (Irreversible Dissipation)", fontsize=12)
    ax2.axhline(0, color='black', linestyle='--'); ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-2, 6])

    ax3.plot(x_axis, nu_smooth, '^-', color='green', linewidth=2)
    ax3.set_title(r"Kinematic Viscosity $\nu$ (Momentum Diffusion)", fontsize=12)
    ax3.axhline(0, color='black', linestyle='--'); ax3.grid(True, alpha=0.3)
    ax3.set_ylim([-2, 3])

    ax4.plot(x_axis, D_macro, 'o-', color='coral', linewidth=2, label=r"Einstein Rel: $D = c^2/\gamma$")
    ax4.plot(x_axis, D_Z_smooth[:len(x_axis)], 'x--', color='red', linewidth=1.5, label="Direct $D_Z$")
    ax4.set_title("Macroscopic Diffusion $D$ (SymLog Scale)", fontsize=12)
    ax4.set_xlabel("Smoothed Bulk Region")
    ax4.axhline(0, color='black', linestyle='--'); ax4.grid(True, alpha=0.3)
    ax4.set_yscale('symlog', linthresh=0.1)
    ax4.set_ylim([-500, 500])
    ax4.legend(loc='upper right')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    if (frame+1) % 10 == 0 or (frame+1) == num_frames:
        print(f"Frame {frame+1:03d}/{num_frames} (t={t_start:.2f}) 処理完了")

# ==========================================
# 4. 動画の生成と保存
# ==========================================
print(f"\n全 {num_frames} フレームのアニメーションを生成します...")
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=200)

out_filename_gif = "SupVideo5_Spatial_Profiles_CG.gif"
print(f"'{out_filename_gif}' として保存します...")
ani.save(out_filename_gif, writer='pillow', fps=5, dpi=150)
print(f"=> '{out_filename_gif}' を出力しました！")
