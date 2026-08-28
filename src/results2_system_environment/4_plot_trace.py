# ----------------------------------------------------------------------
# Part of the reproduction code for:
#   Seiki Saito, "Temporal coarse-graining as the origin of macroscopic
#   friction in quantum spin chains via data-driven Liouvillian extraction",
#   Phys. Rev. Research (2026). DOI: 10.1103/41m6-x2m9
# If you use this code, or a modified version of it, in your work,
# please cite the paper above. (MIT License; see LICENSE.)
# ----------------------------------------------------------------------
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# ==========================================
# 1〜4. データ読み込み・計算（変更なし）
# ==========================================
data_file = "gedmd_raw_data.npz"
window_size = 300
stride = 10       
dt = 4.0 / 2000   
rank_tol = 1e-8

print("=== リウビリアントレース（情報流出入）の解析開始 ===")

try:
    data = np.load(data_file)
    X_S = data['X_S']; dX_S = data['dX_S']
    X_L = data['X_L']; dX_L = data['dX_L']
except FileNotFoundError:
    print(f"エラー: {data_file} が見つかりません。")
    exit()

num_time_steps = X_S.shape[1]
num_windows = (num_time_steps - window_size) // stride + 1

def perform_true_gedmd(X, dX, tol=1e-8):
    if X.shape[0] == 1:
        val = dX[0, 0] / X[0, 0] if abs(X[0,0]) > tol else 0.0
        return np.array([val], dtype=complex)
        
    U, S, Vh = la.svd(X, full_matrices=False, lapack_driver='gesvd')
    r = np.sum(S > tol)
    if r == 0: return np.array([])
    L = dX @ Vh[:r, :].conj().T @ np.diag(1/S[:r]) @ U[:, :r].conj().T
    return la.eigvals(L)

times = []
trace_S_history = []
trace_L_history = []

print(f"解析中 (全 {num_windows} ウィンドウ)...")

for i in range(num_windows):
    start_idx = i * stride
    end_idx = start_idx + window_size
    t_center = (start_idx + window_size / 2) * dt
    
    e_S = perform_true_gedmd(X_S[:, start_idx:end_idx], dX_S[:, start_idx:end_idx], rank_tol)
    e_L = perform_true_gedmd(X_L[:, start_idx:end_idx], dX_L[:, start_idx:end_idx], rank_tol)
    
    times.append(t_center)
    
    tr_S = np.sum(np.real(e_S)) if len(e_S) > 0 else 0.0
    tr_L = np.sum(np.real(e_L)) if len(e_L) > 0 else 0.0
    
    trace_S_history.append(tr_S)
    trace_L_history.append(tr_L)

# ==========================================
# 5. グラフの出力 (改良版)
# ==========================================
print("\n--- グラフの出力中 ---")

# ---------------------------------------------------------
# パターンA：Symlog (対称対数) プロット
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(times, trace_S_history, label='Target System (Dict S) [Outflow]', color='red', linewidth=2)
plt.plot(times, trace_L_history, label='Observer/Pointer (Dict L) [Inflow]', color='blue', linewidth=2)

plt.axvline(x=4.0, color='black', linestyle='-.', label='Interaction ON (t=4.0)')
plt.axhline(y=0.0, color='gray', linestyle='-', linewidth=1)
plt.axvspan(0, 4.0, facecolor='lightgray', alpha=0.3)

plt.title('Time Evolution of Information Flow (Symlog Scale)', fontsize=15)
plt.xlabel('Time (t)', fontsize=13)
plt.ylabel('Trace Re(L) [Rate of Phase Volume Change]', fontsize=13)

# ここが魔法のSymlog設定 (linthresh=1.0 は -1から1の間を線形にする閾値)
plt.yscale('symlog', linthresh=1.0) 

plt.grid(True, alpha=0.4)
plt.legend(loc='best', fontsize=11)
plt.tight_layout()
plt.savefig("Fig5_Trace_Symlog.png", dpi=300)
print("=> 'Fig5_Trace_Symlog.png' を出力しました。")

# ---------------------------------------------------------
# パターンB：2軸 (Dual Y-Axis) プロット（0点合わせ版）
# ---------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 6))

plt.title('Time Evolution of Information Flow (Dual Y-Axis)', fontsize=15)
ax1.set_xlabel('Time (t)', fontsize=13)
ax1.axvline(x=4.0, color='black', linestyle='-.')
ax1.axhline(y=0.0, color='gray', linestyle='-', linewidth=1)
ax1.axvspan(0, 4.0, facecolor='lightgray', alpha=0.3)

# 左軸 (Dict S 用)
color_S = 'red'
ax1.set_ylabel('Tr $L$ of Dict S', color=color_S, fontsize=13)
ax1.plot(times, trace_S_history, color=color_S, linewidth=2, label='Dict S')
ax1.tick_params(axis='y', labelcolor=color_S)

# 右軸を作成 (Dict L 用)
ax2 = ax1.twinx()  
color_L = 'blue'
ax2.set_ylabel('Tr $L$ of Dict L', color=color_L, fontsize=13)
ax2.plot(times, trace_L_history, color=color_L, linewidth=1, label='Dict L')
ax2.tick_params(axis='y', labelcolor=color_L)

# =========================================================
# ★ ここが修正ポイント：0点（ゼロライン）を一致させる処理
# =========================================================
# 左軸の最大絶対値を取得して、上下対称に設定
y1_abs_max = max(abs(np.min(trace_S_history)), abs(np.max(trace_S_history)))
ax1.set_ylim(-y1_abs_max * 1.1, y1_abs_max * 1.1)  # 少し余白を持たせる

# 右軸の最大絶対値を取得して、上下対称に設定
y2_abs_max = max(abs(np.min(trace_L_history)), abs(np.max(trace_L_history)))
ax2.set_ylim(-y2_abs_max * 1.1, y2_abs_max * 1.1)
# =========================================================

# 凡例を統合して表示
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best', fontsize=11)

plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("Fig5_Trace_DualAxis.png", dpi=300)
print("=> 'Fig5_Trace_DualAxis.png' を出力しました！")
