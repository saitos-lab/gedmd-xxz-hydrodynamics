import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ==========================================
# 1. 解析設定
# ==========================================
data_file = "gedmd_raw_data.npz"
window_size = 300
stride = 10       
dt = 4.0 / 2000   
rank_tol = 1e-8

print("=== 20Qubit系: 全時間固有値プロット(4点) & ムービー生成開始 ===")

# ==========================================
# 2. データの読み込み
# ==========================================
try:
    data = np.load(data_file)
    X_S = data['X_S']; dX_S = data['dX_S']
    X_L = data['X_L']; dX_L = data['dX_L']
    X_E = data['X_E']; dX_E = data['dX_E']
except FileNotFoundError:
    print(f"エラー: {data_file} が見つかりません。")
    exit()

num_time_steps = X_S.shape[1]
num_windows = (num_time_steps - window_size) // stride + 1

# ==========================================
# 3. 解析・抽出関数
# ==========================================
#def perform_true_gedmd(X, dX, tol=1e-8):
#    if X.shape[0] == 1:
#        val = dX[0, 0] / X[0, 0] if abs(X[0,0]) > tol else 0.0
#        return np.array([val], dtype=complex)
#    
#    U, S, Vh = la.svd(X, full_matrices=False, lapack_driver='gesvd')
#    r = np.sum(S > tol)
#    if r == 0: return np.array([])
#    L = dX @ Vh[:r, :].conj().T @ np.diag(1/S[:r]) @ U[:, :r].conj().T
#    return la.eigvals(L)

def perform_true_gedmd(X, dX, rcond=1e-3):
    # (※呼び出し元の引数に合わせてデフォルト引数名を tol から rcond に変更しています)
    if X.shape[0] == 1:
        val = dX[0, 0] / X[0, 0] if abs(X[0,0]) > 1e-8 else 0.0
        return np.array([val], dtype=complex)
    
    # ==========================================
    # ★ 修正ポイント: 手動SVDをやめ、np.linalg.pinv と rcond=1e-3 を使用
    # ==========================================
    L = dX @ np.linalg.pinv(X, rcond=rcond)
    return la.eigvals(L)

def get_max_dissipation(eigs):
    real_parts = np.real(eigs)
    diss = real_parts[real_parts < -1e-3]
    return np.min(diss) if len(diss) > 0 else 0.0

# ==========================================
# 4. 全ウィンドウの逐次計算とデータ蓄積
# ==========================================
results = []
times = []
diss_S_history = []
diss_L_history = []
diss_E_history = []

# 複素平面プロット用（時間ごとの全固有値を保存）
complex_eigvals_S, complex_times_S = [], []
complex_eigvals_L, complex_times_L = [], []
complex_eigvals_E, complex_times_E = [], []

print(f"解析中 (全 {num_windows} ウィンドウ)...")

for i in range(num_windows):
    start_idx = i * stride
    end_idx = start_idx + window_size
    t_center = (start_idx + window_size / 2) * dt
    
    e_S = perform_true_gedmd(X_S[:, start_idx:end_idx], dX_S[:, start_idx:end_idx], rank_tol)
    e_L = perform_true_gedmd(X_L[:, start_idx:end_idx], dX_L[:, start_idx:end_idx], rank_tol)
    e_E = perform_true_gedmd(X_E[:, start_idx:end_idx], dX_E[:, start_idx:end_idx], rank_tol)
    
    # ムービーと時系列グラフ用のデータ保存
    results.append({'t': t_center, 'S': e_S, 'L': e_L, 'E': e_E})
    times.append(t_center)
    diss_S_history.append(get_max_dissipation(e_S))
    diss_L_history.append(get_max_dissipation(e_L))
    diss_E_history.append(get_max_dissipation(e_E))
    
    # 複素平面プロット用のデータ蓄積 (S)
    if len(e_S) > 0:
        complex_eigvals_S.extend(e_S)
        complex_times_S.extend([t_center] * len(e_S))
        
    # 複素平面プロット用のデータ蓄積 (L)
    if len(e_L) > 0:
        complex_eigvals_L.extend(e_L)
        complex_times_L.extend([t_center] * len(e_L))
        
    # 複素平面プロット用のデータ蓄積 (E)
    if len(e_E) > 0:
        complex_eigvals_E.extend(e_E)
        complex_times_E.extend([t_center] * len(e_E))
    
    if (i+1) % 20 == 0 or (i+1) == num_windows: 
        print(f"  Window {i+1}/{num_windows} (t={t_center:.3f}) 完了")

# ==========================================
# 5. 静止画(PNG) の出力（時間分割版）
# ==========================================
print("\n--- 静止画グラフの出力中（時間分割） ---")

def plot_split_complex_plane(eigs, t_array, title_base, file_base, marker='o'):
    eigs = np.array(eigs)
    t_array = np.array(t_array)
    
    # 時間範囲の定義
    ranges = [
        (0.0, 3.7, "t=0.0-3.7"),
        (3.7, 8.0, "t=3.7-8.0")
    ]
    
    for t_min, t_max, label in ranges:
        # 指定した時間範囲のデータを抽出
        mask = (t_array >= t_min) & (t_array < t_max)
        if not np.any(mask): continue
        
        plt.figure(figsize=(7, 5))
        scatter = plt.scatter(np.real(eigs[mask]), np.imag(eigs[mask]), 
                              c=t_array[mask], cmap='viridis', s=15, alpha=0.6, 
                              edgecolors='none', marker=marker, vmin=t_min, vmax=t_max)
        
        plt.axvline(x=0.0, color='black', linestyle='-', linewidth=1)
        plt.axhline(y=0.0, color='gray', linestyle='--', linewidth=0.5)
        
        plt.xlim(-2.5, 2.5)
        plt.ylim(-6, 6)
        plt.xlabel('Real Part (Dissipation)', fontsize=12)
        plt.ylabel('Imaginary Part (Frequency/Oscillation)', fontsize=12)
        plt.title(f"{title_base} ({label})", fontsize=14)
        plt.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Time (t)', fontsize=12)
        
        filename = f"{file_base}_{label.replace('.','_')}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"=> '{filename}' を出力しました。")

# 各辞書ごとに前後半の図を出力
plot_split_complex_plane(complex_eigvals_S, complex_times_S, 'Complex Plane Spectrum (Dict S)', "Fig2_ComplexPlane_S")
plot_split_complex_plane(complex_eigvals_L, complex_times_L, 'Complex Plane Spectrum (Dict L)', "Fig3_ComplexPlane_L")
plot_split_complex_plane(complex_eigvals_E, complex_times_E, 'Complex Plane Spectrum (Dict E)', "Fig4_ComplexPlane_E")

# (1) 最大散逸の時系列プロット（これは全時間で1枚）
plt.figure(figsize=(8, 5))
plt.plot(times, np.abs(diss_S_history), label='Dict S', color='red', linewidth=2)
plt.plot(times, np.abs(diss_L_history), label='Dict L', color='blue', linestyle='--', linewidth=2)
plt.plot(times, np.abs(diss_E_history), label='Dict E', color='green', linestyle=':', linewidth=2)
plt.axvline(x=4.0, color='gray', linestyle='-.')
plt.title('Time Evolution of Maximum Dissipation', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('|Re(λ)|', fontsize=12)
plt.xlim([0, 8.0])
plt.ylim([0, 1.0e3])
plt.yscale('symlog', linthresh=1e-2)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("Fig1_TimeEvolution.png", dpi=300)
plt.close()
print("=> 'Fig1_TimeEvolution.png' を出力しました。")