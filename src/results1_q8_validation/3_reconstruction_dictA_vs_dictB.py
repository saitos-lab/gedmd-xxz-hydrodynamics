import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

print("=== Q8系: Predictive Capability (Dict A vs Dict B) ===")

# ==========================================
# 1. データの読み込み
# ==========================================
print("--- データを読み込み中 ---")
X_A = np.load("q8_data_X_A.npy")
dX_A = np.load("q8_data_dX_A.npy")
X_B = np.load("q8_data_X_B.npy")
dX_B = np.load("q8_data_dX_B.npy")

num_time_steps = X_A.shape[1]
evolution_time = 3.0
dt = evolution_time / num_time_steps
time_axis = np.linspace(0, evolution_time, num_time_steps)

# ==========================================
# 2. グローバル リウビリアン L_A の抽出 (Dict A)
# ==========================================
print("--- リウビリアン L_A の計算 (Dict A: 15次元) ---")
U_A, S_A, Vh_A = la.svd(X_A, full_matrices=False, lapack_driver='gesvd')
r_A = np.sum(S_A > 1e-8)
Ur_A = U_A[:, :r_A]; Sr_A = S_A[:r_A]; Vhr_A = Vh_A[:r_A, :]

L_A = dX_A @ Vhr_A.conj().T @ np.diag(1/Sr_A) @ Ur_A.conj().T
evals_A, evecs_A = la.eig(L_A)

# ==========================================
# 3. リウビリアン L_B の抽出 (Dict B, Projected DMD)
# ==========================================
print("--- リウビリアン L_B の計算 (Dict B: 65,535次元) ---")
U_B, S_B, Vh_B = la.svd(X_B, full_matrices=False, lapack_driver='gesvd')
r_B = np.sum(S_B > 1e-10) # 精度確保のため閾値を少し下げる
Ur_B = U_B[:, :r_B]; Sr_B = S_B[:r_B]; Vhr_B = Vh_B[:r_B, :]

# 縮約ジェネレータ空間で固有値分解 (r_B x r_B 次元)
L_tilde_B = Ur_B.conj().T @ dX_B @ Vhr_B.conj().T @ np.diag(1/Sr_B)
evals_B, W_B = la.eig(L_tilde_B)

# ★修正ポイント：Projected DMDモードを採用し、計算を安定化
Phi_B = Ur_B @ W_B

# ==========================================
# 4. 解析的軌道の再構築 (e^{Lt})
# ==========================================
print("--- 解析的軌道の再構築 ---")
# Dict A (次元が小さいので固有値展開でも問題なし)
c_A = la.solve(evecs_A, X_A[:, 0])
X_recon_A = np.zeros_like(X_A, dtype=complex)
for i, t in enumerate(time_axis):
    X_recon_A[:, i] = evecs_A @ (np.exp(evals_A * t) * c_A)
X_recon_real_A = np.real(X_recon_A)

# Dict B (超高精度化: la.expmを使用)
# 初期状態をSVD空間に射影 (r_B 次元)
x0_reduced_B = Ur_B.conj().T @ X_B[:, 0]
X_recon_B = np.zeros_like(X_B, dtype=complex)

for i, t in enumerate(time_axis):
    # 固有値分解を避け、縮約ジェネレータに対して直接行列表数関数を計算
    expLt_reduced = la.expm(L_tilde_B * t)
    # 縮約空間で時間発展させた後、元の空間 (65535次元) に戻す
    X_recon_B[:, i] = Ur_B @ (expLt_reduced @ x0_reduced_B)

X_recon_real_B = np.real(X_recon_B)

# ==========================================
# 5. 結果の可視化
# ==========================================
print("--- グラフを描画中 ---")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Predictive Capability: Macroscopic (Dict A) vs Complete (Dict B)", fontsize=16)

# 代表としてプロットするインデックス (Dict Aの0番目)
target_idx_A = 3
target_data_exact = X_A[target_idx_A, :]

# X_Bの中から同じ観測量を特定する
diffs = np.sum(np.abs(X_B - target_data_exact), axis=1)
target_idx_B = np.argmin(diffs)

# --- パネル1: 軌道の重ね合わせ ---
ax1.plot(time_axis, target_data_exact, color='black', linewidth=3, alpha=0.3, label='Exact Schrödinger Dynamics')
ax1.plot(time_axis, X_recon_real_A[target_idx_A, :], color='blue', linestyle='--', linewidth=2, label='Dict A Reconstruction ($L_{\mathrm{A}}$)')
ax1.plot(time_axis, X_recon_real_B[target_idx_B, :], color='red', linestyle=':', linewidth=2, label='Dict B Reconstruction ($L_{\mathrm{B}}$)')

ax1.set_title(r"Trajectory of Bulk Spin Density $\langle \psi(t) | Z_3 | \psi(t) \rangle$", fontsize=14)
ax1.set_ylabel(r"Expectation Value $\langle Z_3(t) \rangle$", fontsize=12)
ax1.set_ylim(-0.08, 0.08) # Y軸の範囲を調整
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=11)
#ax1.axvspan(1.5, 3.0, color='yellow', alpha=0.1, label='Poincaré Recurrence (Revival) Phase')

# --- パネル2: 再構築誤差 (Absolute Error) に変更 ---
# ゼロ割りを防ぐため、純粋な差の絶対値を取る
abs_error_A = np.abs(target_data_exact - X_recon_real_A[target_idx_A, :])
abs_error_B = np.abs(target_data_exact - X_recon_real_B[target_idx_B, :])

ax2.plot(time_axis, abs_error_A, color='blue', linewidth=1.5, label='Dict A Absolute Error')
ax2.plot(time_axis, abs_error_B, color='red', linewidth=1.5, label='Dict B Absolute Error')

ax2.set_title("Absolute Reconstruction Error", fontsize=14)
ax2.set_xlabel("Time", fontsize=12)
ax2.set_ylabel(r"Absolute Error $|\langle Z_3 \rangle_{\mathrm{exact}} - \langle Z_3 \rangle_{\mathrm{recon}}|$", fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=11)

ax2.set_yscale('log')
ax2.set_ylim(1e-8, 1) # Y軸の範囲を調整

plt.tight_layout()
output_filename = "Fig_Predictive_Capability_Fixed.png"
plt.savefig(output_filename, dpi=300)
print(f"=> '{output_filename}' を出力しました！")
plt.show()