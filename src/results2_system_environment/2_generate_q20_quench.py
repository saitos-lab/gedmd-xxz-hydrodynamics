# ----------------------------------------------------------------------
# Part of the reproduction code for:
#   Seiki Saito, "Temporal coarse-graining as the origin of macroscopic
#   friction in quantum spin chains via data-driven Liouvillian extraction",
#   Phys. Rev. Research (2026). DOI: 10.1103/41m6-x2m9
# If you use this code, or a modified version of it, in your work,
# please cite the paper above. (MIT License; see LICENSE.)
# ----------------------------------------------------------------------
import time
import itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
from qiskit.quantum_info import SparsePauliOp

# ==========================================
# 1. 設定
# ==========================================
num_qubits = 20  
num_target = 4
num_env = 16

# --- カオス的XXZモデルのパラメータ ---
J = 1.0      # XY平面でのスピンホッピング (カレントの源)
Delta = 1.0  # Z方向の隣接相互作用
J2 = 0.5     # Z方向の次近接相互作用 (可積分性の破壊・熱化の源)

dt = 0.002
num_steps_phase = 2000

print(f"=== スクリプトA: カオス的XXZ 真のgEDMD 生データ生成と保存 ===")

# ==========================================
# 2. ハミルトニアンの構築
# ==========================================
print("\n--- ハミルトニアン構築中 ---")
def make_pauli_str(op_dict, n_qubits):
    p = ['I'] * n_qubits
    for idx, char in op_dict.items(): p[idx] = char
    return "".join(p)[::-1]

pauli_isolated = []

# --- 小系 S (0 ~ 3) の内部ハミルトニアン ---
for i in range(num_target - 1): # NN
    pauli_isolated.append((make_pauli_str({i:'X', i+1:'X'}, num_qubits), J))
    pauli_isolated.append((make_pauli_str({i:'Y', i+1:'Y'}, num_qubits), J))
    pauli_isolated.append((make_pauli_str({i:'Z', i+1:'Z'}, num_qubits), Delta))
for i in range(num_target - 2): # NNN
    pauli_isolated.append((make_pauli_str({i:'Z', i+2:'Z'}, num_qubits), J2))

# --- 熱浴 E (4 ~ 19) の内部ハミルトニアン ---
for i in range(num_target, num_qubits - 1): # NN
    pauli_isolated.append((make_pauli_str({i:'X', i+1:'X'}, num_qubits), J))
    pauli_isolated.append((make_pauli_str({i:'Y', i+1:'Y'}, num_qubits), J))
    pauli_isolated.append((make_pauli_str({i:'Z', i+1:'Z'}, num_qubits), Delta))
for i in range(num_target, num_qubits - 2): # NNN
    pauli_isolated.append((make_pauli_str({i:'Z', i+2:'Z'}, num_qubits), J2))

# 孤立系ハミルトニアン (Phase 1用)
H_iso = SparsePauliOp.from_list(pauli_isolated).to_matrix(sparse=True)

# --- 相互作用項の追加 (S と E をシームレスに繋ぐ) ---
pauli_int = pauli_isolated.copy()
# NN: 境界 (3 と 4) を繋ぐ
pauli_int.append((make_pauli_str({3:'X', 4:'X'}, num_qubits), J))
pauli_int.append((make_pauli_str({3:'Y', 4:'Y'}, num_qubits), J))
pauli_int.append((make_pauli_str({3:'Z', 4:'Z'}, num_qubits), Delta))
# NNN: 境界をまたぐ次近接相互作用 (2と4, 3と5)
pauli_int.append((make_pauli_str({2:'Z', 4:'Z'}, num_qubits), J2))
pauli_int.append((make_pauli_str({3:'Z', 5:'Z'}, num_qubits), J2))

# 結合系ハミルトニアン (Phase 2用)
H_int = SparsePauliOp.from_list(pauli_int).to_matrix(sparse=True)

# ==========================================
# 3. 3種類の観測辞書の構築
# ==========================================
print("\n--- 観測辞書の構築中 ---")

# ① 辞書S (Small): 局所系の完全な情報 (変更なし, 255次元)
dict_S_paulis = []
for p in itertools.product(['I','X','Y','Z'], repeat=num_target):
    if all(c=='I' for c in p): continue
    dict_S_paulis.append(make_pauli_str({i: p[i] for i in range(num_target)}, num_qubits))
ops_S = [SparsePauliOp(p).to_matrix(sparse=True) for p in dict_S_paulis]

# ② 辞書L (Large): 巨大熱浴の「マクロ流体」辞書 (スピンカレントを追加!)
dict_L_paulis = []
# 1. 1体スピン磁化 (Z)
for i in range(num_target, num_qubits):
    dict_L_paulis.append(make_pauli_str({i: 'Z'}, num_qubits))
# 2. 2体相関 (ZZ)
for i in range(num_target, num_qubits - 1):
    dict_L_paulis.append(make_pauli_str({i: 'Z', i+1: 'Z'}, num_qubits))
# 3. スピンカレント (XY, YX) - 流体力学の連続の式を閉じるための必須項
for i in range(num_target, num_qubits - 1):
    dict_L_paulis.append(make_pauli_str({i: 'X', i+1: 'Y'}, num_qubits))
    dict_L_paulis.append(make_pauli_str({i: 'Y', i+1: 'X'}, num_qubits))
ops_L = [SparsePauliOp(p).to_matrix(sparse=True) for p in dict_L_paulis]

# ③ 辞書E (Energy): 熱浴のXXZカオスエネルギー
pauli_env_energy = []
for i in range(num_target, num_qubits - 1):
    pauli_env_energy.append((make_pauli_str({i:'X', i+1:'X'}, num_qubits), J))
    pauli_env_energy.append((make_pauli_str({i:'Y', i+1:'Y'}, num_qubits), J))
    pauli_env_energy.append((make_pauli_str({i:'Z', i+1:'Z'}, num_qubits), Delta))
for i in range(num_target, num_qubits - 2):
    pauli_env_energy.append((make_pauli_str({i:'Z', i+2:'Z'}, num_qubits), J2))
ops_E = [SparsePauliOp.from_list(pauli_env_energy).to_matrix(sparse=True)]

# ==========================================
# 4. 厳密時間発展とデータ取得
# ==========================================
np.random.seed(42)
#state = np.random.rand(2**num_qubits) + 1j * np.random.rand(2**num_qubits)
state = (np.random.rand(2**num_qubits) - 0.5) + 1j * (np.random.rand(2**num_qubits) - 0.5)
state /= la.norm(state)

def collect_3dicts(H_sparse, current_state, steps):
    X_S = np.zeros((len(ops_S), steps)); dX_S = np.zeros((len(ops_S), steps))
    X_L = np.zeros((len(ops_L), steps)); dX_L = np.zeros((len(ops_L), steps))
    X_E = np.zeros((len(ops_E), steps)); dX_E = np.zeros((len(ops_E), steps))
    
    start_time = time.time()
    for t in range(steps):
        phi = H_sparse.dot(current_state)
        
        for k, op in enumerate(ops_S):
            O_psi = op.dot(current_state)
            X_S[k, t] = np.real(current_state.conj().T @ O_psi)
            dX_S[k, t] = -2.0 * np.imag(phi.conj().T @ O_psi)
            
        for k, op in enumerate(ops_L):
            O_psi = op.dot(current_state)
            X_L[k, t] = np.real(current_state.conj().T @ O_psi)
            dX_L[k, t] = -2.0 * np.imag(phi.conj().T @ O_psi)
            
        for k, op in enumerate(ops_E):
            O_psi = op.dot(current_state)
            X_E[k, t] = np.real(current_state.conj().T @ O_psi)
            dX_E[k, t] = -2.0 * np.imag(phi.conj().T @ O_psi)
            
        if (t + 1) % 200 == 0:
            print(f"  Step {t + 1}/{steps} 完了... (経過: {time.time() - start_time:.1f}s)")
            
        if t < steps - 1:
            current_state = spla.expm_multiply(-1j * H_sparse * dt, current_state)
            
    return current_state, (X_S, dX_S), (X_L, dX_L), (X_E, dX_E)

print("\n--- フェーズ1 (孤立系: t=0.0~4.0) データ取得中 ---")
state, data_S1, data_L1, data_E1 = collect_3dicts(H_iso, state, num_steps_phase)

print("\n--- フェーズ2 (結合系: t=4.0~8.0) データ取得中 ---")
state, data_S2, data_L2, data_E2 = collect_3dicts(H_int, state, num_steps_phase)

# データの結合
full_X_S = np.hstack([data_S1[0], data_S2[0]]); full_dX_S = np.hstack([data_S1[1], data_S2[1]])
full_X_L = np.hstack([data_L1[0], data_L2[0]]); full_dX_L = np.hstack([data_L1[1], data_L2[1]])
full_X_E = np.hstack([data_E1[0], data_E2[0]]); full_dX_E = np.hstack([data_E1[1], data_E2[1]])
time_axis = np.linspace(0, (num_steps_phase * 2) * dt, num_steps_phase * 2)

# ==========================================
# 5. 生データの保存
# ==========================================
filename = "gedmd_raw_data.npz"
np.savez_compressed(filename, 
                    time_axis=time_axis,
                    X_S=full_X_S, dX_S=full_dX_S,
                    X_L=full_X_L, dX_L=full_dX_L,
                    X_E=full_X_E, dX_E=full_dX_E)

print(f"\nデータ生成完了！ '{filename}' に保存しました。")
