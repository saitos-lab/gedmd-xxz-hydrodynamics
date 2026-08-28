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
# 1. 設定 (8qubit系: S=2, E=6)
# ==========================================
num_qubits = 8
num_target = 2  # S系のqubit数 (0, 1)
J = 1.0; Delta = 1.0; J2 = 0.5
dt = 0.005
num_steps_phase = 2000 # フェーズごと(孤立/結合)のステップ数

print(f"=== [Data Gen] ラプラスの悪魔 (全自由度: {4**num_qubits - 1}次元) 2フェーズ時間発展 ===")

def make_pauli_str(op_dict, n_qubits):
    p = ['I'] * n_qubits
    for idx, char in op_dict.items(): p[idx] = char
    return "".join(p)[::-1] # Qiskitのエンディアン(右端がq0)に合わせる

# ==========================================
# 2. ハミルトニアン構築
# ==========================================
print("--- ハミルトニアン構築中 ---")
pauli_iso = []

# --- S内部 (qubit 0, 1) ---
pauli_iso.append((make_pauli_str({0:'X', 1:'X'}, num_qubits), J))
pauli_iso.append((make_pauli_str({0:'Y', 1:'Y'}, num_qubits), J))
pauli_iso.append((make_pauli_str({0:'Z', 1:'Z'}, num_qubits), Delta))

# --- E内部 (qubit 2〜7) ---
for i in range(2, num_qubits - 1): # NN
    pauli_iso.append((make_pauli_str({i:'X', i+1:'X'}, num_qubits), J))
    pauli_iso.append((make_pauli_str({i:'Y', i+1:'Y'}, num_qubits), J))
    pauli_iso.append((make_pauli_str({i:'Z', i+1:'Z'}, num_qubits), Delta))
for i in range(2, num_qubits - 2): # NNN
    pauli_iso.append((make_pauli_str({i:'Z', i+2:'Z'}, num_qubits), J2))

H_iso_sparse = SparsePauliOp.from_list(pauli_iso).to_matrix(sparse=True)

# --- 相互作用項 (SとEを繋ぐ) ---
pauli_int = pauli_iso.copy()
# NN: 境界 (1と2)
pauli_int.append((make_pauli_str({1:'X', 2:'X'}, num_qubits), J))
pauli_int.append((make_pauli_str({1:'Y', 2:'Y'}, num_qubits), J))
pauli_int.append((make_pauli_str({1:'Z', 2:'Z'}, num_qubits), Delta))
# NNN: 境界をまたぐ (0と2, 1と3)
pauli_int.append((make_pauli_str({0:'Z', 2:'Z'}, num_qubits), J2))
pauli_int.append((make_pauli_str({1:'Z', 3:'Z'}, num_qubits), J2))

H_coupled_sparse = SparsePauliOp.from_list(pauli_int).to_matrix(sparse=True)

# ==========================================
# 3. 全辞書 (65,535次元) の構築
# ==========================================
print("--- 観測量行列の構築中 (Sparse行列でメモリ節約) ---")
labels = ['I', 'X', 'Y', 'Z']
all_pauli_strs = []
sub_indices = []

idx = 0
for p_tuple in itertools.product(labels, repeat=num_qubits):
    if all(c == 'I' for c in p_tuple): 
        continue # 単位行列は除外
    
    # p_tuple = (q0, q1, q2, ..., q7)
    p_dict = {i: p_tuple[i] for i in range(num_qubits)}
    all_pauli_strs.append(make_pauli_str(p_dict, num_qubits))
    
    # S系(qubit 0,1)のみのインデックス抽出 (E系であるq2〜q7がすべて'I'のもの)
    is_S = all(p_tuple[i] == 'I' for i in range(num_target, num_qubits))
    if is_S:
        sub_indices.append(idx)
    
    idx += 1

ops_sparse = [SparsePauliOp(p).to_matrix(sparse=True) for p in all_pauli_strs]
print(f"-> 全辞書: {len(ops_sparse)} 次元, Target S部分: {len(sub_indices)} 次元")

# ==========================================
# 4. 厳密時間発展 (Phase1 -> Phase2)
# ==========================================
np.random.seed(42)
state = (np.random.rand(2**num_qubits) - 0.5) + 1j * (np.random.rand(2**num_qubits) - 0.5)
state /= la.norm(state)

def evolve_phase(H_sparse, current_state, steps, phase_name):
    print(f"\n--- {phase_name} 実行中 ---")
    X = np.zeros((len(ops_sparse), steps))
    dX = np.zeros((len(ops_sparse), steps))
    
    start_time = time.time()
    for t in range(steps):
        phi = H_sparse.dot(current_state)
        
        # einsumの代わりに、Sparse行列を1つずつforループで処理
        for k, op in enumerate(ops_sparse):
            O_psi = op.dot(current_state)
            X[k, t] = np.real(current_state.conj().T @ O_psi)
            dX[k, t] = -2.0 * np.imag(phi.conj().T @ O_psi)
            
        if (t + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {t + 1}/{steps} 完了... (経過: {elapsed:.1f}s)")
            
        if t < steps - 1:
            current_state = spla.expm_multiply(-1j * H_sparse * dt, current_state)
            
    return current_state, X, dX

# 孤立系 (Phase 1)
state, X_p1, dX_p1 = evolve_phase(H_iso_sparse, state, num_steps_phase, "Phase 1 (孤立系: t=0.0~10.0)")

# 結合系 (Phase 2)
state, X_p2, dX_p2 = evolve_phase(H_coupled_sparse, state, num_steps_phase, "Phase 2 (結合系: t=10.0~20.0)")

# データの結合
X_all = np.hstack([X_p1, X_p2]); dX_all = np.hstack([dX_p1, dX_p2])
X_sub = X_all[sub_indices, :]; dX_sub = dX_all[sub_indices, :]

# ==========================================
# 5. データの保存
# ==========================================
print("\n--- データの保存中 ---")
filename = "q8_demon_data_2phase.npz"
np.savez_compressed(filename, X_all=X_all, dX_all=dX_all, X_sub=X_sub, dX_sub=dX_sub)
print(f"=> '{filename}' を保存しました。")