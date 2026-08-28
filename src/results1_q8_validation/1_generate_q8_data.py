# ----------------------------------------------------------------------
# Part of the reproduction code for:
#   Seiki Saito, "Temporal coarse-graining as the origin of macroscopic
#   friction in quantum spin chains via data-driven Liouvillian extraction",
#   Phys. Rev. Research (2026). DOI: 10.1103/41m6-x2m9
# If you use this code, or a modified version of it, in your work,
# please cite the paper above. (MIT License; see LICENSE.)
# ----------------------------------------------------------------------
# ======================================================================
# Results I (Figs. 2-4): 8-qubit homogeneous chaotic XXZ chain.
# Records X(t) and the exact derivative dX(t) for the macroscopic
# dictionary A (Z_i + Z_i Z_{i+1}, 15 observables) and the complete
# Pauli dictionary B (4^8 - 1 = 65,535 observables).
# Outputs: q8_data_X_A.npy, q8_data_dX_A.npy, q8_data_X_B.npy, q8_data_dX_B.npy
# ======================================================================
import time
import itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
from qiskit.quantum_info import SparsePauliOp

# ==========================================
# 1. 設定 (N=8)
# ==========================================
num_qubits = 8
# XXZモデルのパラメータ
J = 1.0      # XY平面でのホッピング（スピンの移動）
Delta = 1.0  # Z方向の隣接相互作用
J2 = 0.5     # Z方向の次近接相互作用 (可積分性の破壊＝カオスの源)
evolution_time = 3.0
num_time_steps = 2000
dt = evolution_time / num_time_steps

print(f"=== [Data Generation] {num_qubits} Qubit カオス的XXZ 厳密時間発展 ===")

# ==========================================
# 2. ハミルトニアン構築 (疎行列化)
# ==========================================
pauli_list = []

# (A) NN: XX + YY 項 (隣接スピンのホッピング)
for i in range(num_qubits - 1):
    px = ['I'] * num_qubits; px[i] = 'X'; px[i+1] = 'X'
    py = ['I'] * num_qubits; py[i] = 'Y'; py[i+1] = 'Y'
    pauli_list.append(("".join(px)[::-1], J))
    pauli_list.append(("".join(py)[::-1], J))

# (B) NN: ZZ 項 (隣接スピン間の相互作用)
for i in range(num_qubits - 1):
    pz = ['I'] * num_qubits; pz[i] = 'Z'; pz[i+1] = 'Z'
    pauli_list.append(("".join(pz)[::-1], Delta))

# (C) NNN: ZZ 項 (次近接スピン間の相互作用 -> カオス化)
for i in range(num_qubits - 2):
    p_nnn = ['I'] * num_qubits; p_nnn[i] = 'Z'; p_nnn[i+2] = 'Z'
    pauli_list.append(("".join(p_nnn)[::-1], J2))

H_sparse = SparsePauliOp.from_list(pauli_list).to_matrix(sparse=True)


# ==========================================
# 3. 宇宙の初期化
# ==========================================
np.random.seed(42)
state = (np.random.rand(2**num_qubits) - 0.5) + 1j * (np.random.rand(2**num_qubits) - 0.5)
state /= la.norm(state)

# ==========================================
# 4. 観測辞書の構築
# ==========================================
print("\n--- 観測辞書の構築中 ---")
# 【辞書A】マクロな観測 (15次元)
dict_A_paulis = []
for i in range(num_qubits):
    p = ['I'] * num_qubits; p[i] = 'Z'; dict_A_paulis.append("".join(p)[::-1])
for i in range(num_qubits - 1):
    p = ['I'] * num_qubits; p[i] = 'Z'; p[i+1] = 'Z'; dict_A_paulis.append("".join(p)[::-1])
ops_A_sparse = [SparsePauliOp(p).to_matrix(sparse=True) for p in dict_A_paulis]

# 【辞書B】完全辞書 (65,535次元)
dict_B_paulis = []
for p_tuple in itertools.product(['I', 'X', 'Y', 'Z'], repeat=num_qubits):
    p_str = "".join(p_tuple)
    if p_str != 'I' * num_qubits:
        dict_B_paulis.append(p_str[::-1])
ops_B_sparse = [SparsePauliOp(p).to_matrix(sparse=True) for p in dict_B_paulis]

print(f"辞書A: {len(ops_A_sparse)} 次元, 辞書B: {len(ops_B_sparse)} 次元")

data_X_A  = np.zeros((len(ops_A_sparse), num_time_steps))
data_dX_A = np.zeros((len(ops_A_sparse), num_time_steps))
data_X_B  = np.zeros((len(ops_B_sparse), num_time_steps))
data_dX_B = np.zeros((len(ops_B_sparse), num_time_steps))

# ==========================================
# 5. オンザフライ時間発展と超高速微分取得
# ==========================================
print("\n--- 厳密時間発展と微分値取得を実行中 ---")
start_time = time.time()

for t_idx in range(num_time_steps):
    phi = H_sparse.dot(state)

    for k, op in enumerate(ops_A_sparse):
        O_psi = op.dot(state)
        data_X_A[k, t_idx] = np.real(state.conj().T @ O_psi)
        data_dX_A[k, t_idx] = -2.0 * np.imag(phi.conj().T @ O_psi)

    for k, op in enumerate(ops_B_sparse):
        O_psi = op.dot(state)
        data_X_B[k, t_idx] = np.real(state.conj().T @ O_psi)
        data_dX_B[k, t_idx] = -2.0 * np.imag(phi.conj().T @ O_psi)

    if (t_idx + 1) % 200 == 0:
        print(f"Step {t_idx + 1}/{num_time_steps} 完了... (経過: {time.time() - start_time:.1f}s)")

    if t_idx < num_time_steps - 1:
        state = spla.expm_multiply(-1j * H_sparse * dt, state)

print(f"時間発展完了: {time.time() - start_time:.2f} 秒")

# ==========================================
# 6. データの保存 (numpyバイナリ形式)
# ==========================================
print("\n--- データをファイルに保存中 ---")
np.save("q8_data_X_A.npy", data_X_A)
np.save("q8_data_dX_A.npy", data_dX_A)
np.save("q8_data_X_B.npy", data_X_B)
np.save("q8_data_dX_B.npy", data_dX_B)
print("保存完了。次の解析スクリプトを実行してください。")
