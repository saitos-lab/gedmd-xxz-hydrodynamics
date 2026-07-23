import sys
import time
import argparse
import numpy as np
import scipy.sparse.linalg as spla
import scipy.linalg as la
from qiskit.quantum_info import SparsePauliOp

# ==========================================
# 0. コマンドライン引数の設定
# ==========================================
parser = argparse.ArgumentParser(description="Generate 20Qubit Spin Chain Data (Ensemble)")
parser.add_argument("--seed", type=int, default=0, help="Random seed for the sample")
args = parser.parse_args()
seed = args.seed

# ==========================================
# 1. 設定
# ==========================================
num_qubits = 20  
J = 1.0; Delta = 1.0; J2 = 0.5
dt = 0.002; num_steps = 2000

print(f"=== [Data Gen] 20Qubit: Length 3 辞書 (Seed: {seed}) ===")

def make_pauli_str(op_dict, n_qubits):
    p = ['I'] * n_qubits
    for idx, char in op_dict.items(): p[idx] = char
    return "".join(p)[::-1]

# ハミルトニアン
pauli_list = []
for i in range(num_qubits - 1):
    pauli_list.append((make_pauli_str({i:'X', i+1:'X'}, num_qubits), J))
    pauli_list.append((make_pauli_str({i:'Y', i+1:'Y'}, num_qubits), J))
    pauli_list.append((make_pauli_str({i:'Z', i+1:'Z'}, num_qubits), Delta))
for i in range(num_qubits - 2):
    pauli_list.append((make_pauli_str({i:'Z', i+2:'Z'}, num_qubits), J2))
H_sparse = SparsePauliOp.from_list(pauli_list).to_matrix(sparse=True)

# ==========================================
# 2. 観測辞書構築 (拡張版: 計149次元)
# ==========================================
ops_Z = [SparsePauliOp(make_pauli_str({i: 'Z'}, num_qubits)).to_matrix(sparse=True) for i in range(20)]
ops_J = [SparsePauliOp.from_list([
            (make_pauli_str({i:'X', i+1:'Y'}, num_qubits), 1.0),
            (make_pauli_str({i:'Y', i+1:'X'}, num_qubits), -1.0)
         ]).to_matrix(sparse=True) for i in range(19)]

ops_ZZ = [SparsePauliOp(make_pauli_str({i: 'Z', i+1: 'Z'}, num_qubits)).to_matrix(sparse=True) for i in range(19)]
ops_K = [SparsePauliOp.from_list([
            (make_pauli_str({i:'X', i+1:'Y'}, num_qubits), 1.0),
            (make_pauli_str({i:'Y', i+1:'X'}, num_qubits), 1.0)
         ]).to_matrix(sparse=True) for i in range(19)]

ops_bath3 = []
for i in range(num_qubits - 2):
    ops_bath3.append(SparsePauliOp(make_pauli_str({i:'Z', i+1:'Z', i+2:'Z'}, num_qubits)).to_matrix(sparse=True))
    ops_bath3.append(SparsePauliOp.from_list([
        (make_pauli_str({i:'Z', i+1:'X', i+2:'Y'}, num_qubits), 1.0),
        (make_pauli_str({i:'Z', i+1:'Y', i+2:'X'}, num_qubits), -1.0)
    ]).to_matrix(sparse=True))
    ops_bath3.append(SparsePauliOp.from_list([
        (make_pauli_str({i:'X', i+1:'Y', i+2:'Z'}, num_qubits), 1.0),
        (make_pauli_str({i:'Y', i+1:'X', i+2:'Z'}, num_qubits), -1.0)
    ]).to_matrix(sparse=True))
    ops_bath3.append(SparsePauliOp.from_list([
        (make_pauli_str({i:'X', i+1:'Z', i+2:'Y'}, num_qubits), 1.0),
        (make_pauli_str({i:'Y', i+1:'Z', i+2:'X'}, num_qubits), -1.0)
    ]).to_matrix(sparse=True))

ops_sparse = ops_Z + ops_J + ops_ZZ + ops_K + ops_bath3
print(f"辞書構築完了: 全{len(ops_sparse)}次元")

# ==========================================
# 3. 宇宙の初期化 (指定されたシード値による無限大温度状態)
# ==========================================
np.random.seed(seed)
state = (np.random.rand(2**num_qubits) - 0.5) + 1j * (np.random.rand(2**num_qubits) - 0.5)
state /= la.norm(state)

# ==========================================
# 4. 時間発展とデータ記録
# ==========================================
X_data = np.zeros((len(ops_sparse), num_steps))
dX_data = np.zeros((len(ops_sparse), num_steps))

print(f"--- 時間発展を開始 (Seed: {seed}) ---")
start_time = time.time()
for t in range(num_steps):
    phi = H_sparse.dot(state)
    for k, op in enumerate(ops_sparse):
        O_psi = op.dot(state)
        X_data[k, t] = np.real(state.conj().T @ O_psi)
        dX_data[k, t] = -2.0 * np.imag(phi.conj().T @ O_psi)
        
    if (t + 1) % 200 == 0: 
        print(f"  Step {t + 1}/{num_steps}")
        
    if t < num_steps - 1: 
        state = spla.expm_multiply(-1j * H_sparse * dt, state)

# 出力ファイル名にシード値を埋め込む
out_filename = f"gedmd_current_len3_sample_{seed:02d}.npz"
np.savez_compressed(out_filename, X_data=X_data, dX_data=dX_data)
print(f"=> データを '{out_filename}' に保存しました。(経過時間: {time.time() - start_time:.1f}s)")
