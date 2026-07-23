import sys
import argparse
import numpy as np
import scipy.linalg as la

# ==========================================
# 1. 真の gEDMD 解析関数 (gesvdを明示)
# ==========================================
def perform_true_gedmd(X, dX, rank_tol=1e-3):
    U, S, Vh = la.svd(X, full_matrices=False)
    #U, S, Vh = la.svd(X, full_matrices=False, lapack_driver='gesvd')
    r = np.sum(S > rank_tol)
    if r == 0: return np.array([])
    Ur = U[:, :r]; Sr = S[:r]; Vhr = Vh[:r, :]
    L_matrix = dX @ Vhr.conj().T @ np.diag(1/Sr) @ Ur.conj().T
    return la.eigvals(L_matrix)

# ==========================================
# メイン処理 (コマンドライン実行)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate gEDMD for a single sliding window.")
    parser.add_argument("window_idx", type=int, help="Index of the window to process (0, 1, 2, ...)")
    args = parser.parse_args()
    i = args.window_idx

    print(f"--- ジョブ開始: Window Index {i} ---")

    # ==========================================
    # 2. データの読み込み (mmap_mode='r' でメモリ節約)
    # ==========================================
    # ※並列実行時にメモリを浪費しないよう、ディスク上のデータを直接参照します
    X_A  = np.load("q8_data_X_A.npy", mmap_mode='r')
    dX_A = np.load("q8_data_dX_A.npy", mmap_mode='r')
    X_B  = np.load("q8_data_X_B.npy", mmap_mode='r')
    dX_B = np.load("q8_data_dX_B.npy", mmap_mode='r')

    num_time_steps = X_A.shape[1]
    evolution_time = 3.0
    dt = evolution_time / num_time_steps

    window_size = 300
    stride = 50
    num_windows = (num_time_steps - window_size) // stride + 1

    if i < 0 or i >= num_windows:
        print(f"エラー: window_idx は 0 から {num_windows-1} の間で指定してください。")
        sys.exit(1)

    # ==========================================
    # 3. 指定された窓のインデックス計算
    # ==========================================
    start_idx = i * stride
    end_idx = start_idx + window_size
    t_center = (start_idx + window_size / 2) * dt

    print(f"  Time Center : t={t_center:.3f} (Step {start_idx} to {end_idx})")

    # ==========================================
    # 4. gEDMD 解析の実行 (スライスしてメモリに載せる)
    # ==========================================
    print("  -> 辞書Aの解析中...")
    eigs_A = perform_true_gedmd(np.array(X_A[:, start_idx:end_idx]), 
                                np.array(dX_A[:, start_idx:end_idx]))
    
    print("  -> 辞書Bの解析中...")
    eigs_B = perform_true_gedmd(np.array(X_B[:, start_idx:end_idx]), 
                                np.array(dX_B[:, start_idx:end_idx]))

    # ==========================================
    # 5. 結果の保存 (npz形式)
    # ==========================================
    output_filename = f"result_window_{i:03d}.npz"
    np.savez(output_filename, 
             window_idx=i, 
             t_center=t_center, 
             eigs_A=eigs_A, 
             eigs_B=eigs_B)
    
    print(f"--- 完了: 結果を {output_filename} に保存しました ---")
