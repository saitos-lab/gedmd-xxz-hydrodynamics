# ======================================================================
# 役割: CSVデータを読み込み、c^2, gamma, nu, D の4パネルグラフを出力する
# ======================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=== [Markovian Plateau] グラフの描画 ===")

# ==========================================
# 1. データの読み込み
# ==========================================
input_csv = "markovian_plateau_data.csv"
try:
    df = pd.read_csv(input_csv)
    print(f"'{input_csv}' を読み込みました。")
except FileNotFoundError:
    print(f"エラー: '{input_csv}' が見つかりません。先に計算スクリプトを実行してください。")
    exit()

delta_t_list = df['delta_t_cg'].values
mean_c2_list = df['c2'].values
mean_gamma_list = df['gamma'].values
mean_nu_list = df['nu'].values
D_list = df['D'].values

# ==========================================
# 2. グラフの描画 (4パネル)
# ==========================================
fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
fig.suptitle(r"Emergence of Markovian Plateau vs Coarse-Graining Time ($\Delta t_{cg}$)", fontsize=16, y=0.95)

# --- パネル1: c^2 (弾性係数) ---
axes[0].plot(delta_t_list, mean_c2_list, 'o-', color='teal', markersize=5, linewidth=2)
axes[0].set_title(r"Pressure Gradient $c^2$ (Elasticity)", fontsize=13)
axes[0].set_ylabel(r"Time-Averaged $c^2$", fontsize=12)
axes[0].grid(True, alpha=0.4)

# --- パネル2: gamma (局所摩擦) ---
axes[1].plot(delta_t_list, mean_gamma_list, 'o-', color='purple', markersize=5, linewidth=2)
axes[1].set_title(r"Local Friction $\gamma$ (Irreversible Dissipation)", fontsize=13)
axes[1].set_ylabel(r"Time-Averaged $\gamma$", fontsize=12)
axes[1].grid(True, alpha=0.4)
axes[1].axhline(0, color='black', linestyle='--')

# --- パネル3: nu (動粘性率 / 運動量拡散) ---
axes[2].plot(delta_t_list, mean_nu_list, 'o-', color='forestgreen', markersize=5, linewidth=2)
axes[2].set_title(r"Kinematic Viscosity $\nu$ (Momentum Diffusion)", fontsize=13)
axes[2].set_ylabel(r"Time-Averaged $\nu$", fontsize=12)
axes[2].grid(True, alpha=0.4)
axes[2].axhline(0, color='black', linestyle='--')

# --- パネル4: D (マクロな拡散係数) ---
axes[3].plot(delta_t_list, D_list, 'o-', color='coral', markersize=5, linewidth=2)
axes[3].set_title(r"Macroscopic Diffusion $D = c^2 / \gamma$", fontsize=13)
axes[3].set_ylabel(r"Time-Averaged $D$", fontsize=12)
axes[3].set_xlabel(r"Coarse-Graining Time $\Delta t_{cg}$ [s]", fontsize=14)
axes[3].grid(True, alpha=0.4)
axes[3].set_ylim([0, 10]) # Dが極端に発散する箇所を見やすくするため制限

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_fig = "Fig_Markovian_Plateau.png"
plt.savefig(out_fig, dpi=300)
print(f"\n=> グラフを '{out_fig}' として保存しました。")
plt.show()