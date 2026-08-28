# ======================================================================
# 比較グラフ生成: 保存した粗視化データを読み込み、1つの図に重ねてプロット
# ======================================================================
import numpy as np
import matplotlib.pyplot as plt
import os

print("=== [Plotting] 粗視化ステップ比較グラフの生成 ===")

# プロットする設定とラベル、色の定義
cg_settings = ['exact', 2, 5, 10, 20, 50]
dt = 0.002

labels = [
    'Exact Derivative',
    f'$\\Delta t_{{\\rm cg}}$={2*dt:.3f}',
    f'$\\Delta t_{{\\rm cg}}$={5*dt:.3f}',
    f'$\\Delta t_{{\\rm cg}}$={10*dt:.3f}',
    f'$\\Delta t_{{\\rm cg}}$={20*dt:.3f}',
    f'$\\Delta t_{{\\rm cg}}$={50*dt:.3f}'
]
colors = ['black', 'blue', 'dodgerblue', 'green', 'orange', 'red']
linestyles = ['--', '-', '-', '-', '-', '-']
linewidths = [1.5, 1.5, 1.5, 2.0, 2.0, 2.5]

# 5段構成のグラフを作成
fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
titles = [
    r"Elasticity $c^2$", 
    r"Friction $\gamma$", 
    r"Viscosity $\nu$", 
    r"Macroscopic Diffusion $D = c^2/\gamma$", 
    r"Direct $D_Z$ (Should be 0)"
]

# 各設定のデータを読み込んでプロット
for idx, cg in enumerate(cg_settings):
    filename = f"cg_data_{cg}.npz"
    
    if not os.path.exists(filename):
        print(f"警告: {filename} が見つかりません。スキップします。")
        continue
        
    data = np.load(filename)
    t = data['time']
    
    # 軸1: c^2
    axes[0].plot(t, data['c2'], color=colors[idx], linestyle=linestyles[idx], 
                 linewidth=linewidths[idx], label=labels[idx])
    # 軸2: gamma
    axes[1].plot(t, data['gamma'], color=colors[idx], linestyle=linestyles[idx], 
                 linewidth=linewidths[idx], label=labels[idx])
    # 軸3: nu
    axes[2].plot(t, data['nu'], color=colors[idx], linestyle=linestyles[idx], 
                 linewidth=linewidths[idx], label=labels[idx])
    # 軸4: D
    axes[3].plot(t, data['D'], color=colors[idx], linestyle=linestyles[idx], 
                 linewidth=linewidths[idx], label=labels[idx])
    # 軸5: Dz
    axes[4].plot(t, data['Dz'], color=colors[idx], linestyle=linestyles[idx], 
                 linewidth=linewidths[idx], label=labels[idx])

# 各パネルの装飾
for i, ax in enumerate(axes):
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel(titles[i], fontsize=12)
    
    # Dのグラフ(index=3)とDz(index=4)は値が飛びやすいためsymlogスケールを適用
    if i == 0:
        ax.set_ylim([-1, 5])
    elif i == 1:
        ax.set_ylim([-1, 2])
    elif i == 2:
        ax.set_ylim([-0.5, 1])    
    elif i == 3:
        ax.set_yscale('symlog', linthresh=1.0)
        ax.set_ylim([-10000, 10000])
    elif  i == 4:
        ax.set_ylim([-0.1, 0.5])
    else:
        ax.set_ylim([-2, 5])

axes[0].legend(loc='upper right', bbox_to_anchor=(1.0, 1.5), ncol=3, fontsize=10)
axes[-1].set_xlabel("Time", fontsize=14)
fig.suptitle("Comparison of Hydrodynamic Coefficients by Time Coarse-Graining", fontsize=16, y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96]) # タイトルと凡例のスペース調整
out_fig = "Fig_CoarseGraining_Comparison.png"
plt.savefig(out_fig, dpi=300)
print(f"\n=> グラフを '{out_fig}' として保存しました。")
plt.show()