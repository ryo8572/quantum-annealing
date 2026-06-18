import numpy as np
from matplotlib import pyplot as plt

# 分割対象となる自然数
n = [2, 3, 5]
N = len(n)
N2 = 2**N

# ---------------------------------------------------------
# 行列フリー化のための対角成分 H_z の計算
# ---------------------------------------------------------
H_z = np.zeros(N2)
for L in range(N2):
    sum_val = 0
    for i in range(N):
        # ビット演算で i 番目のスピンを取り出す (1: グループA, -1: グループB)
        spin_i = 1.0 if ((L >> i) & 1) else -1.0
        for j in range(i+1, N):
            spin_j = 1.0 if ((L >> j) & 1) else -1.0
            sum_val += spin_i * n[i] * spin_j * n[j]
    H_z[L] = sum_val

# スピン系の状態ベクトルを用意
f0 = np.ones(N2, dtype=complex) / np.sqrt(N2)  # |φ(t-Δt)>
f1 = np.zeros(N2, dtype=complex)               # |φ(t)>
f2 = np.zeros(N2, dtype=complex)               # |φ(t+Δt)>

# 時間発展パラメータ
B0 = 1.0
tau = 1.0
TIME = 1000
dt = tau / TIME
hbar = 1.0  # プランク定数 (hバー)

# dt (時間刻み幅) の出力
print(f"dt (時間刻み幅): {dt}")

# ---------------------------------------------------------
# 行列フリー形式でのハミルトニアン作用関数
# ---------------------------------------------------------
def apply_H(f, At, Bt):
    # 対角項 (コスト関数) の作用
    Hf = At * H_z * f
    # 非対角項 (横磁場) の作用: XOR演算で隣接状態を特定
    for bit in range(N):
        flipped_indices = np.arange(N2) ^ (1 << bit)
        Hf += -Bt * f[flipped_indices]
    return Hf

# 時間発展ループ
for time_step in range(TIME):
    t = time_step * dt
    At = t / tau
    Bt = B0 * (1.0 - At)

    if time_step == 0:
        # 【条件: テイラー展開第2項までを利用】 
        # |φ(Δt)> ≒ |φ(0)> - i (Δt/hbar) H |φ(0)>
        Hf0 = apply_H(f0, At, Bt)
        f1 = f0 - (1j * dt / hbar) * Hf0
        
        # 【条件: 最初の計算終了後に正規化】 
        f1 = f1 / np.linalg.norm(f1, ord=2)
    else:
        # 【条件: 対称微分を用いた計算式】 
        # |φ(t+Δt)> = |φ(t-Δt)> - i (2Δt/hbar) H |φ(t)>
        Hf1 = apply_H(f1, At, Bt)
        f2 = f0 - (2j * dt / hbar) * Hf1
        
        # 状態の更新 (バケツリレー)
        f0 = f1
        f1 = f2

# 最終的な正規化
f1 = f1 / np.linalg.norm(f1, ord=2)

# 各状態の観測確率の算出
a = (np.abs(f1)) ** 2

print("\n最終的な確率分布:")
print(a)
print(f"確率の総和: {sum(a)}")

# ---------------------------------------------------------
# グラフの描画 (日本語化)
# ---------------------------------------------------------
x = np.arange(N2)
plt.figure(figsize=(10, 6))
plt.bar(x, a)

# 日本語ラベルの設定
plt.title(f"自然数分割問題の量子アニーリング (リープフロッグ法 & 行列フリー, N={N})")
plt.xlabel("状態インデックス (バイナリ表現に対応)")
plt.ylabel("観測確率")

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()