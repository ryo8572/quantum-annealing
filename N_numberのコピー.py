import numpy as np
from matplotlib import pyplot as plt

from change_shinsu import id


# 分割対象となる自然数
n = [2, 3,5]

# 自然数の数
N = len(n)

# 状態数
N2 = 2**N

# ハミルトニアン
H = np.zeros([N2, N2])

# ハミルトニアンの対角成分
H_x = np.zeros([N2])

# 非対角成分で反転する位置を保存する配列
Io = np.zeros([N2, N2], int)

# スピン系の初期状態 φ0o
f0 = np.ones([N2, 1], complex)
f0 = f0 / np.sqrt(N2)

# φ1
f1 = np.zeros([N2, 1], complex)

# 横磁場の大きさ
B0 = 1.0

# 時間発展関係の変数
tau = 1.0
TIME = 100
dt = tau / TIME

#
T = np.zeros([N2, N2], complex)


# ハミルトニアンの対角成分を求める
for L in range(N2):
    for i in range(N):
        for j in range(i+1, N):
            H_x[L] += (2*id(L, i, N) - 1) * n[i] * (2*id(L, j, N) - 1) * n[j]
print(H_x)


# 非対角成分で反転する位置を保存
for i in range(N2):
    for j in range(N2):
        k = 0
        for l in range(N):
            k += (2*id(i, l, N) - 1) * (2*id(j, l, N) - 1)
        if (k == N-2):
            Io[i][j] = 1
print(Io)


# 時間発展
for time in range(TIME):
    t = time * dt
    At = t / tau
    Bt = B0 * (1.0 - At)

    # ハミルトニアンの非対角成分の完成
    for i in range(N2):
        for j in range(N2):
            H[i][j] = -Bt * Io[i][j]

            T[i][j] = complex(0.0, -0.5*dt*H[i][j])

    # ハミルトニアンの対角成分の完成
    for L in range(N2):
        H[L][L] = At * H_x[L]

        T[L][L] = complex(1.0, -0.5*dt*H[L][L])

    f1 = T @ f0
    f0 = f1

# 正規化
"""
abs_f1 = 0
for L in range(N2):
    abs_f1 = (np.abs(f1[L]))**2
abs_f1 = np.sqrt(abs_f1)
f1 = f1 / abs_f1
"""
   
# 正規化（別解）
f1 = f1 / np.linalg.norm(f1, ord=2)

# 重み計算
a = np.zeros([N2, 1])
for L in range(N2):
    a[L] = (np.abs(f1[L])) ** 2

print(a)
print(sum(a))

x = np.arange(N2)

plt.bar(x, a.flatten()) # Use x as x-values and flatten 'a' for heights
plt.show()