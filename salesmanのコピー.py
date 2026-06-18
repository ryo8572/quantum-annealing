import numpy as np
from matplotlib import pyplot as plt

from change_shinsu import id

# 巡る都市の数
d = [(0,2),
     (2,0)]

n = len(d[0])
N = len(d) * len(d[0])

# 状態数
N2 = 2**N

#ペナルティ項の定数
a=10
b=10

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
B0 = 3.0

# 時間発展関係の変数
tau = 1.0
TIME = 1000
dt = tau / TIME

#
T = np.zeros([N2, N2], complex)

# ハミルトニアンの対角成分
for L in range(N2):

    h = 0
    for alpha in range(n):
        for beta in range(n):
            for i in range(n):
                I_beta = (i+1) * n +beta
                if I_beta >= N:
                    I_beta -= N
                h += d[alpha][beta] * id(L,i*n + alpha,N)*id(L,I_beta,N)
    
    #ペナルティ第一項
    p1 = 0
    for alpha in range(n):
        p1_s = 0
        for i in range(n):
            p1_s += id(L,i*n + alpha, N)
        p1_s = p1_s - 1.0
        p1_s = p1_s **2
        p1 += p1_s
    p1 *= a

    #ペナルティ第二項
    p2 = 0
    for i in range(n):
        p2_s = 0
        for alpha in range(n):
            p2_s += id(L,i*n + alpha, N)
        p2_s = p2_s - 1.0
        p2_s = p2_s **2
        p2 += p2_s
    p2 *= b

    H_x[L] = h + p1 + p2
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

# 正規化（別解）
f1 = f1 / np.linalg.norm(f1, ord=2)

# 重み計算
a = np.zeros([N2, 1])
for L in range(N2):
    a[L] = (np.abs(f1[L])) ** 2

print(a)
print(sum(a))
print(max(a.flatten()[0:,]))

x = np.arange(N2)

plt.bar(x, a.flatten()) 
plt.show()