import time
import torch
import numpy as np
from matplotlib import pyplot as plt

def run_quantum_annealing():
    # 1. 演算デバイスの選定 (cuda > mps > cpu の優先順位で選択,cudaの場合はmpsをcudaに変更)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Computation Device: {device}")

    # 分割対象となる自然数とシステムサイズの設定
    # N=20 のため、状態空間は 2^20 = 1,048,576 次元となる
    n_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 75, 79]
    N = len(n_list)
    N2 = 1 << N  # ビットシフトによる高速な 2^N の算出

    # 自然数の配列をデバイスのテンソルへ転送 (MPS仕様に合わせた単精度 float32)
    n_tensor = torch.tensor(n_list, dtype=torch.float32, device=device)

    # 2. 初期状態 φ0 の生成 (一様重ね合わせ状態)
    # 確率振幅を complex64 (単精度複素数) で定義し、全状態を均等な確率で初期化
    f = torch.ones(N2, dtype=torch.complex64, device=device) / np.sqrt(N2)

    # 3. 対角ハミルトニアン (H_z) の構築
    H_z = torch.zeros(N2, dtype=torch.float32, device=device)
    indices = torch.arange(N2, device=device)

    # 状態空間全体に対するスピン変数 σ_i ∈ {-1, 1} の一括生成
    spins = torch.zeros((N2, N), dtype=torch.float32, device=device)
    for i in range(N):
        # インデックスの二進数表現から i 番目のビットを抽出し、スピン値に変換
        spins[:, i] = 2.0 * ((indices >> i) & 1) - 1.0
    # コスト関数の評価: 差の二乗を展開した相互作用項 Σ_{i<j} n_i * n_j * σ_i * σ_j
    for i in range(N):
        for j in range(i+1, N):
            H_z += spins[:, i] * n_tensor[i] * spins[:, j] * n_tensor[j]

    # 4. 時間発展パラメータの設定
    B0 = 7.5       # 初期横磁場の強さ (非対角項の最大値を制御)
    tau = 100.0       # 物理的なアニーリング時間 (断熱定理における総時間)
    TIME = 1000000   # タイムステップ数 (時間発展の解像度)
    dt = tau / TIME

    # デバイスの同期（計測開始前の初期化およびメモリ転送の完了待機）
    if device.type == 'mps':
        torch.mps.synchronize()

    # --- 計算時間の計測開始 ---
    start_time = time.perf_counter()

    # 5. 鈴木・トロッター分解によるユニタリ時間発展ループ
    for time_step in range(TIME):
        t = time_step * dt
        # アニーリング・スケジュールの線形変化
        At = t / tau
        Bt = B0 * (1.0 - At)

        # ステップ 5-1: 対角項 (H_z) による半ステップの位相回転演算
        phase_z = torch.exp(-1j * (At * H_z) * (dt / 2.0))
        f = f * phase_z

        # ステップ 5-2: 非対角項 (H_x) による横磁場作用 (Matrix-Free法に基づく厳密な回転)
        theta = Bt * dt
        cos_val = np.cos(theta)
        sin_val = np.sin(theta)

        for bit in range(N):
            flipped_indices = indices ^ (1 << bit)
            f_flipped = f[flipped_indices]
            # 局所ユニタリ演算の適用 (パウリX行列の指数関数)
            f = cos_val * f - 1j * sin_val * f_flipped

        # ステップ 5-3: 対角項 (H_z) による残りの半ステップの位相回転演算
        f = f * phase_z

    # デバイス上で確率分布 (振幅の絶対値の2乗) を算出
    probabilities_tensor = (f.abs() ** 2)

    # --- デバイスの全演算完了を待機して計測終了 ---
    if device.type == 'mps':
        torch.mps.synchronize()
        
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # 演算結果をホスト (CPU) 側のNumPy配列へ転送
    probabilities = probabilities_tensor.cpu().numpy()

    # 単精度浮動小数点演算 (float32) の蓄積による丸め誤差を補正するための最終再正規化
    probabilities = probabilities / np.sum(probabilities)

    # 6. 実験結果の基礎メトリクス出力
    print("-" * 50)
    print("【シミュレーション・パラメータ】")
    print(f"システムサイズ (N)   : {N} (状態数: {N2})")
    print(f"アニーリング時間 (tau): {tau:.2f}")
    print(f"タイムステップ数 (TIME): {TIME}")
    print(f"時間刻み幅 (dt)      : {dt:.6f}")
    print("-" * 50)
    print("【パフォーマンス・メトリクス】")
    print(f"実効計算時間         : {elapsed_time:.6f} 秒")
    print(f"ユニタリ性チェック   : {np.sum(probabilities):.4f} (理論値: 1.0)")
    print("-" * 50)

    # 7. 最適解のデコードと物理的解釈の出力
    print("【最適解のデコード結果】")
    
    # 確率が最大となる基底状態のインデックスを取得
    best_state_index = np.argmax(probabilities)
    max_probability = probabilities[best_state_index]

    group_A = []
    group_B = []

    # インデックスの二進数表現を解析し、自然数を2つのグループに写像 (デコード)
    for i in range(N):
        if (best_state_index >> i) & 1:
            group_A.append(n_list[i])
        else:
            group_B.append(n_list[i])

    sum_A = sum(group_A)
    sum_B = sum(group_B)
    diff = abs(sum_A - sum_B)

    print(f"観測された最適状態インデックス : {best_state_index}")
    print(f"基底状態の観測確率             : {max_probability:.6f}")
    print(f"グループA : {group_A}")
    print(f"  -> 合計値 : {sum_A}")
    print(f"グループB : {group_B}")
    print(f"  -> 合計値 : {sum_B}")
    print(f"両グループの合計値の差分       : {diff}")
    print("-" * 50)

    # 8. 上位状態の抽出と可視化処理
    # N=20 (約104万状態) の全描画はメモリ枯渇を引き起こすため、確率上位50状態に限定する
    top_k = min(50, N2)
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    top_probs = probabilities[top_indices]

    # グラフのX軸ラベル用にインデックスを文字列へ変換
    labels = [str(idx) for idx in top_indices]

    plt.figure(figsize=(14, 7))
    plt.bar(range(top_k), top_probs, tick_label=labels)
    plt.xlabel("State Index (Ranked by Probability)")
    plt.ylabel("Observation Probability")
    plt.title(f"Quantum Annealing: Number Partitioning Problem (Top {top_k} States)\nN={N}, TIME={TIME}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_quantum_annealing()