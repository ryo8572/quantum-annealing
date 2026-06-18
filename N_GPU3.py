import math
import os
import time

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
from matplotlib import pyplot as plt


def select_device():
    """cuda > mps > cpu の順で利用可能なデバイスを選ぶ。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_problem_hamiltonian(n_list, device):
    """自然数分割問題の対角ハミルトニアン H_z = (sum_i n_i sigma_i)^2 を作る。"""
    N = len(n_list)
    N2 = 1 << N
    indices = torch.arange(N2, device=device)
    diff = torch.zeros(N2, dtype=torch.float32, device=device)

    for bit, value in enumerate(n_list):
        spin = 2.0 * ((indices >> bit) & 1).to(torch.float32) - 1.0
        diff += value * spin

    raw_energy = diff * diff

    # エネルギーが大きすぎると位相が激しく回り、有限 dt で追いにくくなる。
    # 正規化しても最小解の位置は変わらない。
    scaling_factor = float(sum(abs(v) for v in n_list))
    H_z = raw_energy / scaling_factor
    return H_z, raw_energy


def decode_state(best_state_index, n_list):
    group_A = []
    group_B = []
    for bit, value in enumerate(n_list):
        if (best_state_index >> bit) & 1:
            group_A.append(value)
        else:
            group_B.append(value)

    sum_A = sum(group_A)
    sum_B = sum(group_B)
    return group_A, group_B, sum_A, sum_B, abs(sum_A - sum_B)


def apply_real_time_x_rotation(f, bit, theta):
    """exp(+i theta X_i) を適用する。これは H_x = -B sum_i X_i に対応する。"""
    stride = 1 << bit
    block = stride << 1
    view = f.view(-1, block)

    a_old = view[:, :stride].clone()
    b_old = view[:, stride:block].clone()
    cos_val = math.cos(theta)
    sin_val = math.sin(theta)

    view[:, :stride] = cos_val * a_old + 1j * sin_val * b_old
    view[:, stride:block] = cos_val * b_old + 1j * sin_val * a_old


def apply_imaginary_time_x_step(f, bit, theta):
    """exp(+theta X_i) を適用する。基底状態探索用の虚時間発展。"""
    stride = 1 << bit
    block = stride << 1
    view = f.view(-1, block)

    a_old = view[:, :stride].clone()
    b_old = view[:, stride:block].clone()
    cosh_val = math.cosh(theta)
    sinh_val = math.sinh(theta)

    view[:, :stride] = cosh_val * a_old + sinh_val * b_old
    view[:, stride:block] = cosh_val * b_old + sinh_val * a_old


def schedule(progress):
    """端点でゆっくり変化する smoothstep スケジュール。"""
    return progress * progress * (3.0 - 2.0 * progress)


def exact_number_partition(n_list):
    """全状態を調べて厳密解を返す。N=22 なら約419万状態なので検算として現実的。"""
    total = sum(n_list)
    best_diff = None
    best_state_index = 0

    for state_index in range(1 << len(n_list)):
        sum_A = 0
        for bit, value in enumerate(n_list):
            if (state_index >> bit) & 1:
                sum_A += value
        diff = abs(total - 2 * sum_A)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_state_index = state_index
            if best_diff == 0:
                break

    return best_state_index, best_diff


def choose_best_from_top(probabilities, raw_energy, top_k):
    """確率上位の中から、本来の目的関数が最小の状態を選ぶ。"""
    top_k = min(top_k, probabilities.size)
    top_indices = np.argpartition(probabilities, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(probabilities[top_indices])[::-1]]
    best_state_index = int(top_indices[np.argmin(raw_energy[top_indices])])
    return best_state_index, top_indices


def run_quantum_annealing():
    device = select_device()
    print(f"Computation Device: {device}")

    # 分割対象の自然数。合計が793なので、理論上の最小差分は少なくとも1。 N=25 で約3355万状態、N=26 で約6700万状態。N=22 なら約419万状態で厳密解も現実的に求まる。
    n_list = [2, 3, 5]
    N = len(n_list)
    N2 = 1 << N

    # "real" は物理的なユニタリ時間発展、"imaginary" は解探索が安定しやすい虚時間発展。
    evolution = "imaginary"

    # N_GPU.py のパラメータに準拠
    B0 = 1.0
    tau = 1.0
    TIME = 5000
    top_k = min(50, N2)
    dt = tau / TIME
    force_state_vector_on_cpu = False

    if device.type == "cpu" and not force_state_vector_on_cpu:
        print("-" * 60)
        print("CPU だけで状態ベクトル発展を行うと非常に時間がかかります。")
        print("この実行では厳密探索による最適解の検算だけを行います。")
        print("CPU でも量子発展を試す場合は force_state_vector_on_cpu = True にしてください。")
        print("-" * 60)
        exact_state, exact_diff = exact_number_partition(n_list)
        print("【厳密探索による最適解】")
        print_solution(exact_state, None, n_list)
        print(f"厳密最小差分               : {exact_diff}")
        return

    H_z, raw_energy_tensor = build_problem_hamiltonian(n_list, device)
    f = torch.ones(N2, dtype=torch.complex64, device=device) / math.sqrt(N2)

    if device.type == "mps":
        torch.mps.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for time_step in range(TIME):
        progress = time_step / TIME
        At = schedule(progress)
        Bt = B0 * (1.0 - At)

        if evolution == "real":
            phase_z = torch.exp(-0.5j * At * H_z * dt)
            f = f * phase_z

            theta = Bt * dt
            for bit in range(N):
                apply_real_time_x_rotation(f, bit, theta)

            f = f * phase_z
            
        elif evolution == "imaginary":
            decay = torch.exp(-0.5 * At * H_z * dt).to(torch.complex64)
            f = f * decay

            theta = Bt * dt
            for bit in range(N):
                apply_imaginary_time_x_step(f, bit, theta)

            f = f * decay
            # 虚時間発展ではノルムが保存されないため再規格化が必要
            # MPSのcomplex型非対応エラーを回避するため、手動でL2ノルムを計算
            norm = torch.sqrt(torch.sum(f.abs() ** 2))
            f = f / norm
            
        else:
            raise ValueError("evolution must be 'real' or 'imaginary'")

    probabilities_tensor = f.abs() ** 2

    if device.type == "mps":
        torch.mps.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_time = time.perf_counter() - start_time

    probabilities = probabilities_tensor.cpu().numpy()
    probabilities = probabilities / probabilities.sum()
    raw_energy = raw_energy_tensor.cpu().numpy()

    best_state_index = int(np.argmax(probabilities))
    best_state_from_top, top_indices = choose_best_from_top(probabilities, raw_energy, top_k)
    exact_state, exact_diff = exact_number_partition(n_list)

    print("-" * 60)
    print("【シミュレーション・パラメータ】")
    print(f"システムサイズ (N)         : {N}")
    print(f"状態数 (N2)                : {N2}")
    print(f"発展方式                   : {evolution}")
    print(f"初期横磁場 B0              : {B0}")
    print(f"アニーリング時間 tau       : {tau}")
    print(f"タイムステップ数 TIME      : {TIME}")
    print(f"時間刻み幅 dt              : {dt:.6f}")
    print("-" * 60)
    print("【パフォーマンス】")
    print(f"実効計算時間               : {elapsed_time:.3f} 秒")
    print(f"確率和                     : {probabilities.sum():.8f}")
    print("-" * 60)

    print("【確率最大状態】")
    print_solution(best_state_index, probabilities[best_state_index], n_list)
    print("-" * 60)

    print(f"【確率上位 {len(top_indices)} 状態から選んだ最低エネルギー状態】")
    print_solution(best_state_from_top, probabilities[best_state_from_top], n_list)
    print("-" * 60)

    print("【厳密探索による検算】")
    print_solution(exact_state, probabilities[exact_state], n_list)
    print(f"厳密最小差分               : {exact_diff}")
    
    if best_state_from_top == exact_state or decode_state(best_state_from_top, n_list)[4] == exact_diff:
        print("判定                       : アニーリング候補は最適解に到達")
    else:
        print("判定                       : アニーリング候補は未到達。tau/TIME/B0 を増やして再試行してください")
    print("-" * 60)

    plot_top_states(probabilities, raw_energy, top_indices, TIME, evolution)


def print_solution(best_state_index, probability, n_list):
    group_A, group_B, sum_A, sum_B, diff = decode_state(best_state_index, n_list)
    print(f"状態インデックス           : {best_state_index}")
    if probability is not None:
        print(f"観測確率                   : {probability:.8f}")
    print(f"グループA                  : {group_A}")
    print(f"グループA 合計             : {sum_A}")
    print(f"グループB                  : {group_B}")
    print(f"グループB 合計             : {sum_B}")
    print(f"差分                       : {diff}")


def plot_top_states(probabilities, raw_energy, top_indices, TIME, evolution):
    show_k = min(50, len(top_indices))
    shown = top_indices[:show_k]
    labels = [f"{idx}\nd={int(math.sqrt(raw_energy[idx]))}" for idx in shown]

    plt.figure(figsize=(15, 7))
    plt.bar(range(show_k), probabilities[shown])
    plt.xticks(range(show_k), labels, rotation=90)
    plt.xlabel("State index and partition difference")
    plt.ylabel("Observation probability")
    plt.title(f"Number Partitioning QA ({evolution}, TIME={TIME})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_quantum_annealing()