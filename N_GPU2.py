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
    n = len(n_list)
    n_states = 1 << n
    indices = torch.arange(n_states, device=device)
    diff = torch.zeros(n_states, dtype=torch.float32, device=device)

    for bit, value in enumerate(n_list):
        spin = 2.0 * ((indices >> bit) & 1).to(torch.float32) - 1.0
        diff += value * spin

    raw_energy = diff * diff

    # エネルギーが大きすぎると位相が激しく回り、有限 dt で追いにくくなる。
    # 正規化しても最小解の位置は変わらない。
    energy_scale = float(sum(abs(v) for v in n_list))
    h_problem = raw_energy / energy_scale
    return h_problem, raw_energy


def decode_state(state_index, n_list):
    group_a = []
    group_b = []
    for bit, value in enumerate(n_list):
        if (state_index >> bit) & 1:
            group_a.append(value)
        else:
            group_b.append(value)

    sum_a = sum(group_a)
    sum_b = sum(group_b)
    return group_a, group_b, sum_a, sum_b, abs(sum_a - sum_b)


def apply_real_time_x_rotation(state, bit, theta):
    """exp(+i theta X_i) を適用する。これは H_x = -B sum_i X_i に対応する。"""
    stride = 1 << bit
    block = stride << 1
    view = state.view(-1, block)

    a_old = view[:, :stride].clone()
    b_old = view[:, stride:block].clone()
    c = math.cos(theta)
    s = math.sin(theta)

    view[:, :stride] = c * a_old + 1j * s * b_old
    view[:, stride:block] = c * b_old + 1j * s * a_old


def apply_imaginary_time_x_step(state, bit, theta):
    """exp(+theta X_i) を適用する。基底状態探索用の虚時間発展。"""
    stride = 1 << bit
    block = stride << 1
    view = state.view(-1, block)

    a_old = view[:, :stride].clone()
    b_old = view[:, stride:block].clone()
    c = math.cosh(theta)
    s = math.sinh(theta)

    view[:, :stride] = c * a_old + s * b_old
    view[:, stride:block] = c * b_old + s * a_old


def normalize_state(state):
    """MPS では complex の vector_norm が未対応なので、実数の確率和から正規化する。"""
    norm = torch.sqrt(torch.sum(state.real * state.real + state.imag * state.imag))
    return state / norm


def schedule(progress):
    """端点でゆっくり変化する smoothstep スケジュール。"""
    return progress * progress * (3.0 - 2.0 * progress)


def exact_number_partition(n_list):
    """全状態を調べて厳密解を返す。N=22 なら約419万状態なので検算として現実的。"""
    total = sum(n_list)
    best_diff = None
    best_state = 0

    for state in range(1 << len(n_list)):
        sum_a = 0
        for bit, value in enumerate(n_list):
            if (state >> bit) & 1:
                sum_a += value
        diff = abs(total - 2 * sum_a)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_state = state
            if best_diff == 0:
                break

    return best_state, best_diff


def choose_best_from_top(probabilities, raw_energy, top_k):
    """確率上位の中から、本来の目的関数が最小の状態を選ぶ。"""
    top_k = min(top_k, probabilities.size)
    top_indices = np.argpartition(probabilities, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(probabilities[top_indices])[::-1]]
    best_index = int(top_indices[np.argmin(raw_energy[top_indices])])
    return best_index, top_indices


def run_quantum_annealing():
    device = select_device()
    print(f"Computation Device: {device}")

    # 分割対象の自然数。合計が793なので、理論上の最小差分は少なくとも1。
    n_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 75, 79, 83, 89, 97, 101]
    n = len(n_list)
    n_states = 1 << n

    # "real" は物理的なユニタリ時間発展、"imaginary" は解探索が安定しやすい虚時間発展。
    evolution = "imaginary"

    b0 = 5.0
    tau = 80.0
    time_steps = 100000
    top_k = 2000
    dt = tau / time_steps
    force_state_vector_on_cpu = False

    if device.type == "cpu" and not force_state_vector_on_cpu:
        print("-" * 60)
        print("CPU だけで N=22 の状態ベクトル発展を行うと非常に時間がかかります。")
        print("この実行では厳密探索による最適解の検算だけを行います。")
        print("CPU でも量子発展を試す場合は force_state_vector_on_cpu = True にしてください。")
        print("-" * 60)
        exact_state, exact_diff = exact_number_partition(n_list)
        print("【厳密探索による最適解】")
        print_solution(exact_state, None, n_list)
        print(f"厳密最小差分               : {exact_diff}")
        return

    h_problem, raw_energy_tensor = build_problem_hamiltonian(n_list, device)
    state = torch.ones(n_states, dtype=torch.complex64, device=device) / math.sqrt(n_states)

    if device.type == "mps":
        torch.mps.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for step in range(time_steps):
        progress = step / time_steps
        at = schedule(progress)
        bt = b0 * (1.0 - at)

        if evolution == "real":
            phase = torch.exp(-0.5j * at * h_problem * dt)
            state = state * phase

            theta = bt * dt
            for bit in range(n):
                apply_real_time_x_rotation(state, bit, theta)

            state = state * phase
        elif evolution == "imaginary":
            decay = torch.exp(-0.5 * at * h_problem * dt).to(torch.complex64)
            state = state * decay

            theta = bt * dt
            for bit in range(n):
                apply_imaginary_time_x_step(state, bit, theta)

            state = state * decay
            state = normalize_state(state)
        else:
            raise ValueError("evolution must be 'real' or 'imaginary'")

    probabilities_tensor = state.abs() ** 2

    if device.type == "mps":
        torch.mps.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_time = time.perf_counter() - start_time

    probabilities = probabilities_tensor.cpu().numpy()
    probabilities = probabilities / probabilities.sum()
    raw_energy = raw_energy_tensor.cpu().numpy()

    max_probability_state = int(np.argmax(probabilities))
    best_state, top_indices = choose_best_from_top(probabilities, raw_energy, top_k)
    exact_state, exact_diff = exact_number_partition(n_list)

    print("-" * 60)
    print("【シミュレーション・パラメータ】")
    print(f"自然数の個数 N             : {n}")
    print(f"状態数 2^N                 : {n_states}")
    print(f"発展方式                   : {evolution}")
    print(f"横磁場 B0                  : {b0}")
    print(f"アニーリング時間 tau       : {tau}")
    print(f"タイムステップ数           : {time_steps}")
    print(f"dt                         : {dt:.6f}")
    print("-" * 60)
    print("【パフォーマンス】")
    print(f"計算時間                   : {elapsed_time:.3f} 秒")
    print(f"確率和                     : {probabilities.sum():.8f}")
    print("-" * 60)

    print("【確率最大状態】")
    print_solution(max_probability_state, probabilities[max_probability_state], n_list)
    print("-" * 60)

    print(f"【確率上位 {len(top_indices)} 状態から選んだ最低エネルギー状態】")
    print_solution(best_state, probabilities[best_state], n_list)
    print("-" * 60)

    print("【厳密探索による検算】")
    print_solution(exact_state, probabilities[exact_state], n_list)
    print(f"厳密最小差分               : {exact_diff}")
    if best_state == exact_state or decode_state(best_state, n_list)[4] == exact_diff:
        print("判定                       : アニーリング候補は最適解に到達")
    else:
        print("判定                       : アニーリング候補は未到達。tau/TIME/B0 を増やして再試行してください")
    print("-" * 60)

    plot_top_states(probabilities, raw_energy, top_indices, time_steps, evolution)


def print_solution(state_index, probability, n_list):
    group_a, group_b, sum_a, sum_b, diff = decode_state(state_index, n_list)
    print(f"状態インデックス           : {state_index}")
    if probability is not None:
        print(f"観測確率                   : {probability:.8f}")
    print(f"グループA                  : {group_a}")
    print(f"グループA 合計             : {sum_a}")
    print(f"グループB                  : {group_b}")
    print(f"グループB 合計             : {sum_b}")
    print(f"差分                       : {diff}")


def plot_top_states(probabilities, raw_energy, top_indices, time_steps, evolution):
    show_k = min(50, len(top_indices))
    shown = top_indices[:show_k]
    labels = [f"{idx}\nd={int(math.sqrt(raw_energy[idx]))}" for idx in shown]

    plt.figure(figsize=(15, 7))
    plt.bar(range(show_k), probabilities[shown])
    plt.xticks(range(show_k), labels, rotation=90)
    plt.xlabel("State index and partition difference")
    plt.ylabel("Observation probability")
    plt.title(f"Number Partitioning QA ({evolution}, TIME={time_steps})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_quantum_annealing()
