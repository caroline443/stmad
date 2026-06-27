"""
动态阈值异常检测

支持两种阈值算法：
  1. pot（默认）：Peaks Over Threshold（极值理论，Siffer et al. KDD 2017）
     用广义帕累托分布（GPD）拟合残差尾部，理论上最优的无监督阈值
  2. robust（备用）：鲁棒正态拟合 + 目标预测率校准

POT 参考：
  Siffer A, Fouque PA, Termier A, Ménier C.
  "Anomaly Detection in Streams with Extreme Value Theory."
  KDD 2017. https://dl.acm.org/doi/10.1145/3097983.3098052
"""

import numpy as np
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  POT：极值理论阈值（主推方法）
# ─────────────────────────────────────────────────────────────────────────────

def _pot_threshold(
    r:       np.ndarray,
    q0_pct:  float = 0.98,  # 初始截断分位数（取第 98 百分位为初始阈 u）
    alpha:   float = 1e-3,  # 目标超阈率 P(r > ε*)，越小阈值越高越保守
) -> float:
    """
    POT（Peaks Over Threshold）阈值估计（Siffer et al., KDD 2017）。

    原理（EVT 极值理论）：
      对任意连续分布 F，超过高分位 u 的超量 Y = r - u | r > u
      渐近服从广义帕累托分布（GPD）：
        P(Y > y) ≈ (1 + γy/σ)^{-1/γ}
      利用此近似反推使 P(r > ε*) = alpha 的阈值 ε*。

    优点：
      - 不假设正态/任何分布，理论保证来自极值定理
      - alpha 直接控制误报率，可解释性强
      - 自动适配不同模型残差量级（PSTG vs Mamba 残差不同 → u 不同 → ε* 自适应）
    """
    try:
        from scipy.stats import genpareto
    except ImportError:
        return float(np.quantile(r, 1.0 - alpha))

    n  = len(r)
    u  = float(np.quantile(r, q0_pct))
    excesses = r[r > u] - u
    Nt = len(excesses)

    if Nt < 20:
        return float(np.quantile(r, 1.0 - alpha))

    try:
        gamma, _, sigma = genpareto.fit(excesses, floc=0)

        # 反推阈值：(Nt/n)·P(Y > e) = alpha  =>  P(Y > e) = alpha·n/Nt
        p_exceed = alpha * n / Nt

        if abs(gamma) < 1e-8:
            # 指数极限（γ→0）：P(Y>e) = exp(-e/σ)
            excess_t = -sigma * np.log(max(p_exceed, 1e-300))
        else:
            # 一般情况：P(Y>e) = (1+γe/σ)^{-1/γ}
            excess_t = (sigma / gamma) * (p_exceed ** (-gamma) - 1)

        eps = u + float(excess_t)

        # 合理性检验
        if not np.isfinite(eps) or eps < u or eps > float(np.max(r)) * 3:
            eps = float(np.quantile(r, 1.0 - alpha))

        return eps

    except Exception:
        return float(np.quantile(r, 1.0 - alpha))


# ─────────────────────────────────────────────────────────────────────────────
#  Robust：鲁棒正态拟合（备用）
# ─────────────────────────────────────────────────────────────────────────────

def _robust_threshold(r: np.ndarray) -> float:
    """鲁棒正态拟合 + 目标预测率校准（旧版方法）。
    _find_optimal_threshold 是本函数的向后兼容别名。"""
    T = len(r)
    if T < 10:
        return float(np.max(r))

    r_sorted = np.sort(r)
    half     = max(T // 2, 5)
    mu_n     = float(np.mean(r_sorted[:half]))
    sigma_n  = max(float(np.std(r_sorted[:half])), 1e-9)

    eps_0    = mu_n + 3.0 * sigma_n
    est_rate = max(float(np.mean(r > eps_0)), 1e-5)
    target   = min(est_rate * 2.0, 0.05)

    eps_pct  = float(np.percentile(r, 100.0 * (1.0 - target)))
    return max(eps_pct, eps_0)


# 向后兼容别名（evaluate_ma.py 等旧脚本使用）
_find_optimal_threshold = _robust_threshold


# ─────────────────────────────────────────────────────────────────────────────
#  辅助
# ─────────────────────────────────────────────────────────────────────────────

def smooth_residuals(r: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return r.astype(np.float32)
    kernel = np.ones(window) / window
    return np.convolve(r, kernel, mode="same").astype(np.float32)


def _count_sequences(binary: np.ndarray) -> int:
    count, in_seq = 0, False
    for v in binary:
        if v and not in_seq:   count += 1; in_seq = True
        elif not v:            in_seq = False
    return count


def _extract_sequences_with_pos(
    r: np.ndarray, eps: float
) -> List[Tuple[int, int, np.ndarray]]:
    seqs, in_seq, start = [], False, 0
    mask = r >= eps
    for i, v in enumerate(mask):
        if v and not in_seq:    start = i; in_seq = True
        elif not v and in_seq:  seqs.append((start, i - 1, r[start:i])); in_seq = False
    if in_seq:
        seqs.append((start, len(r) - 1, r[start:]))
    return seqs


def _prune_false_positives(
    seqs:       List[Tuple],
    mu_r:       float,
    sigma_r:    float,
    min_peak_z: float = 1.0,
) -> List[Tuple]:
    """
    按序列峰值绝对强度剪枝（保留峰值 > μ + z×σ 的序列）。

    比原版相邻峰值差剪枝的改进：
      不会因多个相似幅度的真实事件互相剪掉（21% 规则的缺陷）。
    """
    threshold_peak = mu_r + min_peak_z * sigma_r
    return [(s, e, seq) for (s, e, seq) in seqs if np.max(seq) >= threshold_peak]


def _compute_anomaly_scores(
    r:          np.ndarray,
    eps:        float,
    mu_r:       float,
    sigma_r:    float,
    min_peak_z: float = 1.0,
) -> np.ndarray:
    """连续异常分数（公式 27）：s = (peak - ε*) / (μ + σ)"""
    scores = np.zeros(len(r), dtype=np.float32)
    seqs   = _extract_sequences_with_pos(r, eps)
    if not seqs:
        return scores

    seqs  = _prune_false_positives(seqs, mu_r, sigma_r, min_peak_z)
    denom = mu_r + sigma_r + 1e-9
    for s_start, s_end, seq in seqs:
        scores[s_start : s_end + 1] = max(0.0, float(np.max(seq) - eps) / denom)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
#  对预计算残差直接应用阈值（供 evaluate_ma.py 等使用）
# ─────────────────────────────────────────────────────────────────────────────

def threshold_signal(
    r:          np.ndarray,   # 已平滑的一维残差信号
    method:     str   = "pot",
    pot_q0:     float = 0.98,
    pot_alpha:  float = 4e-3,
    min_peak_z: float = 1.5,
) -> np.ndarray:
    """
    对预计算的残差信号 r [T] 应用阈值，返回异常分数 [T]。

    与 detect_anomalies 的区别：
      - 输入直接是残差（不做平滑），适合 evaluate_ma 等已自行构造信号的场合
    """
    r = r.astype(np.float64)
    mu_r    = float(np.mean(r))
    sigma_r = float(np.std(r))

    if method == "pot":
        eps = _pot_threshold(r, q0_pct=pot_q0, alpha=pot_alpha)
    else:
        eps = _robust_threshold(r)

    scores = _compute_anomaly_scores(r, eps, mu_r, sigma_r, min_peak_z)
    n_seq  = _count_sequences(scores > 0)
    print(f"  [{method}] ε* = {eps:.5f}  "
          f"初始pred_rate = {(r >= eps).mean()*100:.3f}%")
    print(f"  剪枝后序列数 = {n_seq}  "
          f"最终pred_rate = {(scores > 0).mean()*100:.3f}%")
    return scores.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  完整 Φ 算子（对外接口，兼容旧参数签名）
# ─────────────────────────────────────────────────────────────────────────────

def detect_anomalies(
    x_true:        np.ndarray,   # [T, C]
    x_pred:        np.ndarray,   # [T, C]
    smooth_window: int   = 105,
    p_tfi:         float = 0.21,  # 保留（兼容旧调用，不再使用）
    n_candidates:  int   = 300,   # 保留（兼容旧调用，不再使用）
    min_peak_z:    float = 1.5,
    method:        str   = "pot",
    pot_q0:        float = 0.98,
    pot_alpha:     float = 4e-3,
) -> np.ndarray:
    """
    Φ 算子：输入真值和预测，输出连续异常分数 S ∈ R^T。

    默认使用 POT 阈值（Siffer et al., KDD 2017），
    设 method="robust" 切回旧的鲁棒正态拟合方法。
    """
    r_raw    = np.abs(x_true - x_pred).max(axis=1).astype(np.float64)
    r_smooth = smooth_residuals(r_raw, smooth_window).astype(np.float64)

    mu_r    = float(np.mean(r_smooth))
    sigma_r = float(np.std(r_smooth))

    if method == "pot":
        eps = _pot_threshold(r_smooth, q0_pct=pot_q0, alpha=pot_alpha)
    else:
        eps = _robust_threshold(r_smooth)

    scores = _compute_anomaly_scores(r_smooth, eps, mu_r, sigma_r, min_peak_z)

    n_seq = _count_sequences(scores > 0)
    print(f"  [{method}] ε* = {eps:.5f}  "
          f"初始pred_rate = {(r_smooth >= eps).mean()*100:.3f}%")
    print(f"  剪枝后序列数 = {n_seq}  "
          f"最终pred_rate = {(scores > 0).mean()*100:.3f}%")

    return scores.astype(np.float32)
