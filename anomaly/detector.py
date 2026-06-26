"""
动态阈值异常检测（论文 Section 3.3）

修正版：原 Telemanom 评分函数在长序列（7.69M 点）下存在根本性缺陷：
  score = (Δμ/μ + Δσ/σ) / (|r_a| + |R_seq|²)
  |r_a| 为万级，|R_seq|² 为百级 → 分母被 |r_a| 主导
  → 分数随阈值升高单调递增 → 永远选最保守阈值 → Recall 极低

修正方案：鲁棒正态拟合 + 目标预测率校准
  1. 用下半段数据（不含异常）拟合正常分布的 μ/σ
  2. 初始阈值 = μ_normal + 3σ（几乎覆盖所有正常点）
  3. 以此估计异常率，设目标预测率 = 2× 估计率
  4. 最终阈值 = 对应百分位（让模型捕获更多事件）
  5. 假阳性剪枝改为：按异常序列相对于全局均值的倍数过滤
     而非按相邻序列峰值差距（原方案会把相似幅度的真实事件全部剪掉）
"""

import numpy as np
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  核心：鲁棒阈值估计
# ─────────────────────────────────────────────────────────────────────────────

def _find_optimal_threshold(
    r:            np.ndarray,
    p_tfi:        float = 0.21,   # 保留参数（不再作为主搜索范围）
    n_candidates: int   = 300,
) -> float:
    """
    改进版阈值估计：鲁棒正态拟合 + 目标预测率校准

    Step 1：用下 50% 数据估计正常分布（对异常点完全免疫）
    Step 2：初始阈值 eps_0 = μ_n + 3σ_n（正常数据 P > 0.1%）
    Step 3：eps_0 以上的点比例 = 粗估异常率 r_hat
    Step 4：目标预测率 = 2× r_hat（留余量以覆盖所有事件）
    Step 5：最终阈值 = 对应百分位，且不低于 eps_0

    与原 Telemanom 区别：
      不再用 (Δμ/μ + Δσ/σ)/(|r_a|+|R_seq|²) 打分——
      该式在长序列下单调偏向高阈值，无法找到真实分割点。
    """
    T = len(r)
    if T < 10:
        return float(np.max(r))

    r_sorted = np.sort(r)

    # Step 1：鲁棒正态估计（下 50%）
    half     = max(T // 2, 5)
    mu_n     = float(np.mean(r_sorted[:half]))
    sigma_n  = max(float(np.std(r_sorted[:half])), 1e-9)

    # Step 2：初始阈值（正常分布的 3σ 以上）
    eps_0 = mu_n + 3.0 * sigma_n

    # Step 3：粗估异常率
    est_rate = max(float(np.mean(r > eps_0)), 1e-5)

    # Step 4：目标预测率（2× 估计率，最多 5%）
    target_rate = min(est_rate * 2.0, 0.05)

    # Step 5：百分位阈值，确保不低于 eps_0
    eps_pct = float(np.percentile(r, 100.0 * (1.0 - target_rate)))
    eps     = max(eps_pct, eps_0)

    return eps


# ─────────────────────────────────────────────────────────────────────────────
#  辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _count_sequences(binary: np.ndarray) -> int:
    count, in_seq = 0, False
    for v in binary:
        if v and not in_seq:  count += 1; in_seq = True
        elif not v:            in_seq = False
    return count


def _extract_sequences_with_pos(
    r: np.ndarray, eps: float
) -> List[Tuple[int, int, np.ndarray]]:
    """提取所有超过 eps 的连续序列，返回 (start, end, values)"""
    seqs, in_seq, start = [], False, 0
    mask = r >= eps
    for i, v in enumerate(mask):
        if v and not in_seq:   start = i; in_seq = True
        elif not v and in_seq: seqs.append((start, i - 1, r[start:i])); in_seq = False
    if in_seq:                 seqs.append((start, len(r) - 1, r[start:]))
    return seqs


# ─────────────────────────────────────────────────────────────────────────────
#  假阳性剪枝（改进版）
# ─────────────────────────────────────────────────────────────────────────────

def _prune_false_positives(
    seqs:    List[Tuple],
    mu_r:    float,
    sigma_r: float,
    min_peak_z: float = 1.5,   # 序列峰值需超过 mu + min_peak_z*sigma 才保留
) -> List[Tuple]:
    """
    改进版假阳性剪枝：

    原方法（相邻峰值差距 < p_δ → 剪掉后续全部）的问题：
      若多个真实异常事件峰值相差 < 21%，只留一个，漏掉其余。

    新方法：按序列峰值的绝对强度过滤：
      保留峰值 > μ + min_peak_z × σ 的序列（即 1.5σ 以上的显著偏差）
      所有真实异常序列（残差远高于均值）都能保留
      低幅度噪声序列（残差刚超过 eps）被剪掉
    """
    threshold_peak = mu_r + min_peak_z * sigma_r
    return [(s, e, seq) for (s, e, seq) in seqs if np.max(seq) >= threshold_peak]


# ─────────────────────────────────────────────────────────────────────────────
#  异常分数计算（公式 27）
# ─────────────────────────────────────────────────────────────────────────────

def _compute_anomaly_scores(
    r:           np.ndarray,
    eps:         float,
    mu_r:        float,
    sigma_r:     float,
    min_peak_z:  float = 1.5,
) -> np.ndarray:
    """
    对每个保留的异常序列计算严重度分数（公式 27）：
      s = (max(r_seq) - ε*) / (μ(r) + σ(r))
    """
    scores = np.zeros(len(r), dtype=np.float32)
    seqs   = _extract_sequences_with_pos(r, eps)

    if not seqs:
        return scores

    # 改进版假阳性剪枝
    seqs = _prune_false_positives(seqs, mu_r, sigma_r, min_peak_z)

    denom = mu_r + sigma_r + 1e-9
    for s_start, s_end, seq in seqs:
        s_val = max(0.0, float(np.max(seq) - eps) / denom)
        scores[s_start : s_end + 1] = s_val

    return scores


# ─────────────────────────────────────────────────────────────────────────────
#  平滑
# ─────────────────────────────────────────────────────────────────────────────

def smooth_residuals(r: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return r.astype(np.float32)
    kernel = np.ones(window) / window
    return np.convolve(r, kernel, mode="same").astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  完整异常检测（Φ 算子）
# ─────────────────────────────────────────────────────────────────────────────

def detect_anomalies(
    x_true:        np.ndarray,   # [T, C]
    x_pred:        np.ndarray,   # [T, C]
    smooth_window: int   = 105,
    p_tfi:         float = 0.21,
    n_candidates:  int   = 300,
    min_peak_z:    float = 1.5,
) -> np.ndarray:
    """
    Φ 算子：计算连续异常分数 S ∈ R^T。

    流程：
      1. 残差 r = max_c |x_true_c - x_pred_c|，平滑
      2. 鲁棒阈值估计（改进版）
      3. 计算原始残差和反射残差的异常分数（捕获 silent failure）
      4. 取最大值合并
    """
    T, C = x_true.shape

    # 1. 残差（跨通道最大值）+ 平滑
    r_raw    = np.abs(x_true - x_pred).max(axis=1).astype(np.float64)
    r_smooth = smooth_residuals(r_raw, smooth_window).astype(np.float64)

    mu_r    = float(np.mean(r_smooth))
    sigma_r = float(np.std(r_smooth))

    # 2. 阈值（改进版）
    eps = _find_optimal_threshold(r_smooth, p_tfi, n_candidates)

    # 3. 原始残差的异常分数
    # 注：不使用反射残差（r_ref = 2μ - r）
    # ESA-AD 的正常数据本身就在 [0.006, 0.016] 之间自然波动
    # 反射后低残差正常点会被误判为"静默异常"→ 大量假阳性序列
    scores_raw = _compute_anomaly_scores(r_smooth, eps, mu_r, sigma_r, min_peak_z)

    anomaly_scores = scores_raw.astype(np.float32)

    n_seq = _count_sequences(anomaly_scores > 0)
    print(f"  阈值 ε* = {eps:.5f}  初始pred_rate = {(r_smooth >= eps).mean()*100:.3f}%")
    print(f"  剪枝后序列数 = {n_seq}  最终pred_rate = {(anomaly_scores>0).mean()*100:.3f}%")

    return anomaly_scores
