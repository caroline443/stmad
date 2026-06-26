"""
动态阈值异常检测（论文 Section 3.3，公式 26-28）

基于 Kotowski / Telemanom (Hundman et al. 2018) 的非参数动态阈值方法。

流程（对应 Algorithm 1 Part 3）：
  1. 计算残差序列 r = |X - X̂|（逐通道 max 聚合为 1D）
  2. 平滑处理（移动平均）
  3. 动态阈值求解（公式 26）
  4. 对原始残差和反射残差各做一次（捕获 silent failure）
  5. 计算各异常序列的严重度分数（公式 27）
  6. 假阳性剪枝（公式 28）
  7. 合并两次结果，生成最终异常分数 S
"""

import numpy as np
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  核心：动态阈值求解（公式 26）
# ─────────────────────────────────────────────────────────────────────────────

def _find_optimal_threshold(
    r: np.ndarray,          # 1D 残差序列
    p_tfi: float = 0.21,    # 搜索范围的最大百分位（论文：p_tfi 控制候选 ε 范围）
    n_candidates: int = 50, # 候选阈值数量
) -> float:
    """
    公式 26：ε* = argmax_ε (Δμ(r_s)/μ(r_s) + Δσ(r_s)/σ(r_s)) / (|r_a| + |R_seq|²)

    搜索范围：[μ(r) + σ(r), μ(r) + p_tfi × max(r)]（Telemanom 默认范围）
    """
    mu_r = np.mean(r)
    sigma_r = np.std(r)
    max_r = np.max(r)

    if sigma_r < 1e-9:
        return mu_r + sigma_r  # 退化情况

    # 候选 ε 范围（参考 Telemanom 源码）
    epsilon_low  = mu_r + sigma_r
    epsilon_high = mu_r + p_tfi * max_r

    if epsilon_high <= epsilon_low:
        return epsilon_low

    best_eps = epsilon_low
    best_score = -np.inf

    for eps in np.linspace(epsilon_low, epsilon_high, n_candidates):
        # r_s：正常残差（低于 ε）
        r_s = r[r < eps]
        # r_a：异常残差（≥ ε）
        r_a = r[r >= eps]

        if len(r_s) < 5 or len(r_a) == 0:
            continue

        mu_s = np.mean(r_s)
        sigma_s = np.std(r_s)

        if mu_s < 1e-9 or sigma_s < 1e-9:
            continue

        delta_mu    = mu_r - mu_s
        delta_sigma = sigma_r - sigma_s

        # 计算连续异常序列数 |R_seq|
        n_seqs = _count_sequences(r >= eps)

        # 公式 26 的分子/分母
        numerator   = delta_mu / mu_s + delta_sigma / sigma_s
        denominator = len(r_a) + n_seqs ** 2

        if denominator < 1e-9:
            continue

        score = numerator / denominator
        if score > best_score:
            best_score = score
            best_eps = eps

    return best_eps


def _count_sequences(binary: np.ndarray) -> int:
    """统计二值序列中连续 True 的段数"""
    count = 0
    in_seq = False
    for v in binary:
        if v and not in_seq:
            count += 1
            in_seq = True
        elif not v:
            in_seq = False
    return count


def _extract_anomaly_seqs(r: np.ndarray, eps: float) -> List[np.ndarray]:
    """提取所有超过阈值 ε 的连续异常子序列"""
    seqs = []
    in_seq = False
    start = 0
    mask = r >= eps
    for i, v in enumerate(mask):
        if v and not in_seq:
            start = i
            in_seq = True
        elif not v and in_seq:
            seqs.append(r[start:i])
            in_seq = False
    if in_seq:
        seqs.append(r[start:])
    return seqs


# ─────────────────────────────────────────────────────────────────────────────
#  异常分数计算（公式 27）+ 假阳性剪枝（公式 28）
# ─────────────────────────────────────────────────────────────────────────────

def _compute_anomaly_scores(
    r: np.ndarray,          # 1D 残差序列
    eps: float,             # 最优阈值 ε*
    p_delta: float = 0.21,  # 假阳性剪枝阈值 p_δ（对应 p_tfi）
) -> np.ndarray:
    """
    对每个异常序列计算严重度分数（公式 27），并进行假阳性剪枝（公式 28）。

    Returns:
        scores: [T] 浮点数异常分数（正常点为 0）
    """
    mu_r = np.mean(r)
    sigma_r = np.std(r) + 1e-9

    scores = np.zeros(len(r), dtype=np.float32)
    mask = r >= eps

    # 提取异常序列及其位置
    seqs_with_pos = []
    in_seq = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_seq:
            start = i
            in_seq = True
        elif not v and in_seq:
            seqs_with_pos.append((start, i - 1, r[start:i]))
            in_seq = False
    if in_seq:
        seqs_with_pos.append((start, len(r) - 1, r[start:]))

    if not seqs_with_pos:
        return scores

    # 按 max 降序排列（用于假阳性剪枝）
    seqs_with_pos.sort(key=lambda x: np.max(x[2]), reverse=True)

    # 假阳性剪枝（公式 28）
    prev_max = None
    valid_seqs = []
    for s_start, s_end, seq in seqs_with_pos:
        cur_max = np.max(seq)
        if prev_max is not None:
            d = (prev_max - cur_max) / (prev_max + 1e-9)
            if d < p_delta:
                # 剩余序列都剪枝
                break
        valid_seqs.append((s_start, s_end, seq))
        prev_max = cur_max

    # 计算分数（公式 27）
    for s_start, s_end, seq in valid_seqs:
        s_val = (np.max(seq) - eps) / (mu_r + sigma_r)
        s_val = max(0.0, float(s_val))
        scores[s_start:s_end + 1] = s_val

    return scores


# ─────────────────────────────────────────────────────────────────────────────
#  完整异常检测（Φ 算子）
# ─────────────────────────────────────────────────────────────────────────────

def smooth_residuals(r: np.ndarray, window: int) -> np.ndarray:
    """移动平均平滑（沿时间轴）"""
    if window <= 1:
        return r
    kernel = np.ones(window) / window
    return np.convolve(r, kernel, mode="same")


def detect_anomalies(
    x_true: np.ndarray,     # [T, C] 真实值（归一化后）
    x_pred: np.ndarray,     # [T, C] 预测值（归一化后）
    smooth_window: int = 105,
    p_tfi: float = 0.21,
    n_candidates: int = 50,
) -> np.ndarray:
    """
    Φ 算子：给定真实值和预测值，输出连续异常分数序列 S ∈ R^T（公式 1）。

    实现 Algorithm 1 Part 3：
    1. 计算多通道残差，逐通道最大值聚合为 1D
    2. 平滑
    3. 对原始残差和反射残差各做一次动态阈值检测
    4. 合并为最终异常分数

    Args:
        x_true: [T, C]
        x_pred: [T, C]
        smooth_window: 移动平均窗口大小（W_s = p_s × n_s × B_s）
        p_tfi: 候选阈值上界参数（同时用于假阳性剪枝）
        n_candidates: 阈值搜索候选数量

    Returns:
        anomaly_scores: [T] 连续分数
    """
    T, C = x_true.shape

    # 1. 残差：逐通道绝对误差，取各通道的最大值
    residual_per_channel = np.abs(x_true - x_pred)   # [T, C]
    r = residual_per_channel.max(axis=1).astype(np.float64)   # [T]

    # 2. 平滑
    r_smooth = smooth_residuals(r, smooth_window)

    # 3. 原始残差：动态阈值
    eps_raw = _find_optimal_threshold(r_smooth, p_tfi, n_candidates)
    scores_raw = _compute_anomaly_scores(r_smooth, eps_raw, p_tfi)

    # 4. 反射残差（捕获 silent failure，即异常表现为信号骤降）
    mu_r = np.mean(r_smooth)
    r_ref = 2 * mu_r - r_smooth   # 反射（公式描述中的 r_ref = 2μ(r) - r）
    r_ref = np.clip(r_ref, 0, None)   # 保持非负
    eps_ref = _find_optimal_threshold(r_ref, p_tfi, n_candidates)
    scores_ref = _compute_anomaly_scores(r_ref, eps_ref, p_tfi)

    # 5. 合并：取两次结果的最大值
    anomaly_scores = np.maximum(scores_raw, scores_ref).astype(np.float32)

    return anomaly_scores


def build_full_prediction(
    model,
    test_loader,
    device: str,
    context_len: int,
    forecast_len: int,
    tau: int,
    total_test_len: int,
) -> np.ndarray:
    """
    Algorithm 1 Part 2：滑动窗口推理，拼接为完整预测序列 X̂。

    每个窗口预测 F 步，只保留前 τ 步（τ=1），
    拼接得到与测试集等长的预测序列。

    Returns:
        x_pred_full: [T_pred, C]，其中 T_pred = total_test_len - context_len
    """
    import torch

    model.eval()
    all_preds = []

    with torch.no_grad():
        for context, t_idx in test_loader:
            context = context.to(device, non_blocking=True)   # [B, C, L]
            pred = model(context)                              # [B, C, F]
            # 只保留前 τ 步（τ=1）
            pred_tau = pred[:, :, :tau]                       # [B, C, τ]
            # 转置为 [B, τ, C]，然后 reshape 为 [B*τ, C]
            pred_tau = pred_tau.permute(0, 2, 1).reshape(-1, pred.shape[1])
            all_preds.append(pred_tau.cpu().numpy())

    x_pred_full = np.concatenate(all_preds, axis=0)   # [T_pred, C]
    return x_pred_full.astype(np.float32)
