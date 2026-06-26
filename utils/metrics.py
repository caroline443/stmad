"""
ESA-AD 评估指标（论文 Section 4.3，公式 29）

1. Event-wise F0.5  ：事件级别精度优先 F 分数（论文主要指标）
2. Affiliation-based F0.5：时间域对齐质量评估

注意：不使用 Point Adjustment（PA）协议（论文明确说明）。
"""

import numpy as np
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  辅助函数：事件提取
# ─────────────────────────────────────────────────────────────────────────────

def extract_events(binary: np.ndarray) -> List[Tuple[int, int]]:
    """
    从二值序列中提取连续事件（1 序列）。

    Args:
        binary: [T] 的 0/1 数组
    Returns:
        events: List[(start, end)]，end 是最后一个 1 的索引（含）
    """
    events = []
    in_event = False
    start = 0
    for i, v in enumerate(binary):
        if v == 1 and not in_event:
            start = i
            in_event = True
        elif v == 0 and in_event:
            events.append((start, i - 1))
            in_event = False
    if in_event:
        events.append((start, len(binary) - 1))
    return events


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """F_beta 分数，beta<1 时更重视 precision。"""
    if precision + recall < 1e-9:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall)


# ─────────────────────────────────────────────────────────────────────────────
#  1. Event-wise F0.5
# ─────────────────────────────────────────────────────────────────────────────

def event_wise_metrics(
    y_true: np.ndarray,   # [T] 真实二值标签
    y_pred: np.ndarray,   # [T] 预测二值标签
    beta: float = 0.5,
) -> dict:
    """
    事件级别的 Precision / Recall / F_beta。

    - Precision：预测的事件中有多少与真实事件重叠
    - Recall：真实事件中有多少被预测到

    这里使用"重叠"判断：预测事件与真实事件有任意一个时间步重叠即算命中。
    """
    gt_events = extract_events(y_true)
    pred_events = extract_events(y_pred)

    if len(pred_events) == 0:
        prec = 1.0 if len(gt_events) == 0 else 0.0
        rec  = 1.0 if len(gt_events) == 0 else 0.0
        return {"precision": prec, "recall": rec, "f0.5": f_beta(prec, rec, beta)}

    if len(gt_events) == 0:
        return {"precision": 0.0, "recall": 1.0, "f0.5": 0.0}

    def overlaps(p_start, p_end, g_start, g_end):
        return p_start <= g_end and p_end >= g_start

    # Recall：每个 gt 事件是否被任意 pred 事件覆盖
    gt_detected = 0
    for (gs, ge) in gt_events:
        for (ps, pe) in pred_events:
            if overlaps(ps, pe, gs, ge):
                gt_detected += 1
                break

    # Precision：每个 pred 事件是否与任意 gt 事件重叠
    pred_correct = 0
    for (ps, pe) in pred_events:
        for (gs, ge) in gt_events:
            if overlaps(ps, pe, gs, ge):
                pred_correct += 1
                break

    precision = pred_correct / len(pred_events)
    recall    = gt_detected  / len(gt_events)
    return {
        "precision": precision,
        "recall":    recall,
        "f0.5":      f_beta(precision, recall, beta),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  2. Affiliation-based F0.5（公式 29）
# ─────────────────────────────────────────────────────────────────────────────

def _affiliation_zone(gt_events: List[Tuple], T: int, j: int) -> Tuple[int, int]:
    """
    计算第 j 个 gt 事件的 affiliation zone I_j。

    Affiliation zone：从前一事件结束到后一事件开始的中点范围。
    若没有前/后事件，则延伸到序列边界。
    """
    gs, ge = gt_events[j]
    # 左边界：前一事件结束后的中点
    if j == 0:
        left = 0
    else:
        prev_end = gt_events[j - 1][1]
        left = (prev_end + gs) // 2 + 1
    # 右边界：后一事件开始前的中点
    if j == len(gt_events) - 1:
        right = T - 1
    else:
        next_start = gt_events[j + 1][0]
        right = (ge + next_start) // 2
    return left, right


def _survival_func(d: float, zone_len: float) -> float:
    """
    均匀分布的生存函数 F̄(d)（Huet et al. 2022，均匀随机参照）。
    F̄(d) = max(0, 1 - d / (zone_len / 2))
    即距离 d 越小越好，d=0 时得分为 1，d≥zone_len/2 时为 0。
    """
    if zone_len <= 0:
        return 1.0 if d == 0 else 0.0
    return max(0.0, 1.0 - d / (zone_len / 2.0))


def _min_dist_to_set(t: int, points: np.ndarray) -> float:
    """点 t 到点集 points 的最小距离。若 points 为空返回 inf。"""
    if len(points) == 0:
        return float("inf")
    return float(np.min(np.abs(points - t)))


def affiliation_metrics(
    y_true: np.ndarray,   # [T] 真实二值标签
    y_pred: np.ndarray,   # [T] 预测二值标签
    beta: float = 0.5,
) -> dict:
    """
    Affiliation-based Precision / Recall / F_beta（公式 29）。

    基于时间域对齐质量：预测区间与真实区间的时间距离越小，分数越高。
    """
    T = len(y_true)
    gt_events = extract_events(y_true)
    N = len(gt_events)

    if N == 0:
        return {"precision": 1.0, "recall": 1.0, "f0.5": 1.0}

    pred_points = np.where(y_pred == 1)[0]  # 所有预测为异常的时间点
    pred_events = extract_events(y_pred)

    # ── P_aff 计算 ──────────────────────────────────────────────────────────
    # P_aff = (1/N) × [Σ_{j∈J} 1/|pred∩I_j| × ∫_{x∈pred∩I_j} F̄(min_y|x-y|) dx
    #                  + (N - |J|) × 0.5]

    J_count = 0     # |J|：有预测覆盖的 gt 事件数
    prec_sum = 0.0

    for j, (gs, ge) in enumerate(gt_events):
        iz_left, iz_right = _affiliation_zone(gt_events, T, j)
        zone_len = iz_right - iz_left + 1
        gt_points = np.arange(gs, ge + 1)

        # pred 在 affiliation zone 内的点
        pred_in_zone = np.where(
            (y_pred[iz_left:iz_right + 1] == 1)
        )[0] + iz_left

        if len(pred_in_zone) == 0:
            # 这个 gt 事件未被检测（贡献 0.5 给 P_aff）
            prec_sum += 0.5
        else:
            J_count += 1
            # 对 pred_in_zone 中每个点，计算到 gt 的最小距离，然后算生存函数值
            contrib = 0.0
            for x in pred_in_zone:
                d = _min_dist_to_set(x, gt_points)
                contrib += _survival_func(d, zone_len)
            prec_sum += contrib / len(pred_in_zone)

    P_aff = prec_sum / N

    # ── R_aff 计算 ──────────────────────────────────────────────────────────
    # R_aff = (1/N) × Σ_{j=1}^N 1/|gt_j| × ∫_{y∈gt_j} F̄(min_{x∈pred∩I_j} |x-y|) dy

    rec_sum = 0.0

    for j, (gs, ge) in enumerate(gt_events):
        iz_left, iz_right = _affiliation_zone(gt_events, T, j)
        zone_len = iz_right - iz_left + 1
        gt_points = np.arange(gs, ge + 1)

        # pred 在 affiliation zone 内的点
        pred_in_zone = np.where(
            (y_pred[iz_left:iz_right + 1] == 1)
        )[0] + iz_left

        contrib = 0.0
        for y in gt_points:
            if len(pred_in_zone) == 0:
                # 无预测点，距离无限大，生存函数为 0
                pass
            else:
                d = _min_dist_to_set(y, pred_in_zone)
                contrib += _survival_func(d, zone_len)
        rec_sum += contrib / max(len(gt_points), 1)

    R_aff = rec_sum / N

    return {
        "precision": P_aff,
        "recall":    R_aff,
        "f0.5":      f_beta(P_aff, R_aff, beta),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  综合评估入口
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(
    y_true: np.ndarray,      # [T] 真实标签
    anomaly_scores: np.ndarray,  # [T] 连续异常分数
    threshold: float,            # 二值化阈值
) -> dict:
    """
    给定阈值，计算 event-wise 和 affiliation-based 两组指标。
    """
    y_pred = (anomaly_scores > threshold).astype(np.int32)

    ew = event_wise_metrics(y_true, y_pred)
    af = affiliation_metrics(y_true, y_pred)

    return {
        "event_wise": ew,
        "affiliation": af,
        "threshold": threshold,
        "pred_anomaly_rate": y_pred.mean(),
    }


def find_best_threshold(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    metric: str = "event_f05",
    n_thresholds: int = 200,
) -> Tuple[float, dict]:
    """
    在分数分布中搜索使指标最优的阈值。

    注意：对高度不平衡的数据（异常比例 <1%），不能用 p1~p99 作为搜索范围，
    因为 p99 可能为 0。改为用实际分数分布的有效范围来搜索。
    """
    s_min = float(anomaly_scores.min())
    s_max = float(anomaly_scores.max())

    if s_max <= s_min + 1e-9:
        # 所有分数相同，无法区分
        result = evaluate_all(y_true, anomaly_scores, s_min)
        return s_min, result

    # 在 [min, max] 上均匀搜索，覆盖从"全部标记"到"全部忽略"的完整范围
    best_score = -1.0
    best_result = None
    best_thresh = s_min

    for thresh in np.linspace(s_min, s_max, n_thresholds):
        result = evaluate_all(y_true, anomaly_scores, thresh)
        if metric == "event_f05":
            score = result["event_wise"]["f0.5"]
        else:
            score = result["affiliation"]["f0.5"]
        if score > best_score:
            best_score = score
            best_result = result
            best_thresh = thresh

    return best_thresh, best_result
