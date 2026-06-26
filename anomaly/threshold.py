"""
异常阈值选择。

实现两种方式：

1. PSTG 论文 Eq. 26-28 的阈值公式
   e* = argmax [Δμ(r_ε)/μ(r) + Δσ(r_ε)/σ(r)] / [|r_ε| + |R_seq|]

2. Oracle 评估（扫描所有候选阈值，取使 F0.5 最大的那个）
   用于确认模型是否有判别能力，与阈值选择无关
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


# ── 工具 ──────────────────────────────────────────────────────────────────────

def _smooth(scores: np.ndarray, n_s: int = 30) -> np.ndarray:
    if n_s <= 1:
        return scores
    return uniform_filter1d(scores.astype(np.float64), size=n_s, mode="nearest").astype(np.float32)


def _fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def _extract_events(binary: np.ndarray) -> list[tuple[int, int]]:
    events, in_ev, s = [], False, 0
    for i, v in enumerate(binary):
        if v and not in_ev:
            s, in_ev = i, True
        elif not v and in_ev:
            events.append((s, i))
            in_ev = False
    if in_ev:
        events.append((s, len(binary)))
    return events


# ── 1. PSTG 论文 Eq. 26-28 ────────────────────────────────────────────────────

class PSTGThreshold:
    """PSTG 论文 Section 3.3 Statistical Anomaly Decision 的完整实现。

    Eq. 26: 最优阈值 e* = argmax score(ε)
            score(ε) = [Δμ/μ + Δσ/σ] / [|r_ε| + |R_seq|]

    Eq. 27: 异常严重性评分 s^(i) = (max(r_seq^(i)) - e*) / (μ + σ)

    Eq. 28: FP 剪枝，d^(i) = (r^(i-1)_max - r^(i)_max) / r^(i-1)_max
            若 d^(i) < p_s，将该段及后续低 rank 段归为正常
    """

    def __init__(
        self,
        p_fit: float = 0.21,   # 候选阈值搜索范围下界分位数（如只取 top 21%）
        p_s:   float = 0.05,   # FP 剪枝阈值 d^(i) < p_s 则剪枝
        n_s:   int   = 30,     # 平滑窗口
    ) -> None:
        self.p_fit     = p_fit
        self.p_s       = p_s
        self.n_s       = n_s
        self.threshold: float | None = None

    def fit(self, scores: np.ndarray) -> "PSTGThreshold":
        """在（无标签）训练/验证集上用 Eq. 26 找最优阈值。"""
        r = _smooth(scores, self.n_s).astype(np.float64)
        r = r[~np.isnan(r)]

        mu    = r.mean()
        sigma = r.std()
        if mu == 0 or sigma == 0:
            self.threshold = float(np.percentile(r, (1 - self.p_fit) * 100))
            return self

        # 候选 ε：从 (1-p_fit) 分位数到最大值，取 500 个候选点
        lo  = float(np.percentile(r, (1 - self.p_fit) * 100))
        hi  = float(r.max())
        candidates = np.linspace(lo, hi, 500)

        best_score = -np.inf
        best_e     = lo

        for e in candidates:
            r_above = r[r > e]
            if len(r_above) == 0:
                continue

            # Δμ, Δσ：移除异常点后的统计量变化
            r_below = r[r <= e]
            if len(r_below) == 0:
                continue
            delta_mu    = mu    - r_below.mean()
            delta_sigma = sigma - r_below.std()

            # |R_seq|：连续超阈值段的数量
            binary  = (r > e).astype(np.int32)
            R_seq   = len(_extract_events(binary))

            denom = len(r_above) + R_seq
            if denom == 0:
                continue

            # Eq. 26
            score = (delta_mu / mu + delta_sigma / sigma) / denom
            if score > best_score:
                best_score = score
                best_e     = float(e)

        self.threshold = best_e
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """生成二进制预测，并应用 FP 剪枝（Eq. 28）。"""
        if self.threshold is None:
            raise RuntimeError("先调用 fit()")

        r      = _smooth(scores, self.n_s)
        binary = (r >= self.threshold).astype(np.int32)

        # Eq. 28 FP 剪枝
        binary = self._prune_fp(r, binary)
        return binary

    def _prune_fp(self, r: np.ndarray, binary: np.ndarray) -> np.ndarray:
        """Eq. 28：按严重性排序预测段，剪掉严重性落差大的低 rank 段。"""
        events = _extract_events(binary)
        if len(events) < 2:
            return binary

        # 每段的最大残差（严重性代理）
        severities = [float(r[s:e].max()) for s, e in events]

        # 按严重性降序排列，计算相邻段的落差
        sorted_idx = np.argsort(severities)[::-1]
        result = binary.copy()

        for rank in range(1, len(sorted_idx)):
            prev_idx = sorted_idx[rank - 1]
            curr_idx = sorted_idx[rank]
            s_prev   = severities[prev_idx]
            s_curr   = severities[curr_idx]

            if s_prev == 0:
                break

            d = (s_prev - s_curr) / s_prev   # Eq. 28
            if d < self.p_s:
                # 这段及后续低 rank 段全部剪掉
                for j in range(rank, len(sorted_idx)):
                    s, e = events[sorted_idx[j]]
                    result[s:e] = 0
                break

        return result


# ── 2. Oracle 评估（用于诊断模型判别能力）────────────────────────────────────

def oracle_best_f(
    scores: np.ndarray,
    labels: np.ndarray,
    beta:   float = 0.5,
    n_candidates: int = 500,
    level:  str   = "point",   # "point" 或 "event"
) -> dict[str, float]:
    """扫描所有候选阈值，返回最优 F-beta 及对应的 precision / recall / threshold。

    这是理论上限：如果这里 F0.5 很低，说明模型的分数根本没有区分能力；
    如果 F0.5 较高，说明模型可以，只是阈值选择有问题。
    """
    from utils.metrics import event_wise_fbeta

    lo  = float(np.nanpercentile(scores, 1))
    hi  = float(np.nanpercentile(scores, 99))
    candidates = np.linspace(lo, hi, n_candidates)

    best = {"f_score": -1.0, "threshold": lo, "precision": 0.0, "recall": 0.0}

    for thr in candidates:
        pred = (scores >= thr).astype(np.int32)

        if level == "event":
            r    = event_wise_fbeta(labels.astype(np.int32), pred, beta)
            fsco = r["f_score"]
            prec = r["precision"]
            rec  = r["recall"]
        else:
            tp = int(((pred == 1) & (labels == 1)).sum())
            fp = int(((pred == 1) & (labels == 0)).sum())
            fn = int(((pred == 0) & (labels == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fsco = _fbeta(prec, rec, beta)

        if fsco > best["f_score"]:
            best = {"f_score": fsco, "threshold": float(thr),
                    "precision": prec, "recall": rec}

    return best


# ── 旧 API 兼容（DynamicThreshold → PSTGThreshold）────────────────────────────

class DynamicThreshold(PSTGThreshold):
    """向后兼容别名。"""

    def fit_optimal(
        self,
        val_scores: np.ndarray,
        val_labels: np.ndarray,
        beta: float = 0.5,
        n_candidates: int = 500,
    ) -> "DynamicThreshold":
        """当 val 有标签时，直接做 oracle 搜索。"""
        res = oracle_best_f(val_scores, val_labels, beta=beta,
                            n_candidates=n_candidates, level="point")
        self.threshold = res["threshold"]
        return self
