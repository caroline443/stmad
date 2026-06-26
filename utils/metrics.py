"""
Evaluation Metrics — aligned with ESA-ADB (Kotowski et al. 2024) and PSTG (Chen et al. 2026).

Primary metrics reported (no point adjustment, matches PSTG Table 5):

1. Event-wise F0.5  — uses the TNR-corrected precision from ESA-ADB / Sehili et al. (2023).
                      Formula: precision_ew = (TP/(TP+FP)) × TNR
                      where TNR = 1 − FP_steps / nominal_steps

2. Affiliation F0.5 — exact Huet et al. (KDD 2022) via official library.
                      ESA-ADB uses nanosecond timestamps; we use integer indices.
                      For uniformly sampled data (no gaps) the ratios are identical.

Secondary metrics (for cross-paper comparison):

3. Point-wise F1 / AUC — standard, comparable with GDN / ContrastAD / FuSAGNet.

Installation:
    pip install git+https://github.com/ahstat/affiliation-metrics-py.git

References
----------
Huet et al. (2022) KDD.          https://github.com/ahstat/affiliation-metrics-py
Sehili et al. (2023)              precision TNR correction
Kotowski et al. (2024) ESA-ADB.  https://github.com/kplabs-pl/ESA-ADB
Chen et al. (2026) PSTG.         Entropy 28, 426.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


# ── Affiliation library ───────────────────────────────────────────────────────

try:
    from affiliation.generics import convert_vector_to_events as _cvt
    from affiliation.metrics  import pr_from_events            as _pr
    _AFFILIATION_AVAILABLE = True
except ImportError:
    _AFFILIATION_AVAILABLE = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_events(binary: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous 1-runs as half-open (start, end) tuples.

    Matches affiliation-metrics library convention:
        [0,0,1,1,0,1,0] → [(2,4), (5,6)]
    """
    events: list[tuple[int, int]] = []
    in_event = False
    start = 0
    for i, v in enumerate(binary):
        if v and not in_event:
            start    = i
            in_event = True
        elif not v and in_event:
            events.append((start, i))
            in_event = False
    if in_event:
        events.append((start, len(binary)))
    return events


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True if half-open intervals [a0,a1) and [b0,b1) share ≥1 timestep."""
    return a[0] < b[1] and b[0] < a[1]


def _fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall)


# ── 1. Event-wise F0.5 (ESA-ADB / Sehili et al. 2023) ───────────────────────

def event_wise_fbeta(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta:   float = 0.5,
) -> dict[str, float]:
    """Event-wise Precision / Recall / F-beta matching ESA-ADB ESAScores.

    Precision uses the Sehili et al. (2023) TNR correction adopted by ESA-ADB:

        FP_events      = predicted events overlapping NO GT anomaly event
        FP_steps       = Σ duration(FP_events)
        nominal_steps  = total_T − Σ duration(GT_events)
        TNR            = clamp(1 − FP_steps / nominal_steps, 0, 1)
        precision_ew   = (TP / (TP + |FP_events|)) × TNR

    Recall (standard event-level):
        TP_gt     = GT events overlapped by ≥1 predicted event
        recall    = TP_gt / N_gt

    This makes precision stricter than plain TP/(TP+FP): false-positive events
    that cover large stretches of nominal time are penalised more heavily.

    Args:
        y_true: binary ground-truth (T,) — only actual anomaly events, not
                rare nominal events; filter those out before calling.
        y_pred: binary predictions  (T,)
        beta:   F-beta parameter (0.5 = precision-weighted, matches PSTG)

    Returns:
        dict: precision, recall, f_score, n_gt, n_pred, tnr, fp_steps
    """
    T = len(y_true)
    gt_events   = _extract_events(y_true)
    pred_events = _extract_events(y_pred)

    n_gt   = len(gt_events)
    n_pred = len(pred_events)

    # ── Edge cases ────────────────────────────────────────────────────────
    if n_gt == 0 and n_pred == 0:
        return {"precision": 1.0, "recall": 1.0, "f_score": 1.0,
                "n_gt": 0, "n_pred": 0, "tnr": 1.0, "fp_steps": 0}
    if n_gt == 0:
        # No GT anomalies; all predictions are FPs → precision = 0
        return {"precision": 0.0, "recall": 0.0, "f_score": 0.0,
                "n_gt": 0, "n_pred": n_pred, "tnr": 0.0,
                "fp_steps": sum(e - s for s, e in pred_events)}
    if n_pred == 0:
        return {"precision": 0.0, "recall": 0.0, "f_score": 0.0,
                "n_gt": n_gt, "n_pred": 0, "tnr": 1.0, "fp_steps": 0}

    # ── TP / FP classification ────────────────────────────────────────────
    # TP (for precision): predicted events overlapping ≥1 GT event
    tp_pred = [pe for pe in pred_events if any(_overlaps(pe, ge) for ge in gt_events)]
    fp_pred = [pe for pe in pred_events if not any(_overlaps(pe, ge) for ge in gt_events)]

    n_tp_pred  = len(tp_pred)
    n_fp_pred  = len(fp_pred)
    fp_steps   = sum(e - s for s, e in fp_pred)

    # TNR over nominal (non-anomaly) time
    gt_steps      = sum(e - s for s, e in gt_events)
    nominal_steps = max(1, T - gt_steps)          # avoid /0 if entire series is anomaly
    tnr           = max(0.0, 1.0 - fp_steps / nominal_steps)

    # TNR-corrected event-wise precision
    if n_tp_pred + n_fp_pred == 0:
        precision = 0.0
    else:
        precision = (n_tp_pred / (n_tp_pred + n_fp_pred)) * tnr

    # ── Recall ────────────────────────────────────────────────────────────
    # TP (for recall): GT events overlapped by ≥1 predicted event
    tp_gt = sum(
        1 for ge in gt_events
        if any(_overlaps(ge, pe) for pe in pred_events)
    )
    recall = tp_gt / n_gt

    return {
        "precision": precision,
        "recall":    recall,
        "f_score":   _fbeta(precision, recall, beta),
        "n_gt":      n_gt,
        "n_pred":    n_pred,
        "tnr":       tnr,
        "fp_steps":  fp_steps,
    }


# ── 2. Affiliation-based F0.5 (Huet et al. KDD 2022) ─────────────────────────

def affiliation_fbeta(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta:   float = 0.5,
) -> dict[str, float]:
    """Affiliation-based Precision / Recall / F-beta via official library.

    Uses pr_from_events() from https://github.com/ahstat/affiliation-metrics-py
    which is the same function used in the ESA-ADB benchmark.

    Timestamp convention: we pass integer indices [0, T).
    ESA-ADB uses nanosecond timestamps internally, but for uniformly sampled
    data with no gaps the survival-function ratios are identical (only the
    units change, not the relative distances).

    For data with irregular gaps, convert to actual timestamps before calling.

    Raises ImportError if affiliation-metrics is not installed.
    """
    if not _AFFILIATION_AVAILABLE:
        raise ImportError(
            "Install: pip install git+https://github.com/ahstat/affiliation-metrics-py.git"
        )

    T = len(y_true)
    gt_events   = _extract_events(y_true)
    pred_events = _extract_events(y_pred)

    if not gt_events:
        if not pred_events:
            return {"precision": 1.0, "recall": 1.0, "f_score": 1.0}
        return {"precision": 0.0, "recall": 0.0, "f_score": 0.0}
    if not pred_events:
        return {"precision": 0.0, "recall": 0.0, "f_score": 0.0}

    pr = _pr(pred_events, gt_events, Trange=(0, T))
    precision = float(pr["precision"])
    recall    = float(pr["recall"])

    return {
        "precision": precision,
        "recall":    recall,
        "f_score":   _fbeta(precision, recall, beta),
    }


# ── 3. Point-wise metrics ─────────────────────────────────────────────────────

def point_wise_metrics(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    scores:  np.ndarray | None = None,
) -> dict[str, float]:
    """Standard point-level metrics (no point adjustment).

    Comparable with GDN, ContrastAD, FuSAGNet, MSHTrans.
    """
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    result: dict[str, float] = {
        "precision": float(prec),
        "recall":    float(rec),
        "f1":        float(f1),
        "f05":       _fbeta(float(prec), float(rec), beta=0.5),
    }
    if scores is not None and len(np.unique(y_true)) > 1:
        try:
            result["auc"] = float(roc_auc_score(y_true, scores))
        except ValueError:
            result["auc"] = float("nan")
    return result


# ── 4. Combined evaluator ─────────────────────────────────────────────────────

def evaluate(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    scores:  np.ndarray | None = None,
    beta:    float = 0.5,
    strict:  bool  = True,
) -> dict[str, dict[str, float]]:
    """Run all metrics and return a nested result dict.

    Args:
        y_true:  binary ground-truth (T,) — anomalies only, rare nominal events
                 must be excluded BEFORE calling (set to 0 in y_true).
        y_pred:  binary predictions  (T,)
        scores:  continuous anomaly scores for AUC — optional
        beta:    F-beta parameter (0.5 matches PSTG)
        strict:  raise ImportError when affiliation-metrics missing (default True)

    Returns::

        {
            "point":       {precision, recall, f1, f05, auc},
            "event":       {precision, recall, f_score, n_gt, n_pred, tnr, fp_steps},
            "affiliation": {precision, recall, f_score},
        }
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    result = {
        "point": point_wise_metrics(y_true, y_pred, scores),
        "event": event_wise_fbeta(y_true, y_pred, beta),
    }

    try:
        result["affiliation"] = affiliation_fbeta(y_true, y_pred, beta)
    except ImportError:
        if strict:
            raise
        result["affiliation"] = {
            "precision": float("nan"),
            "recall":    float("nan"),
            "f_score":   float("nan"),
        }

    return result
