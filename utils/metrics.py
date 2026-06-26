"""
Evaluation Metrics for Time-Series Anomaly Detection.

Implements the metrics used in PSTG (Chen et al., 2026) and other
top-tier anomaly detection benchmarks:

1. Point-wise metrics (F1, Precision, Recall, AUC-ROC)
   — standard, allows comparison with GDN / ContrastAD / FuSAGNet

2. Event-wise F-beta (no point adjustment)
   — detects if anomaly *events* are identified; controls false alarms
   — PSTG reports Event-wise F0.5 = 0.917

3. Affiliation-based F-beta
   — measures temporal localisation quality via boundary proximity
   — PSTG reports Affiliation F0.5 = 0.892

References
----------
Huet et al. (2022), "Local Evaluation of Time Series Anomaly Detection
Algorithms", KDD 2022.  (affiliation metric)

Point adjustment is deliberately NOT applied in the primary evaluation
to match PSTG's no-PA protocol.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


# ── Helper: extract contiguous events ─────────────────────────────────────────

def _extract_events(binary: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end) inclusive indices of contiguous 1-runs."""
    events = []
    in_event = False
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


# ── Event-wise metrics ────────────────────────────────────────────────────────

def event_wise_fbeta(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 0.5,
) -> dict[str, float]:
    """Event-level precision / recall / F-beta.

    A predicted event (contiguous run of 1s) is a true positive if it
    overlaps with *at least one* ground-truth anomaly event.

    A ground-truth event is recalled if *at least one* predicted event
    overlaps it.

    Args:
        y_true: binary ground-truth labels (T,)
        y_pred: binary predictions (T,)
        beta:   F-beta beta (0.5 = precision-weighted)

    Returns:
        dict with keys: precision, recall, f_score
    """
    gt_events   = _extract_events(y_true)
    pred_events = _extract_events(y_pred)

    if not pred_events:
        return {"precision": 0.0, "recall": 0.0, "f_score": 0.0}
    if not gt_events:
        return {"precision": 0.0, "recall": 0.0, "f_score": 0.0}

    def overlaps(e1: tuple[int, int], e2: tuple[int, int]) -> bool:
        return e1[0] <= e2[1] and e2[0] <= e1[1]

    # Precision: fraction of predicted events that hit at least one GT event
    tp_pred = sum(
        1 for pe in pred_events if any(overlaps(pe, ge) for ge in gt_events)
    )
    precision = tp_pred / len(pred_events)

    # Recall: fraction of GT events detected by at least one predicted event
    tp_gt = sum(
        1 for ge in gt_events if any(overlaps(ge, pe) for pe in pred_events)
    )
    recall = tp_gt / len(gt_events)

    f_score = _fbeta_from_pr(precision, recall, beta)
    return {"precision": precision, "recall": recall, "f_score": f_score}


# ── Affiliation metric ────────────────────────────────────────────────────────

def affiliation_fbeta(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 0.5,
) -> dict[str, float]:
    """Affiliation-based precision / recall / F-beta.

    Based on Huet et al. (KDD 2022).  Each predicted point is affiliated
    to its nearest ground-truth anomaly event; the affiliation scores
    measure both coverage and boundary precision.

    This is a *simplified* implementation that captures the spirit of the
    metric.  For exact replication of PSTG numbers, install the official
    `affiliation-metrics` package (pip install affiliation-metrics) and
    replace this function with its API.

    Args:
        y_true: binary ground-truth labels (T,)
        y_pred: binary predictions (T,)
        beta:   F-beta parameter

    Returns:
        dict with keys: precision, recall, f_score
    """
    gt_events = _extract_events(y_true)
    if not gt_events:
        return {"precision": 0.0, "recall": 0.0, "f_score": 0.0}

    T = len(y_true)

    # For each GT event, compute affiliation precision and recall
    # as the average over time steps.

    total_aff_prec = 0.0
    total_aff_rec  = 0.0

    for ge_start, ge_end in gt_events:
        gt_len   = ge_end - ge_start + 1
        pred_seg = y_pred[ge_start : ge_end + 1]

        # Affiliation recall: what fraction of the GT event is covered
        aff_rec = float(pred_seg.sum()) / gt_len

        # Affiliation precision: for each TP in [ge_start, ge_end],
        # how close is it to the GT event boundary?
        # Simplified: fraction of predicted anomalies in the GT window
        # that actually fall inside (always 1.0 here since we're in the window).
        # We extend by looking at predictions outside the window that are "close".
        # Full implementation would weight by normalised distance to GT event.
        pred_in_window    = int(pred_seg.sum())
        pred_total_nearby = int(y_pred.sum())   # simplified: no distance weighting
        if pred_total_nearby == 0:
            aff_prec = 0.0
        else:
            aff_prec = pred_in_window / max(pred_total_nearby, 1)

        total_aff_prec += aff_prec
        total_aff_rec  += aff_rec

    n = len(gt_events)
    prec   = total_aff_prec / n
    recall = total_aff_rec  / n
    f_score = _fbeta_from_pr(prec, recall, beta)
    return {"precision": prec, "recall": recall, "f_score": f_score}


# ── Point-wise metrics ────────────────────────────────────────────────────────

def point_wise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict[str, float]:
    """Standard point-level metrics.

    Args:
        y_true:  binary (T,)
        y_pred:  binary (T,)
        scores:  continuous anomaly scores (T,) — used for AUC; optional

    Returns:
        dict with keys: precision, recall, f1, f05, auc (if scores given)
    """
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    result: dict[str, float] = {
        "precision": float(prec),
        "recall":    float(rec),
        "f1":        float(f1),
        "f05":       _fbeta_from_pr(float(prec), float(rec), beta=0.5),
    }

    if scores is not None and len(np.unique(y_true)) > 1:
        result["auc"] = float(roc_auc_score(y_true, scores))

    return result


# ── Combined evaluator ────────────────────────────────────────────────────────

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
    beta: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Run all evaluation metrics and return a nested dict.

    Returns
    -------
    {
        "point":       {precision, recall, f1, f05, auc},
        "event":       {precision, recall, f_score},
        "affiliation": {precision, recall, f_score},
    }
    """
    return {
        "point":       point_wise_metrics(y_true, y_pred, scores),
        "event":       event_wise_fbeta(y_true, y_pred, beta),
        "affiliation": affiliation_fbeta(y_true, y_pred, beta),
    }


# ── Private helpers ────────────────────────────────────────────────────────────

def _fbeta_from_pr(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision + recall == 0:
        return 0.0
    beta2 = beta ** 2
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)
