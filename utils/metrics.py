"""
Evaluation Metrics — fully aligned with PSTG (Chen et al., Entropy 2026).

Two primary metrics (no point adjustment):

1. Event-wise F0.5
   Definition follows the ESA-AD benchmark (Kotowski et al., 2024).
   A predicted event is a TP if it overlaps any GT anomaly event.
   A GT event is recalled if any predicted event overlaps it.
   Multiple predictions covering the same GT event count as 1 recall TP.

2. Affiliation-based F0.5
   Exact implementation of Huet et al. (KDD 2022) via the official library.
   Measures boundary proximity and coverage completeness.
   Uses the `affiliation-metrics` library when available; raises if not installed.

Installation:
    pip install git+https://github.com/ahstat/affiliation-metrics-py.git

References
----------
Huet et al. (2022) "Local Evaluation of Time Series Anomaly Detection
  Algorithms", KDD 2022.  https://github.com/ahstat/affiliation-metrics-py
Kotowski et al. (2024) "European Space Agency Benchmark for Anomaly
  Detection in Satellite Telemetry", arXiv 2406.xxxxx.
Chen et al. (2026) "Progressive Spatiotemporal Graph Modelling for
  Spacecraft Anomaly Detection", Entropy 28, 426.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


# ── Affiliation library (required for primary metric) ─────────────────────────

try:
    from affiliation.generics import convert_vector_to_events as _cvt
    from affiliation.metrics  import pr_from_events            as _pr
    _AFFILIATION_AVAILABLE = True
except ImportError:
    _AFFILIATION_AVAILABLE = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_events(binary: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous 1-runs as (start, end_exclusive) tuples.

    Matches the convention used by the affiliation-metrics library:
        events_pred = [(4, 5), (8, 9)]  ← half-open intervals
    """
    events: list[tuple[int, int]] = []
    in_event = False
    start = 0
    for i, v in enumerate(binary):
        if v and not in_event:
            start    = i
            in_event = True
        elif not v and in_event:
            events.append((start, i))   # [start, i)  ← i is exclusive
            in_event = False
    if in_event:
        events.append((start, len(binary)))
    return events


def _fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall)


# ── 1. Event-wise F0.5 ────────────────────────────────────────────────────────

def event_wise_fbeta(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 0.5,
) -> dict[str, float]:
    """Event-level Precision / Recall / F-beta.

    Matching rule (ESA-AD / Kotowski 2024):
      - A predicted event is a True Positive  if it overlaps ≥1 GT event.
      - A GT event     is a True Positive  if it is overlapped by ≥1 prediction.
      - Multiple predictions covering the same GT event  → 1 recall TP.
      - Multiple GT events covered by the same prediction → 1 precision TP
        (the prediction is still one event).

    Args:
        y_true: binary ground-truth  (T,)
        y_pred: binary predictions   (T,)
        beta:   F-beta parameter (0.5 = precision-weighted)

    Returns:
        {"precision": …, "recall": …, "f_score": …, "n_gt": …, "n_pred": …}
    """
    gt_events   = _extract_events(y_true)
    pred_events = _extract_events(y_pred)

    n_gt   = len(gt_events)
    n_pred = len(pred_events)

    if n_gt == 0 and n_pred == 0:
        return {"precision": 1.0, "recall": 1.0, "f_score": 1.0,
                "n_gt": 0, "n_pred": 0}
    if n_gt == 0 or n_pred == 0:
        return {"precision": 0.0, "recall": 0.0, "f_score": 0.0,
                "n_gt": n_gt, "n_pred": n_pred}

    def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
        # half-open intervals [a0, a1) and [b0, b1) overlap iff a0 < b1 and b0 < a1
        return a[0] < b[1] and b[0] < a[1]

    # Precision: fraction of predicted events that hit ≥1 GT event
    tp_pred = sum(
        1 for pe in pred_events
        if any(overlaps(pe, ge) for ge in gt_events)
    )
    precision = tp_pred / n_pred

    # Recall: fraction of GT events hit by ≥1 predicted event
    tp_gt = sum(
        1 for ge in gt_events
        if any(overlaps(ge, pe) for pe in pred_events)
    )
    recall = tp_gt / n_gt

    return {
        "precision": precision,
        "recall":    recall,
        "f_score":   _fbeta(precision, recall, beta),
        "n_gt":      n_gt,
        "n_pred":    n_pred,
    }


# ── 2. Affiliation-based F0.5 (Huet et al. KDD 2022) ─────────────────────────

def affiliation_fbeta(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 0.5,
) -> dict[str, float]:
    """Affiliation-based Precision / Recall / F-beta.

    Exact implementation via the `affiliation-metrics` library (Huet et al.).
    The library must be installed:
        pip install git+https://github.com/ahstat/affiliation-metrics-py.git

    The metric measures temporal localisation quality:
      • Precision: how close predicted points are to GT anomaly boundaries.
      • Recall:    how well predicted points cover GT anomaly intervals.
    Both use survival-function weighting over affiliation zones I_j.

    Args:
        y_true: binary ground-truth  (T,)
        y_pred: binary predictions   (T,)
        beta:   F-beta parameter (0.5 = precision-weighted)

    Returns:
        {"precision": …, "recall": …, "f_score": …}

    Raises:
        ImportError: if affiliation-metrics is not installed.
    """
    if not _AFFILIATION_AVAILABLE:
        raise ImportError(
            "The `affiliation-metrics` library is required for this metric.\n"
            "Install with:\n"
            "  pip install git+https://github.com/ahstat/affiliation-metrics-py.git"
        )

    T = len(y_true)

    gt_events   = _extract_events(y_true)
    pred_events = _extract_events(y_pred)

    # Edge cases
    if len(gt_events) == 0:
        if len(pred_events) == 0:
            return {"precision": 1.0, "recall": 1.0, "f_score": 1.0}
        return {"precision": 0.0, "recall": 0.0, "f_score": 0.0}

    if len(pred_events) == 0:
        return {"precision": 0.0, "recall": 0.0, "f_score": 0.0}

    Trange = (0, T)
    pr = _pr(pred_events, gt_events, Trange)

    precision = float(pr["precision"])
    recall    = float(pr["recall"])

    return {
        "precision": precision,
        "recall":    recall,
        "f_score":   _fbeta(precision, recall, beta),
    }


# ── 3. Point-wise metrics ─────────────────────────────────────────────────────

def point_wise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict[str, float]:
    """Standard point-level metrics (no point adjustment).

    Useful for comparison with papers that report F1 / AUC.

    Returns:
        {"precision": …, "recall": …, "f1": …, "f05": …, "auc": … (if scores)}
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
        y_true:  binary ground-truth (T,)
        y_pred:  binary predictions  (T,)
        scores:  continuous anomaly scores (T,) for AUC — optional
        beta:    F-beta parameter (default 0.5 = precision-weighted, matches PSTG)
        strict:  if True, raise ImportError when affiliation-metrics is missing;
                 if False, return NaN values instead

    Returns::

        {
            "point":       {precision, recall, f1, f05, auc},
            "event":       {precision, recall, f_score, n_gt, n_pred},
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
        result["affiliation"] = {"precision": float("nan"),
                                 "recall":    float("nan"),
                                 "f_score":   float("nan")}

    return result
