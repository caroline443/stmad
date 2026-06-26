"""
Dynamic Thresholding.

Implements the PSTG-style threshold fitting strategy:

1. Compute anomaly scores on the *validation* (or training) set.
2. Sort scores and take the p_fit percentile as the initial threshold.
3. Optionally apply Gaussian smoothing to the *test* scores before
   comparing against the threshold (reduces single-point false positives).
4. (Optional) search over a range of thresholds to maximise F-beta on a
   labelled validation set when labels are available.

Reference
---------
Chen et al., "Progressive Spatiotemporal Graph Modelling for Spacecraft
Anomaly Detection", Entropy 2026, Table 3: p_fit=0.21, p_s=0.05, n_s=30.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


class DynamicThreshold:
    """Percentile-based dynamic threshold with optional smoothing.

    Args:
        p_fit:  percentile (0–1) of the score distribution on the clean
                (train/val) set used as threshold.  Default 0.21 (PSTG).
        p_s:    fraction of n_s that defines the Gaussian std for
                smoothing (currently unused; kept for API compatibility).
        n_s:    smoothing window size (uniform moving average). 0 = no smoothing.
    """

    def __init__(
        self,
        p_fit: float = 0.21,
        p_s:   float = 0.05,
        n_s:   int   = 30,
    ) -> None:
        self.p_fit = p_fit
        self.p_s   = p_s
        self.n_s   = n_s
        self.threshold: float | None = None

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, scores: np.ndarray) -> "DynamicThreshold":
        """Fit threshold from clean (train/val) anomaly scores.

        Args:
            scores: 1-D float array of reconstruction errors on clean data

        Returns:
            self (for chaining)
        """
        self.threshold = float(np.percentile(scores, (1 - self.p_fit) * 100))
        return self

    def fit_optimal(
        self,
        val_scores: np.ndarray,
        val_labels: np.ndarray,
        beta: float = 0.5,
        n_candidates: int = 200,
    ) -> "DynamicThreshold":
        """Grid-search for the threshold that maximises F-beta on the val set.

        Args:
            val_scores:  1-D float array of anomaly scores on the val set
            val_labels:  1-D binary array (0/1) of ground-truth labels
            beta:        F-beta beta parameter (0.5 = precision-weighted)
            n_candidates: number of threshold candidates to evaluate
        """
        lo, hi = float(val_scores.min()), float(val_scores.max())
        best_score = -1.0
        best_thr   = lo

        for thr in np.linspace(lo, hi, n_candidates):
            preds = (val_scores >= thr).astype(np.int32)
            fb    = _fbeta(val_labels, preds, beta)
            if fb > best_score:
                best_score = fb
                best_thr   = float(thr)

        self.threshold = best_thr
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Return binary predictions (1 = anomaly) for test scores.

        Applies optional smoothing before thresholding.

        Args:
            scores: 1-D float array of test anomaly scores

        Returns:
            preds: 1-D int array (0/1)
        """
        if self.threshold is None:
            raise RuntimeError("Call fit() or fit_optimal() before predict()")

        s = self._smooth(scores)
        return (s >= self.threshold).astype(np.int32)

    def _smooth(self, scores: np.ndarray) -> np.ndarray:
        if self.n_s <= 1:
            return scores
        return uniform_filter1d(scores, size=self.n_s, mode="nearest")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fbeta(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5) -> float:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    beta2     = beta ** 2
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)
