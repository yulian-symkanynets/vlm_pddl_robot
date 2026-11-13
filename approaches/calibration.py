"""Choosing the confidence threshold from data instead of by feel.

The server plans only for objects whose identity confidence is at least
`min_confidence`; below that it asks for another view. Set too low, the robot
sorts objects into the wrong zone; set too high, it keeps asking for views it
does not need.

Given a log of past detections (confidence, was the label correct?) this module
picks the lowest threshold for which the error rate among accepted detections is
below a target `alpha`, with a finite-sample safety margin (exact binomial bound) so
that the guarantee holds with probability at least 1 - delta, not just on average.
This is risk control for selective prediction: the same idea that KnowNo uses
with conformal prediction to decide when a robot should ask for help.

The guarantee assumes future detections come from the same conditions as the
log (lighting, objects, camera). Re-calibrate when those change.
"""
from __future__ import annotations

import math


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed in log space."""
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log(1 - p)
    total = 0.0
    for i in range(k + 1):
        total += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                          + i * lp + (n - i) * lq)
    return min(1.0, total)


def error_bound(errors: int, n: int, delta: float) -> float:
    """Exact (Clopper-Pearson) upper confidence bound on the true error rate after
    seeing `errors` in `n`: the largest rate under which so few errors would still
    occur with probability delta."""
    if n == 0:
        return 1.0
    if errors >= n:
        return 1.0
    lo, hi = errors / n, 1.0
    for _ in range(40):                                   # bisection
        mid = (lo + hi) / 2
        if _binom_cdf(errors, n, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def calibrate(log: list[tuple[float, bool]], alpha: float = 0.05, delta: float = 0.1,
              grid: int = 100) -> dict:
    """-> {"threshold", "accepted_fraction", "empirical_error", "error_bound", "curve"}

    log: (confidence, correct) pairs. If no threshold meets the target, threshold
    is None: with this much data nothing can be accepted safely.
    """
    curve, best = [], None
    for k in range(grid + 1):
        tau = k / grid
        acc = [c for conf, c in log if conf >= tau]
        errors = sum(1 for c in acc if not c)
        row = {"threshold": tau, "accepted_fraction": len(acc) / max(1, len(log)),
               "empirical_error": errors / len(acc) if acc else 0.0,
               "error_bound": error_bound(errors, len(acc), delta), "n": len(acc)}
        curve.append(row)
        if best is None and acc and row["error_bound"] <= alpha:
            best = row
    out = dict(best) if best else {"threshold": None, "accepted_fraction": 0.0,
                                   "empirical_error": None, "error_bound": None}
    out["curve"] = curve
    out["alpha"], out["delta"], out["n_log"] = alpha, delta, len(log)
    return out


def synthetic_log(n: int, seed: int = 0, sharpness: float = 14.0) -> list[tuple[float, bool]]:
    """A stand-in detection log: higher confidence means more often correct, but the
    detector is over-confident (0.8 confidence is right only about 87% of the time)."""
    import random
    rng = random.Random(seed)
    log = []
    for _ in range(n):
        conf = rng.betavariate(5, 1.6)
        p_correct = 1 / (1 + math.exp(-sharpness * (conf - 0.66)))
        log.append((conf, rng.random() < p_correct))
    return log
