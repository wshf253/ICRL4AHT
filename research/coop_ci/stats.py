"""Bootstrap helpers (the resampling unit is whatever the caller passes: seeds, states, streams)."""
from __future__ import annotations

import random
from statistics import mean


def boot_ci(xs, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    ms = sorted(mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return ms[int(alpha / 2 * n)], ms[int((1 - alpha / 2) * n) - 1]


def paired_diff(a: dict, b: dict, **kw):
    """Mean and bootstrap CI of a[k] - b[k] over keys present in both."""
    keys = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in keys]
    if not d:
        return None, (float("nan"), float("nan")), 0
    return mean(d), boot_ci(d, **kw), len(d)


def fmt(x, d=1):
    return "-" if x is None or x != x else f"{x:.{d}f}"
