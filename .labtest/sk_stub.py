"""Inject a minimal but NUMERICALLY REAL sklearn into sys.modules.

Metrics and isotonic regression are genuine implementations, so plumbing results
are trustworthy. HistGradientBoostingClassifier is aliased to LogisticRegression
because a real GBM is out of scope here — this harness validates WIRING, not
model quality. Anything that depends on tree behaviour must be run on Modal.
"""
import sys, types, importlib.machinery
import numpy as np


# ───────────────────────── metrics ─────────────────────────
def roc_auc_score(y, p):
    y = np.asarray(y).astype(float).ravel(); p = np.asarray(p, dtype=float).ravel()
    pos, neg = y > 0.5, y <= 0.5
    n1, n0 = pos.sum(), neg.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.empty(len(p))                      # average ranks, ties handled
    o = np.argsort(p, kind="mergesort")
    ps = p[o]
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _pr_ascending(y, p):
    """Precision/recall in INCREASING-recall order (descending score), plus the
    matching descending thresholds. This is the natural order for AP."""
    y = (np.asarray(y).ravel() > 0.5).astype(int)
    p = np.asarray(p, dtype=float).ravel()
    o = np.argsort(-p, kind="mergesort")
    ys, ps = y[o], p[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    keep = np.r_[np.diff(ps) != 0, True]      # last index of each distinct score
    tp, fp, th = tp[keep], fp[keep], ps[keep]
    # trim everything after recall first reaches its maximum, as sklearn does
    last = int(np.searchsorted(tp, tp[-1])) + 1
    tp, fp, th = tp[:last], fp[:last], th[:last]
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / max(int(y.sum()), 1)
    return prec, rec, th


def precision_recall_curve(y, p):
    """sklearn convention: recall DECREASING, final point (precision 1, recall 0),
    thresholds increasing and one shorter than precision/recall."""
    prec, rec, th = _pr_ascending(y, p)
    return (np.r_[prec[::-1], 1.0], np.r_[rec[::-1], 0.0], th[::-1])


def average_precision_score(y, p):
    prec, rec, _ = _pr_ascending(y, p)
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def f1_score(y, pred, zero_division=0):
    y = (np.asarray(y).ravel() > 0.5).astype(int)
    pred = (np.asarray(pred).ravel() > 0.5).astype(int)
    tp = int((y & pred).sum()); fp = int(((1 - y) & pred).sum())
    fn = int((y & (1 - pred)).sum())
    return float(zero_division) if 2 * tp + fp + fn == 0 else 2.0 * tp / (2 * tp + fp + fn)


def confusion_matrix(y, pred, labels=(0, 1)):
    y = (np.asarray(y).ravel() > 0.5).astype(int)
    pred = (np.asarray(pred).ravel() > 0.5).astype(int)
    return np.array([[int(((1 - y) & (1 - pred)).sum()), int(((1 - y) & pred).sum())],
                     [int((y & (1 - pred)).sum()), int((y & pred).sum())]])


# ───────────────────────── models ─────────────────────────
class LogisticRegression:
    def __init__(self, max_iter=300, class_weight=None, **kw):
        self.max_iter, self.class_weight = max_iter, class_weight

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64); y = (np.asarray(y).ravel() > 0.5).astype(float)
        n, d = X.shape
        w = np.ones(n)
        if self.class_weight == "balanced":
            p = max(y.sum(), 1.0)
            w = np.where(y > 0, n / (2 * p), n / (2 * max(n - p, 1.0)))
        self.coef_ = np.zeros(d); self.intercept_ = 0.0
        lr = 0.5
        for _ in range(self.max_iter):
            z = X @ self.coef_ + self.intercept_
            q = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = (w * (q - y))
            self.coef_ -= lr * (X.T @ g) / n
            self.intercept_ -= lr * g.sum() / n
        return self

    def predict_proba(self, X):
        z = np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_
        q = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return np.c_[1 - q, q]


class IsotonicRegression:
    """Real pool-adjacent-violators, weighted, increasing."""
    def __init__(self, out_of_bounds="clip", y_min=None, y_max=None, increasing=True):
        self.y_min, self.y_max = y_min, y_max

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        o = np.argsort(x, kind="mergesort")
        xs, ys = x[o], y[o]
        val = [ys[0]]; wt = [1.0]; xr = [xs[0]]
        for i in range(1, len(ys)):
            val.append(ys[i]); wt.append(1.0); xr.append(xs[i])
            while len(val) > 1 and val[-2] > val[-1]:
                v2, w2 = val.pop(), wt.pop()
                v1, w1 = val.pop(), wt.pop()
                val.append((v1 * w1 + v2 * w2) / (w1 + w2)); wt.append(w1 + w2)
                xr.pop(-2)
        self._x = np.array(xr); self._y = np.array(val)
        if self.y_min is not None or self.y_max is not None:
            self._y = np.clip(self._y, self.y_min if self.y_min is not None else -np.inf,
                              self.y_max if self.y_max is not None else np.inf)
        return self

    def predict(self, x):
        return np.interp(np.asarray(x, dtype=np.float64).ravel(), self._x, self._y)


class Ridge:
    """Closed-form ridge, intercept fitted by centering. Exact, not iterative.

    Used only by mem_leak_probe, which asks "how much of log(event count) is a
    LINEAR readout of the node memory". A closed-form solve is the same estimator
    sklearn uses for this configuration, so unlike the classifier stubs above this
    one is faithful rather than a stand-in.
    """
    def __init__(self, alpha=1.0, **kw): self.alpha = float(alpha)

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mx_, self.my_ = X.mean(0), y.mean()
        Xc = X - self.mx_
        d = Xc.shape[1]
        self.coef_ = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(d),
                                     Xc.T @ (y - self.my_))
        return self

    def predict(self, X):
        return (np.asarray(X, float) - self.mx_) @ self.coef_ + self.my_


def r2_score(y_true, y_pred):
    y = np.asarray(y_true, float); p = np.asarray(y_pred, float)
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def install():
    sk = types.ModuleType("sklearn")
    lm = types.ModuleType("sklearn.linear_model"); lm.LogisticRegression = LogisticRegression
    lm.Ridge = Ridge
    en = types.ModuleType("sklearn.ensemble")
    en.HistGradientBoostingClassifier = LogisticRegression   # STUB, see docstring
    me = types.ModuleType("sklearn.metrics")
    for f in (roc_auc_score, precision_recall_curve, average_precision_score,
              f1_score, confusion_matrix, r2_score):
        setattr(me, f.__name__, f)
    iso = types.ModuleType("sklearn.isotonic"); iso.IsotonicRegression = IsotonicRegression
    sk.linear_model, sk.ensemble, sk.metrics, sk.isotonic = lm, en, me, iso
    for n, m in (("sklearn", sk), ("sklearn.linear_model", lm), ("sklearn.ensemble", en),
                 ("sklearn.metrics", me), ("sklearn.isotonic", iso)):
        # torch._dynamo.trace_rules walks sys.modules calling importlib find_spec,
        # which raises "ValueError: sklearn.__spec__ is None" on a bare ModuleType.
        m.__spec__ = importlib.machinery.ModuleSpec(n, None)
        m.__path__ = []
        sys.modules[n] = m
