"""Offline validation of modal_lab.py.

Drives functions from modal_lab against synthetic NPZ fixtures in temporary
storage. Uses the actual scikit-learn installed in the selected environment.
This helper provides no certification of causality or CTU-13 accuracy.
"""
import os, sys, types, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import sklearn  # Fail on a missing dependency instead of substituting fake estimators.
import numpy as np

# Keep generated fixtures away from preserved inputs. Keep the TemporaryDirectory
# alive while the module is loaded; tests may explicitly override DATA with a temp path.
_FIXTURE_ROOT = tempfile.TemporaryDirectory(prefix="tgdetect-fixtures-")
ROOT = pathlib.Path(_FIXTURE_ROOT.name)
DATA = ROOT / "real" / "ctu13"


def make_fixtures(n=30000, seed=0):
    """3 fake captures. Cumulative columns ramp with stream position, and the
    label prevalence ramps from ~0.1% early to ~8% late, reproducing the head
    bias and the train->test prevalence jump seen on the real corpus."""
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for k in ("s01", "s02", "s03"):
        idx = np.arange(n)
        frac = idx / n
        X = rng.normal(0, 1, (n, 22)).astype(np.float32)
        for c in (13, 14, 17, 18, 19):                 # cumulative: ramp + noise
            X[:, c] = (np.log1p(idx * (1 + 0.5 * rng.random())) / 3.0
                       + rng.normal(0, 0.3, n)).astype(np.float32)
        for c in (7, 8, 9, 20, 21):                    # binary flags
            X[:, c] = (rng.random(n) < 0.4).astype(np.float32)
        p = 0.001 + 0.08 * frac ** 3                   # prevalence ramps late
        y = (rng.random(n) < p).astype(np.float32)
        X[y > 0, 0] += 1.2                             # a real, stationary signal
        X[y > 0, 5] -= 0.9
        ts = np.sort(rng.random(n) * 86400.0).astype(np.float32)
        # int32, MATCHING modal_data_real.py's writer. This is not cosmetic: when
        # the fixture wrote int64 it silently satisfied index_copy_'s Long
        # requirement, and the real int32 data raised
        #   "index_copy_(): Expected a long tensor for index, but got Int"
        # only on Modal. A fixture that does not reproduce the real dtypes cannot
        # catch dtype bugs, which are exactly the bugs an offline harness is for.
        # Node ids MUST mimic modal_data_real.py, which does
        #     node_id.setdefault(ip, len(node_id))
        # i.e. ids are assigned per capture in ORDER OF FIRST APPEARANCE. So they
        # GROW with stream position and their maximum EXCEEDS every fixed-size table
        # in this repo (NODE_BUCKETS = 1<<17 = 131,072).
        #
        # The old fixture drew ids from [0,400) and [0,900). Those fit inside every
        # table, so the fixture was structurally incapable of catching the
        # out-of-bounds memory indexing that surfaced on Modal as
        #   "RuntimeError: CUDA error: device-side assert triggered"
        # the moment WINDOW flipped from "head" to "tail". That is the SECOND bug of
        # exactly this shape (the first was int32 vs int64 src/dst). Lesson, now
        # encoded: a fixture must reproduce the real data's RANGES, not just its
        # shapes and dtypes.
        hi = 200_000                                    # deliberately > NODE_BUCKETS
        frontier = (frac * hi).astype(np.int64) + 2      # ids seen so far
        src = np.clip(frontier - rng.integers(0, 50, n), 0, hi)          # recent host
        dst = (rng.random(n) * frontier).astype(np.int64)                # any earlier
        src = src.astype(np.int32)
        dst = np.clip(dst, 0, hi).astype(np.int32)
        np.savez(DATA / f"{k}.npz", X=X, ts=ts, src=src, dst=dst, y=y)
    print(f"fixtures: 3 captures x {n:,d} events -> {DATA}")


def patch():
    import modal_lab as L
    make_fixtures()
    L.DATA_DIR = str(DATA)
    dummy = types.SimpleNamespace(commit=lambda: None)
    L.volume = dummy
    import modal_common as C
    C.DATA_DIR = str(DATA)
    # FAMILY must only contain the fixtures we actually wrote
    L.FAMILY = {"s01": "Neris", "s02": "Rbot", "s03": "Virut"}
    return L


def raw(cls, name):
    """Unwrap a modal-decorated method back to the plain function."""
    f = cls.__dict__[name]
    for attr in ("_get_raw_f", "raw_f"):
        g = getattr(f, attr, None)
        if g is not None:
            return g() if callable(g) and attr == "_get_raw_f" else g
    return f


def build(L, which, **params):
    uc = getattr(L, which)._get_user_cls()
    obj = object.__new__(uc)
    for k, v in params.items():
        setattr(obj, k, v)
    raw(uc, "setup")(obj)
    return uc, obj
