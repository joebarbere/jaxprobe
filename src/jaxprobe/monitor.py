"""Watch a model's internals while it trains or serves: health statistics computed inside the compiled step,
running baselines that flag what moved, drift of SAE features against a saved baseline, and a JSON-lines writer.

Health statistics are capture reducers, so they run where the tap is (inside the layer scan) and come back as a
few numbers per layer next to the loss, never the activations themselves:

    probes = {"layers.resid": monitor.health(mask=mask), "layers.pattern": monitor.attention(mask=mask),
              "layers.mlp_act": monitor.units(mask=mask)}
    def step(params, opt, batch):
        def loss_fn(p):
            out, m = jaxprobe.run(model, p, batch["tokens"], capture=probes)
            return loss(out, batch), (m, monitor.logit_stats(out, mask=mask))
        (l, (m, lm)), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
        ...
        return params, opt, l, {"acts": m, "logits": lm}

    base = monitor.Running()
    for i, batch in enumerate(batches):
        params, opt, l, m = step(params, opt, batch)
        if i % 50 == 0:                                     # fetching syncs with the device: do it every N steps
            m = jax.device_get(m)
            writer.write(i, m, loss=float(l))
            print(base.flags(m, z=6.0))                    # what is far outside its own history
            base.update(m)

Nothing here calls back to the host from inside jit; the caller decides when to fetch. For TensorBoard or W&B,
`monitor.scalars(m)` flattens a metrics pytree to {"layers.resid/rms/3": 1.27, ...}.

Activations are (..., batch, T, d) as everywhere in jaxprobe; a mask (batch, T) marks real tokens.
"""
from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["JsonlWriter", "Running", "attention", "drift", "health", "logit_stats", "project", "read_jsonl",
           "scalars", "units"]


def _tok_weights(mask, shape):
    """Weights over the (…, batch, T) token axes of an activation of `shape` (…, batch, T, d)."""
    if mask is None:
        return jnp.ones(shape[:-1], jnp.float32)
    return jnp.broadcast_to(jnp.asarray(mask, jnp.float32), shape[:-1])


# ---------------------------------------------------------------- health of the residual stream and friends
def health(mask=None, outlier: float = 6.0) -> Callable:
    """Reducer for a residual-stream-like tap (…, batch, T, d) -> dict of scalars (per layer under a scan):

    rms           mean over real tokens of each token's RMS
    rms_max       the largest token RMS (attention sinks and massive activations show up here first)
    max_abs       the largest |x|
    outlier_dims  how many dimensions have mean |x| above `outlier` × the median dimension's (outlier features,
                  Dettmers et al. 2022); these grow during training and break low-precision inference
    nonfinite     count of NaN / Inf entries
    """
    def f(x):
        xf = x.astype(jnp.float32)
        w = _tok_weights(mask, x.shape)
        n = jnp.maximum(w.sum(), 1.0)
        finite = jnp.isfinite(xf)
        xz = jnp.where(finite, xf, 0.0)
        tok_rms = jnp.sqrt(jnp.mean(xz * xz, -1))
        per_dim = jnp.einsum("...d,...->d", jnp.abs(xz), w) / n
        return {"rms": (tok_rms * w).sum() / n,
                "rms_max": jnp.where(w > 0, tok_rms, 0.0).max(),
                "max_abs": jnp.where(w[..., None] > 0, jnp.abs(xz), 0.0).max(),
                "outlier_dims": (per_dim > outlier * jnp.median(per_dim)).sum().astype(jnp.int32),
                "nonfinite": (~finite).sum().astype(jnp.int32)}
    return f


def attention(mask=None, per_head: bool = True) -> Callable:
    """Reducer for attention probabilities (batch, …heads…, T, S) -> dict:

    entropy     mean entropy over keys, in nats (per head when per_head: the middle axes are kept); a head whose
                entropy collapses toward 0 attends to one key, the attention-logit-growth instability
                (Dehghani et al. 2023, Wortsman et al. 2023)
    max_prob    mean of each query's largest weight
    sink        mean weight on key 0 (the attention sink, Xiao et al. 2023)
    mask (batch, T) marks real query positions; padded keys should already have weight 0.
    """
    def f(p):
        pf = p.astype(jnp.float32)
        ent = -(pf * jnp.log(jnp.where(pf > 0, pf, 1.0))).sum(-1)              # (batch, …, T)
        B, T = p.shape[0], p.shape[-2]
        m = jnp.ones((B, T), jnp.float32) if mask is None else jnp.asarray(mask, jnp.float32)
        m = m.reshape((B,) + (1,) * (p.ndim - 3) + (T,))
        m = jnp.broadcast_to(m, ent.shape)
        def avg(v, keep):
            axes = (0, v.ndim - 1) if keep else tuple(range(v.ndim))
            return (v * m).sum(axes) / jnp.maximum(m.sum(axes), 1.0)
        return {"entropy": avg(ent, per_head), "max_prob": avg(pf.max(-1), False), "sink": avg(pf[..., 0], False)}
    return f


def units(mask=None, threshold: float = 0.0) -> Callable:
    """Reducer for hidden units (…, batch, T, H), e.g. MLP activations -> dict:

    dead      fraction of units never above `threshold` on any real token of this batch
    active    mean fraction of units above it per token
    """
    def f(h):
        on = (h.astype(jnp.float32) > threshold)
        w = _tok_weights(mask, h.shape)
        n = jnp.maximum(w.sum(), 1.0)
        fired = jnp.einsum("...h,...->h", on.astype(jnp.float32), w)
        return {"dead": (fired == 0).mean(), "active": fired.sum() / (n * h.shape[-1])}
    return f


def logit_stats(logits, mask=None) -> dict:
    """Output-logit health (…, batch, T, V) -> dict: max_logit, z (mean log-sum-exp; the z-loss of PaLM and
    Wortsman et al. 2023 is mean z²), entropy (mean, nats) and nonfinite. Drifting z or max_logit precedes the
    output-logit divergence instability."""
    lf = logits.astype(jnp.float32)
    w = _tok_weights(mask, lf.shape)
    n = jnp.maximum(w.sum(), 1.0)
    z = jax.nn.logsumexp(lf, -1)
    lp = lf - z[..., None]
    ent = -(jnp.exp(lp) * lp).sum(-1)
    return {"max_logit": jnp.where(w[..., None] > 0, lf, -jnp.inf).max(), "z": (z * w).sum() / n,
            "z2": (z * z * w).sum() / n, "entropy": (ent * w).sum() / n,
            "nonfinite": (~jnp.isfinite(lf)).sum().astype(jnp.int32)}


def project(directions, select: Callable | None = None, bias=None) -> Callable:
    """Reducer: activations · directions, for activation monitors (a refusal direction, a concept vector, a
    probe's weight vector). directions: (d,), (k, d) or {name: (d,)}; select picks positions first (default
    `jaxprobe.reduce.last`). Returns (…, batch) or (…, batch, k), or a dict of (…, batch) for a dict."""
    from . import reduce
    select = select or reduce.last
    named = isinstance(directions, Mapping)
    U = jnp.stack([jnp.asarray(v, jnp.float32) for v in directions.values()]) if named else jnp.asarray(
        directions, jnp.float32)
    b = 0.0 if bias is None else jnp.asarray(bias, jnp.float32)

    def f(x):
        s = select(x).astype(jnp.float32) @ (U.T if U.ndim == 2 else U) + b
        return {k: s[..., i] for i, k in enumerate(directions)} if named else s
    return f


# ---------------------------------------------------------------- running baselines
class Running:
    """Exponentially weighted mean and variance of every leaf of a metrics pytree (host side, NumPy), and flags
    for values far outside that history. decay=None keeps plain running averages (Welford)."""

    def __init__(self, decay: float | None = None, warmup: int = 5):
        self.decay, self.warmup = decay, warmup
        self.n = 0
        self.mean: Any = None
        self.var: Any = None

    def update(self, tree):
        x = jax.tree.map(lambda a: np.asarray(a, np.float64), tree)
        self.n += 1
        if self.mean is None:
            self.mean = x
            self.var = jax.tree.map(np.zeros_like, x)
            return self
        a = 1.0 / self.n if self.decay is None else 1.0 - self.decay
        self.var = jax.tree.map(lambda m, v, xi: (1 - a) * (v + a * (xi - m) ** 2), self.mean, self.var, x)
        self.mean = jax.tree.map(lambda m, xi: m + a * (xi - m), self.mean, x)
        return self

    def zscores(self, tree, floor: float = 1e-6):
        """(x − mean) / std per element, with std floored at `floor` × (|mean| + 1) so constant series don't
        divide by zero."""
        return jax.tree.map(lambda x, m, v: (np.asarray(x, np.float64) - m) / np.maximum(
            np.sqrt(v), floor * (np.abs(m) + 1.0)), tree, self.mean, self.var)

    def flags(self, tree, z: float = 6.0) -> dict[str, float]:
        """{flattened name: z-score} for every element more than z standard deviations from its history (empty
        during warmup)."""
        if self.mean is None or self.n < self.warmup:
            return {}
        return {k: v for k, v in scalars(self.zscores(tree)).items() if math.isfinite(v) and abs(v) > z}


# ---------------------------------------------------------------- SAE feature drift
def _rate_var(st, rate):
    """Variance of a feature's firing rate (fired / tokens) treating sequences, not tokens, as the samples: SAE
    features fire in bursts within a passage, so token-level binomial variance is far too small. Falls back to
    the binomial for Collector results without per-sequence sums."""
    n_tok, N = float(st["n_tokens"]), float(st["n_seqs"])
    binom = rate * (1 - rate) / max(n_tok, 1.0)
    if "fired_sq" not in st or N < 2:
        return binom
    ss = (np.asarray(st["fired_sq"], np.float64) - 2 * rate * np.asarray(st["fired_m"], np.float64)
          + rate * rate * float(st["tok_sq"]))
    mbar = n_tok / N
    return np.maximum(ss / (N * (N - 1) * mbar * mbar), binom)


def drift(base: Mapping, cur: Mapping, z: float = 5.0, min_expected: float = 5.0, top: int = 20) -> dict:
    """Compare two `jaxprobe.dash.Collector` results (a baseline and now) feature by feature.

    rate_z       z-score of the change in firing rate (+ = fires more now). The variance counts sequences as the
                 samples (a cluster-robust ratio estimate), because features fire in bursts; a token-level binomial
                 test flags thousands of features between two halves of the same text.
    log2_ratio   log2 of the density ratio, with half-count smoothing
    js           Jensen–Shannon divergence (bits) between the two activation-size histograms, for features that
                 fired at least min_expected times in both
    Summary: features significantly more / less active (|rate_z| > z) among those that fired in both, gone
    (fired in the baseline, not now, rate_z < −z), new (the reverse), and the `top` features by |rate_z|. A
    feature that fired in one passage only can't be significant, however many tokens it fired on."""
    nb, nc = float(base["n_tokens"]), float(cur["n_tokens"])
    fb, fc = np.asarray(base["fired"], np.float64), np.asarray(cur["fired"], np.float64)
    rb, rc = (fb + 0.5) / (nb + 1.0), (fc + 0.5) / (nc + 1.0)
    rate_z = (rc - rb) / np.sqrt(_rate_var(base, rb) + _rate_var(cur, rc))
    log2r = np.log2(rc / rb)
    hb, hc = np.asarray(base["hist"], np.float64), np.asarray(cur["hist"], np.float64)
    ok = (fb >= min_expected) & (fc >= min_expected)
    P = hb / np.maximum(hb.sum(1, keepdims=True), 1)
    Q = hc / np.maximum(hc.sum(1, keepdims=True), 1)
    M = 0.5 * (P + Q)

    def kl(a, b):
        return np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / np.where(b > 0, b, 1)), 0).sum(1)
    js = np.where(ok, 0.5 * kl(P, M) + 0.5 * kl(Q, M), np.nan)
    order = np.argsort(-np.abs(np.nan_to_num(rate_z)))[:top]
    return {"rate_z": rate_z, "log2_ratio": log2r, "js": js,
            "more": int(((rate_z > z) & (fb > 0)).sum()), "less": int(((rate_z < -z) & (fc > 0)).sum()),
            "gone": int(((fc == 0) & (fb > 0) & (rate_z < -z)).sum()),
            "new": int(((fb == 0) & (fc > 0) & (rate_z > z)).sum()),
            "median_js": float(np.nanmedian(js)) if ok.any() else float("nan"),
            "top": [(int(i), float(rate_z[i]), float(log2r[i])) for i in order]}


# ---------------------------------------------------------------- writing
def scalars(tree, prefix: str = "", sep: str = "/") -> dict[str, float]:
    """Flatten a metrics pytree to {name: float}; array elements get their index appended ("layers.resid/rms/3")."""
    out: dict[str, float] = {}

    def walk(t, name):
        if isinstance(t, Mapping):
            for k, v in t.items():
                walk(v, f"{name}{sep}{k}" if name else str(k))
        elif isinstance(t, list | tuple):
            for i, v in enumerate(t):
                walk(v, f"{name}{sep}{i}")
        else:
            a = np.asarray(t)
            if a.ndim == 0:
                out[name] = float(a)
            else:
                for idx in np.ndindex(a.shape):
                    out[name + "".join(f"{sep}{i}" for i in idx)] = float(a[idx])
    walk(tree, prefix)
    return out


def _finite(v):
    if isinstance(v, list):
        return [_finite(x) for x in v]
    return v if not isinstance(v, float) or math.isfinite(v) else None


def _plain(t):
    """Nested dicts / lists of plain numbers, floats rounded to 6 decimals, NaN and Inf as None."""
    if isinstance(t, Mapping):
        return {str(k): _plain(v) for k, v in t.items()}
    if isinstance(t, list | tuple):
        return [_plain(v) for v in t]
    a = np.asarray(t)
    if a.dtype.kind in "iub":
        return a.tolist()
    return _finite(np.round(a.astype(np.float64), 6).tolist())


class JsonlWriter:
    """Append one JSON line per call: {"step", "time", **extra, "metrics": nested lists}. Device arrays are
    fetched here, so call it every N steps, not every step. NaN/Inf become null."""

    def __init__(self, path, mode: str = "a"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open(mode, encoding="utf-8")

    def write(self, step: int, metrics, **extra):
        rec = {"step": int(step), "time": round(time.time(), 3), **{k: _plain(v) for k, v in extra.items()},
               "metrics": _plain(jax.device_get(metrics))}
        self._f.write(json.dumps(rec, allow_nan=False) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
