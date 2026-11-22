"""Linear probes: logistic regression on activations, e.g. to flag dataset rows by what the model represents rather
than by keywords.

    pr = jaxprobe.probe.fit_logistic(acts, labels, l2=1e-3)     # acts (N, d), labels (N,) in {0, 1}
    s = jaxprobe.probe.score(pr, new_acts)                       # logits; > 0 predicts 1
    jaxprobe.probe.auc(s, new_labels)

Features are standardised with the training mean and std; the fit is full-batch BFGS (jax.scipy.optimize), fine
for d up to a few thousand and N up to ~10⁵ (subsample beyond that).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.optimize import minimize


def fit_logistic(x, y, l2: float = 1e-3, maxiter: int = 500, weights=None) -> dict:
    """Minimise mean weighted log-loss + l2/2 ‖w‖² on standardised x. Returns {w, b, mean, std, converged}."""
    x = jnp.asarray(x, jnp.float32)
    y = jnp.asarray(y, jnp.float32)
    sw = jnp.ones_like(y) if weights is None else jnp.asarray(weights, jnp.float32)
    mean, std = x.mean(0), x.std(0) + 1e-6
    z = (x - mean) / std
    d = x.shape[1]

    def loss(theta):
        w, b = theta[:d], theta[d]
        logit = z @ w + b
        ll = jnp.logaddexp(0.0, logit) - y * logit  # −log p(y | x), stable
        return (sw * ll).sum() / sw.sum() + 0.5 * l2 * (w * w).sum()

    res = minimize(jax.jit(loss), jnp.zeros(d + 1), method="BFGS", options={"maxiter": maxiter})
    return {"w": res.x[:d] / std, "b": res.x[d] - (res.x[:d] / std) @ mean, "mean": mean, "std": std,
            "converged": bool(res.success)}


def score(probe: dict, x):
    """Logits w·x + b (in the original, unstandardised coordinates)."""
    return jnp.asarray(x, jnp.float32) @ probe["w"] + probe["b"]


def direction(probe: dict):
    """The probe's unit direction in activation space (compare with a difference-in-means direction)."""
    return probe["w"] / jnp.linalg.norm(probe["w"])


def auc(scores, labels) -> float:
    """Area under the ROC curve: the chance a random positive scores above a random negative (ties count half)."""
    s, y = np.asarray(scores, np.float64), np.asarray(labels).astype(bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        raise ValueError("auc needs both classes")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # average ranks over ties
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
