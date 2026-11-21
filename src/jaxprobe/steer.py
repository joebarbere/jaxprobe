"""Steering vectors and directional ablation, as patches for `jaxprobe.run` and as weight edits.

    u = jaxprobe.steer.mean_difference(pos_acts, neg_acts)            # CAA (Rimsky et al. 2023, arXiv 2312.06681)
    out, _ = jaxprobe.run(model, params, toks, patch={"layers.resid": jaxprobe.at(12, jaxprobe.steer.add(u, 4.0))})
    out, _ = jaxprobe.run(model, params, toks, patch={"layers.*": jaxprobe.steer.ablate(u)})   # remove u everywhere

Ablating a direction at runtime after every write to the residual stream equals editing the weights that write
there: W ← W − α (W u) uᵀ for each matrix whose output lands in the stream (`orthogonalize`). Arditi et al. 2024
(arXiv 2406.11717) used this to remove a refusal direction; it's shown here on benign directions.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def unit(v):
    v = jnp.asarray(v, jnp.float32)
    return v / jnp.linalg.norm(v)


def mean_difference(pos, neg, normalize: bool = False):
    """Contrastive activation addition: mean(pos) − mean(neg) over the leading axis (N, d) → (d,)."""
    v = jnp.asarray(pos, jnp.float32).mean(0) - jnp.asarray(neg, jnp.float32).mean(0)
    return unit(v) if normalize else v


def actadd(pos_act, neg_act):
    """ActAdd (Turner et al. 2023, arXiv 2308.10248): the difference of two prompts' activations at one position."""
    return jnp.asarray(pos_act, jnp.float32) - jnp.asarray(neg_act, jnp.float32)


def add(v, coeff: float = 1.0, positions=None):
    """Patch: x + coeff·v, at every position or at a boolean mask / index array over the sequence axis (-2)."""
    v = jnp.asarray(v)

    def patch(x):
        y = x + (coeff * v).astype(x.dtype)
        if positions is None:
            return y
        m = jnp.zeros(x.shape[-2], bool).at[positions].set(True)
        return jnp.where(m[:, None], y, x)
    return patch


def ablate(u, alpha: float = 1.0):
    """Patch: x − α (x·û) û. α = 1 removes the component along u; α = 2 reflects it."""
    u = unit(u)

    def patch(x):
        xf = x.astype(jnp.float32)
        return (xf - alpha * (xf @ u)[..., None] * u).astype(x.dtype)
    return patch


def orthogonalize(W, u, alpha: float = 1.0):
    """W − α (W û) ûᵀ for a matrix W (..., d) whose rows write to the residual stream: afterwards nothing W writes
    has a component along u (α = 1). Batched over leading axes (e.g. stacked layers)."""
    u = unit(u)
    Wf = W.astype(jnp.float32)
    return (Wf - alpha * (Wf @ u)[..., None] * u).astype(W.dtype)


def sweep(score, layers, coeffs):
    """Evaluate score(layer, coeff) -> dict of floats over a grid; returns a list of dicts with layer and coeff.
    Score behaviour *and* damage (e.g. KL on unrelated text) in the same dict, and choose on both."""
    rows = []
    for layer in layers:
        for c in coeffs:
            r = {k: float(v) for k, v in score(layer, c).items()}
            rows.append({"layer": int(layer), "coeff": float(c), **r})
    return rows


def kl(p_logits, q_logits):
    """KL(p ‖ q) per position in nats (p = clean, q = steered: the usual damage measure)."""
    lp, lq = jax.nn.log_softmax(p_logits.astype(jnp.float32)), jax.nn.log_softmax(q_logits.astype(jnp.float32))
    return (jnp.exp(lp) * (lp - lq)).sum(-1)
