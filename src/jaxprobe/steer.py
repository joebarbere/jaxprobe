"""Steering vectors and directional ablation, as patches for `jaxprobe.run` and as weight edits.

    u = jaxprobe.steer.mean_difference(pos_acts, neg_acts)            # CAA (Rimsky et al. 2023, arXiv 2312.06681)
    out, _ = jaxprobe.run(model, params, toks, patch={"layers.resid": jaxprobe.at(12, jaxprobe.steer.add(u, 4.0))})
    out, _ = jaxprobe.run(model, params, toks, patch={"layers.*": jaxprobe.steer.ablate(u)})   # remove u everywhere

Ablating a direction at runtime after every write to the residual stream equals editing the weights that write
there: W ← W − α (W u) uᵀ for each matrix whose output lands in the stream (`orthogonalize`). Arditi et al. 2024
(arXiv 2406.11717) used this to remove a refusal direction; `jaxprobe.directions` packages that recipe.

`add` and `ablate` return pytrees, so under `jaxprobe.bind` their vectors and coefficients are traced: one compile
for a whole sweep, and `jax.vmap` over the coefficient works.
"""
from __future__ import annotations

import dataclasses

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


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Add:
    """Patch: x + coeff·v, at every position or at the given positions of the sequence axis (-2)."""
    v: jax.Array
    coeff: jax.Array
    positions: jax.Array | None = None

    def __call__(self, x):
        y = x + (self.coeff * self.v).astype(x.dtype)
        if self.positions is None:
            return y
        m = jnp.zeros(x.shape[-2], bool).at[self.positions].set(True)
        return jnp.where(m[:, None], y, x)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Ablate:
    """Patch: x − α (x·û) û. α = 1 removes the component along u; α = 2 reflects it."""
    u: jax.Array
    alpha: jax.Array

    def __call__(self, x):
        u = unit(self.u)
        xf = x.astype(jnp.float32)
        return (xf - self.alpha * (xf @ u)[..., None] * u).astype(x.dtype)


def add(v, coeff=1.0, positions=None) -> Add:
    """Patch: x + coeff·v, at every position or at a boolean mask / index array over the sequence axis (-2)."""
    return Add(jnp.asarray(v), jnp.asarray(coeff, jnp.float32), None if positions is None else jnp.asarray(positions))


def ablate(u, alpha=1.0) -> Ablate:
    """Patch: x − α (x·û) û. α = 1 removes the component along u; α = 2 reflects it."""
    return Ablate(jnp.asarray(u, jnp.float32), jnp.asarray(alpha, jnp.float32))


def orthogonalize(W, u, alpha: float = 1.0, axis: int = -1):
    """W − α (W û) ûᵀ along `axis`, the axis of W that lands in the residual stream (the last one for JAX-style
    (in, d) matrices, embeddings and biases; 0 for PyTorch-style (d, in) weights). Afterwards nothing W writes has
    a component along u (α = 1). Batched over the other axes (e.g. stacked layers)."""
    u = unit(u)
    Wf = jnp.moveaxis(jnp.asarray(W).astype(jnp.float32), axis, -1)
    out = Wf - alpha * (Wf @ u)[..., None] * u
    return jnp.moveaxis(out, -1, axis).astype(W.dtype)


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
