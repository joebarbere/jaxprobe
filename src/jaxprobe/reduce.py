"""Reducers for `capture={pattern: reducer}`: they run where the tap is (inside the scan body), so a run over a
long sequence and many layers returns only what you keep, e.g. (layers, batch, d) instead of (layers, batch, T, d).

    _, acts = jaxprobe.run(model, params, toks, capture={"layers.resid": jaxprobe.reduce.last})
    _, acts = jaxprobe.run(model, params, toks,
                           capture={"layers.resid": jaxprobe.reduce.compose(jaxprobe.reduce.positions([-3, -2, -1]),
                                                                            jaxprobe.reduce.batch_sum())})

Activations are taken to be (..., batch, T, d): batch is axis -3 and the sequence axis -2, as in `steer.add`.
"""
from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp


def last(x):
    """The last position: (..., T, d) -> (..., d)."""
    return x[..., -1, :]


def positions(idx) -> Callable:
    """Positions idx (an int or a list of ints, negative counts from the end): (..., T, d) -> (..., P, d)."""
    idx = jnp.atleast_1d(jnp.asarray(idx, jnp.int32))
    return lambda x: jnp.take(x, idx, axis=-2)


def masked_mean(mask) -> Callable:
    """Mean over the positions where mask (batch, T) is true: (..., batch, T, d) -> (..., batch, d), in float32."""
    m = jnp.asarray(mask, jnp.float32)

    def f(x):
        s = jnp.einsum("...btd,bt->...bd", x.astype(jnp.float32), m)
        return s / jnp.maximum(m.sum(-1), 1.0)[:, None]
    return f


def batch_sum(weights=None, axis: int = 0) -> Callable:
    """Sum over the batch axis in float32, each row weighted (weights (batch,), e.g. 0 for padding rows)."""
    def f(x):
        xf = x.astype(jnp.float32)
        if weights is None:
            return xf.sum(axis)
        w = jnp.asarray(weights, jnp.float32)
        w = w.reshape(w.shape + (1,) * (xf.ndim - axis % xf.ndim - 1))
        return jnp.where(w > 0, xf * w, 0.0).sum(axis)  # a NaN in a weight-0 row doesn't leak into the sum
    return f


def astype(dtype) -> Callable:
    return lambda x: x.astype(dtype)


def compose(*fns) -> Callable:
    """compose(f, g)(x) = g(f(x)): reducers apply left to right."""
    def f(x):
        for fn in fns:
            x = fn(x)
        return x
    return f
