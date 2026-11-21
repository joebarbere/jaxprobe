"""Sparse autoencoders on activations: ReLU + L1, TopK (+ AuxK), BatchTopK and JumpReLU.

An SAE is a dict of arrays: W_enc (d, n), b_enc (n,), W_dec (n, d), b_dec (d,), plus log_threshold (n,) for JumpReLU
and threshold () for BatchTopK at inference. Activations go in scaled to mean squared norm 1 (`norm_scale`), so loss
coefficients don't depend on the layer's scale.

    sae = jaxprobe.sae.init(key, d=576, n=8192)
    scale = jaxprobe.sae.norm_scale(acts)
    (loss, aux), grads = jax.value_and_grad(jaxprobe.sae.loss, has_aux=True)(sae, acts * scale, kind="topk", k=32)

References: Bricken et al. 2023 (ReLU + L1); Gao et al. 2024, arXiv 2406.04093 (TopK, AuxK); Bussmann et al. 2024,
arXiv 2412.06410 (BatchTopK); Rajamanoharan et al. 2024, arXiv 2407.14435 (JumpReLU).
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

KINDS = ("relu", "topk", "batchtopk", "jumprelu")


def init(key, d: int, n: int, *, jumprelu_threshold: float = 0.001):
    """Unit-norm decoder rows; the encoder starts as the decoder's transpose (Gao et al.)."""
    W_dec = jax.random.normal(key, (n, d), jnp.float32)
    W_dec = W_dec / jnp.linalg.norm(W_dec, axis=1, keepdims=True)
    return {"W_enc": W_dec.T, "b_enc": jnp.zeros(n, jnp.float32), "W_dec": W_dec, "b_dec": jnp.zeros(d, jnp.float32),
            "log_threshold": jnp.full(n, jnp.log(jumprelu_threshold), jnp.float32)}


def pre(sae, x):
    return (x - sae["b_dec"]) @ sae["W_enc"] + sae["b_enc"]


def topk_relu(p, k: int):
    """Keep each row's k largest values (after a ReLU), zero the rest. A threshold mask rather than a scatter of
    top_k's indices: same result barring exact ties, and faster on GPUs where scatter is slow."""
    kth = jax.lax.top_k(jax.lax.stop_gradient(p), k)[0][..., -1:]
    return jnp.where(p >= kth, jax.nn.relu(p), 0.0)


def batch_topk_relu(p, k: int):
    """Keep the k·N largest values across the whole batch of N rows: k active per row on average."""
    flat = jax.lax.stop_gradient(p).reshape(-1)
    kth = jax.lax.top_k(flat, k * p.shape[0])[0][-1]
    return jnp.where(p >= kth, jax.nn.relu(p), 0.0)


# JumpReLU: z = x·H(x − θ). The threshold θ gets gradients through a rectangle-kernel pseudo-derivative of the step
# (bandwidth eps), as in Rajamanoharan et al. 2024.
def _rect(u):
    return (jnp.abs(u) < 0.5).astype(u.dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def jumprelu(x, theta, eps: float = 0.001):
    return x * (x > theta)


def _jr_fwd(x, theta, eps):
    return jumprelu(x, theta, eps), (x, theta)


def _jr_bwd(eps, res, g):
    x, theta = res
    dx = (x > theta) * g
    dtheta = -(theta / eps) * _rect((x - theta) / eps) * g
    return dx, dtheta.reshape(-1, theta.shape[-1]).sum(0)


jumprelu.defvjp(_jr_fwd, _jr_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def step(x, theta, eps: float = 0.001):
    """Heaviside H(x − θ), with a pseudo-derivative for θ (used for JumpReLU's L0 penalty)."""
    return (x > theta).astype(x.dtype)


def _st_fwd(x, theta, eps):
    return step(x, theta, eps), (x, theta)


def _st_bwd(eps, res, g):
    x, theta = res
    dtheta = -(1.0 / eps) * _rect((x - theta) / eps) * g
    return jnp.zeros_like(x), dtheta.reshape(-1, theta.shape[-1]).sum(0)


step.defvjp(_st_fwd, _st_bwd)


def encode(sae, x, kind: str = "topk", k: int = 32, *, train: bool = False, eps: float = 0.001):
    """Sparse code z (N, n). BatchTopK uses the batch's own threshold when train=True and the stored
    sae["threshold"] otherwise (see `batchtopk_threshold`)."""
    p = pre(sae, x)
    if kind == "topk":
        return topk_relu(p, k)
    if kind == "batchtopk":
        if train or "threshold" not in sae:
            return batch_topk_relu(p, k)
        return jnp.where(p > sae["threshold"], jax.nn.relu(p), 0.0)
    if kind == "jumprelu":
        return jumprelu(jax.nn.relu(p), jnp.exp(sae["log_threshold"]), eps)
    if kind == "relu":
        return jax.nn.relu(p)
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def decode(sae, z):
    return z @ sae["W_dec"] + sae["b_dec"]


def loss(sae, x, dead=None, *, kind: str = "topk", k: int = 32, l1: float = 0.0, l0: float = 0.0,
         k_aux: int = 256, aux_weight: float = 1 / 32, eps: float = 0.001):
    """Training loss for activations x (N, d), already scaled with `norm_scale`. Returns (total, metrics).

    relu:      MSE + l1 · Σ z_i ‖W_dec,i‖
    topk:      MSE + aux_weight · AuxK (the residual reconstructed by the k_aux largest *dead* latents)
    batchtopk: same as topk, with the k·N largest across the batch
    jumprelu:  MSE + l0 · Σ H(pre_i − θ_i), the L0 penalty with a straight-through threshold gradient
    dead: bool (n,) latents that haven't fired recently (for AuxK); None disables AuxK."""
    p = pre(sae, x)
    z = encode(sae, x, kind, k, train=True, eps=eps)
    x_hat = decode(sae, z)
    mse = ((x - x_hat) ** 2).sum(-1).mean()
    aux = jnp.float32(0)
    if kind in ("topk", "batchtopk") and dead is not None:
        dead_p = jnp.where(dead, p, -1e9)
        z_aux = jnp.where(dead, topk_relu(dead_p, min(k_aux, p.shape[-1])), 0.0)
        aux = ((jax.lax.stop_gradient(x - x_hat) - z_aux @ sae["W_dec"]) ** 2).sum(-1).mean()
        aux = jnp.where(dead.any(), aux, 0.0) * aux_weight
    if kind == "relu":
        aux = l1 * (z * jnp.linalg.norm(sae["W_dec"], axis=1)).sum(-1).mean()
    if kind == "jumprelu":
        aux = l0 * step(jax.nn.relu(p), jnp.exp(sae["log_threshold"]), eps).sum(-1).mean()
    fired = z > 0
    return mse + aux, {"mse": mse, "aux": aux, "l0": fired.sum(-1).mean(), "fired": fired.any(0),
                       "min_active": jnp.where(fired, z, jnp.inf).min()}


def renorm_decoder(sae):
    """Put decoder rows back at unit length (TopK-family SAEs; it stops the encoder from shrinking codes)."""
    W = sae["W_dec"]
    return dict(sae, W_dec=W / jnp.linalg.norm(W, axis=1, keepdims=True))


def batchtopk_threshold(sae, xs, k: int):
    """The inference threshold for a BatchTopK SAE: the mean over batches of the smallest kept activation
    (Bussmann et al.). xs: iterable of (N, d) batches."""
    mins = [float(loss(sae, x, kind="batchtopk", k=k)[1]["min_active"]) for x in xs]
    return dict(sae, threshold=jnp.float32(sum(mins) / len(mins)))


def norm_scale(acts):
    """Scalar s with mean ‖s·x‖² = 1."""
    acts = jnp.asarray(acts, jnp.float32)
    return 1.0 / jnp.sqrt((acts**2).sum(-1).mean())


# ---------------------------------------------------------------- metrics
def fvu(x, x_hat):
    """Fraction of variance unexplained: Σ‖x − x̂‖² / Σ‖x − mean(x)‖²."""
    return ((x - x_hat) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()


def loss_recovered(clean: float, spliced: float, ablated: float) -> float:
    """How much of the language-model loss the SAE keeps when spliced in: 1 at clean, 0 at the ablated baseline."""
    return (ablated - spliced) / (ablated - clean)


def splice(sae, kind: str = "topk", k: int = 32, scale: float = 1.0, keep=None):
    """A patch (for `jaxprobe.run`) that replaces an activation by its SAE reconstruction. `scale` is the factor
    the SAE was trained with. keep: optional fn(x) -> bool mask (..., ) of positions to leave alone (e.g. attention
    sinks with huge norms that the SAE wasn't trained on)."""
    def patch(x):
        xf = x.astype(jnp.float32)
        rec = decode(sae, encode(sae, xf * scale, kind, k)) / scale
        if keep is not None:
            rec = jnp.where(keep(xf)[..., None], xf, rec)
        return rec.astype(x.dtype)
    return patch


def density(fired_counts, n_tokens: int):
    """Firing frequency per latent; dead = 0, dense = above ~0.1 in the usual reading."""
    return jnp.asarray(fired_counts) / n_tokens
