"""jaxprobe.sae: encoders, losses, gradients and metrics on random data."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxprobe import sae as S

D, N = 16, 64
X = jax.random.normal(jax.random.key(4), (128, D))
X = X * S.norm_scale(X)


def test_init_and_scale():
    s = S.init(jax.random.key(3), D, N)
    np.testing.assert_allclose(jnp.linalg.norm(s["W_dec"], axis=1), 1.0, rtol=1e-5)
    assert float((X**2).sum(-1).mean()) == pytest.approx(1.0, rel=1e-5)


@pytest.mark.parametrize("kind", S.KINDS)
def test_loss_and_grads_are_finite(kind):
    s = S.init(jax.random.key(3), D, N)
    dead = jnp.zeros(N, bool).at[:10].set(True)
    (tot, m), g = jax.value_and_grad(S.loss, has_aux=True)(s, X, dead, kind=kind, k=5, l1=1e-2, l0=1e-3)
    assert np.isfinite(float(tot))
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(g))
    if kind == "topk":
        assert float(m["l0"]) <= 5 and float(m["aux"]) > 0
    if kind == "batchtopk":
        assert float(m["l0"]) == pytest.approx(5, abs=0.5)
    if kind == "jumprelu":
        assert float(jnp.abs(g["log_threshold"]).sum()) > 0  # the threshold learns


def test_topk_keeps_k_per_row():
    z = S.encode(S.init(jax.random.key(3), D, N), X, "topk", 5)
    assert int((z > 0).sum(1).max()) <= 5 and float(z.min()) >= 0


def test_batchtopk_inference_threshold():
    s = S.init(jax.random.key(3), D, N)
    s = S.batchtopk_threshold(s, [X[:64], X[64:]], k=5)
    z = S.encode(s, X, "batchtopk", 5)
    assert 2 < float((z > 0).sum(1).mean()) < 10


def test_jumprelu_vjp_matches_definition():
    x = jnp.array([[1.0, 1.0, 2.0]])
    th = jnp.array([1.0002, 0.5, 3.0])
    _, vjp = jax.vjp(lambda a, t: S.jumprelu(a, t, 0.001), x, th)
    dx, dth = vjp(jnp.ones_like(x))
    np.testing.assert_array_equal(dx, [[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(dth, [-1.0002 / 0.001, 0.0, 0.0], rtol=1e-5)  # only x within eps/2 of θ


def test_training_reduces_loss_and_metrics():
    s = S.init(jax.random.key(3), D, N)
    lr = 1e-2
    step = jax.jit(lambda s: jax.tree.map(lambda p, g: p - lr * g, s,
                                         jax.grad(lambda s: S.loss(s, X, kind="topk", k=8)[0])(s)))
    before = float(S.loss(s, X, kind="topk", k=8)[1]["mse"])
    for _ in range(200):
        s = S.renorm_decoder(step(s))
    after = float(S.loss(s, X, kind="topk", k=8)[1]["mse"])
    assert after < 0.7 * before
    x_hat = S.decode(s, S.encode(s, X, "topk", 8))
    assert 0 < float(S.fvu(X, x_hat)) < 1
    assert float(S.fvu(X, X)) == 0.0
    assert S.loss_recovered(3.0, 3.5, 8.0) == pytest.approx(0.9)


def test_splice_patch():
    s = S.init(jax.random.key(3), D, N)
    x = X[None] * 3.0  # (1, T, D) at a different scale
    keep = lambda a: jnp.linalg.norm(a, axis=-1) > 100.0
    y = S.splice(s, "topk", 8, scale=1 / 3.0, keep=keep)(x)
    assert y.shape == x.shape
    big = x.at[0, 0].set(1e3)
    np.testing.assert_array_equal(S.splice(s, "topk", 8, scale=1 / 3.0, keep=keep)(big)[0, 0], big[0, 0])
