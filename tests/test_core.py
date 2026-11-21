"""jaxprobe.core on a tiny residual MLP stack: capture, patch, scan stacking, and the JAX transforms."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprobe as jp

L, D, H = 4, 8, 16


def _params(key=0):
    k1, k2, k3 = jax.random.split(jax.random.key(key), 3)
    return {"w1": jax.random.normal(k1, (L, D, H)) * 0.3, "w2": jax.random.normal(k2, (L, H, D)) * 0.3,
            "out": jax.random.normal(k3, (D, 3))}


def block(x, p):
    h = jp.tap("hidden", jax.nn.gelu(x @ p["w1"]))
    return x + jp.tap("mlp_out", h @ p["w2"])


def model(params, x, remat=False):
    x = jp.tap("embed", x)
    body = (lambda c, p: (block(c, p), None))
    if remat:
        body = jp.checkpoint(body)
    x, _ = jp.scan(body, x, {"w1": params["w1"], "w2": params["w2"]})
    return jp.tap("final", x) @ params["out"]


def manual(params, x):
    """The same model with plain Python loops, returning every intermediate."""
    hs, ms, xs = [], [], [x]
    for i in range(L):
        h = jax.nn.gelu(x @ params["w1"][i])
        m = h @ params["w2"][i]
        x = x + m
        hs.append(h), ms.append(m), xs.append(x)
    return x @ params["out"], jnp.stack(hs), jnp.stack(ms)


P = _params()
X = jax.random.normal(jax.random.key(1), (5, D))


def test_tap_is_identity_outside_run():
    np.testing.assert_allclose(model(P, X), manual(P, X)[0], rtol=1e-5, atol=1e-6)


def test_capture_matches_manual_and_stacks_over_layers():
    out, acts = jp.run(model, P, X, capture=True)
    ref, hs, ms = manual(P, X)
    assert set(acts) == {"embed", "layers.hidden", "layers.mlp_out", "final"}
    assert acts["layers.hidden"].shape == (L, 5, H)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(acts["layers.hidden"], hs, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(acts["layers.mlp_out"], ms, rtol=1e-5, atol=1e-6)


def test_glob_patterns_select():
    _, acts = jp.run(model, P, X, capture="layers.*")
    assert set(acts) == {"layers.hidden", "layers.mlp_out"}
    _, acts = jp.run(model, P, X, capture=["*mlp_out", "final"])
    assert set(acts) == {"layers.mlp_out", "final"}


def test_identity_patch_is_bit_exact_noop():
    a, _ = jp.run(model, P, X)
    b, _ = jp.run(model, P, X, patch={"*": lambda x: x})
    np.testing.assert_array_equal(a, b)


def test_patch_at_one_layer():
    _, base = jp.run(model, P, X, capture="layers.mlp_out")
    _, acts = jp.run(model, P, X, capture="layers.mlp_out", patch={"layers.mlp_out": jp.at(2, jnp.zeros_like)})
    m = acts["layers.mlp_out"]
    assert float(jnp.abs(m[2]).max()) == 0.0
    np.testing.assert_allclose(m[:2], base["layers.mlp_out"][:2])
    assert not np.allclose(m[3], base["layers.mlp_out"][3])  # later layers see the edit


def test_activation_patching_roundtrip():
    """Capture from one input, patch every layer into a run on another: the output becomes the first run's."""
    X2 = X[::-1] * 2.0
    out1, acts1 = jp.run(model, P, X, capture="layers.mlp_out")
    out2, _ = jp.run(model, P, X2, patch={"embed": X, "layers.mlp_out": acts1["layers.mlp_out"]})
    np.testing.assert_allclose(out2, out1, rtol=1e-5, atol=1e-6)


def test_index_inside_scan():
    _, acts = jp.run(model, P, X, capture="layers.hidden",
                     patch={"layers.hidden": lambda h: h * 0 + jp.index().astype(h.dtype)})
    np.testing.assert_array_equal(acts["layers.hidden"][:, 0, 0], np.arange(L))
    assert jp.index() is None


def test_under_jit_grad_vmap():
    patched = {"layers.mlp_out": jp.at(1, lambda m: 0.5 * m)}

    def loss(p, x):
        out, acts = jp.run(model, p, x, capture="layers.hidden", patch=patched)
        return (out**2).sum() + acts["layers.hidden"].sum(), acts

    (l1, acts), g = jax.jit(jax.value_and_grad(loss, has_aux=True))(P, X)
    l0, _ = loss(P, X)
    np.testing.assert_allclose(l1, l0, rtol=1e-5)
    assert acts["layers.hidden"].shape == (L, 5, H)
    # gradients flow through the patch: compare with finite differences on one weight
    eps = 1e-3
    Pp = jax.tree.map(lambda a: a, P)
    Pp["w2"] = P["w2"].at[1, 0, 0].add(eps)
    fd = (loss(Pp, X)[0] - l0) / eps
    np.testing.assert_allclose(g["w2"][1, 0, 0], fd, rtol=2e-2, atol=2e-2)
    # vmap over examples
    vout, vacts = jax.vmap(lambda x: jp.run(model, P, x[None], capture="final"))(X)
    np.testing.assert_allclose(vout[:, 0], model(P, X), rtol=1e-5, atol=1e-6)
    assert vacts["final"].shape == (5, 1, D)


def test_checkpoint_routes_captures():
    _, acts = jp.run(model, P, X, capture=True, remat=True)
    _, hs, _ = manual(P, X)
    np.testing.assert_allclose(acts["layers.hidden"], hs, rtol=1e-5, atol=1e-6)
    g1 = jax.grad(lambda p: jp.run(model, p, X, remat=True, patch={"layers.hidden": jp.at(0, jnp.tanh)})[0].sum())(P)
    g2 = jax.grad(lambda p: jp.run(model, p, X, remat=False, patch={"layers.hidden": jp.at(0, jnp.tanh)})[0].sum())(P)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6), g1, g2)


def test_nested_scan_and_scope():
    def inner(x, w):
        return x + jp.tap("v", jnp.tanh(x * w)), None

    def outer(x, ws):
        x, _ = jp.scan(inner, x, ws, name="sub")
        return x, None

    def f(x):
        with jp.scope("pre"):
            x = jp.tap("x", x + 1)
        return jp.scan(outer, x, jnp.ones((3, 2)), name="blocks")[0]

    out, acts = jp.run(f, jnp.zeros(4), capture=True)
    assert set(acts) == {"pre.x", "blocks.sub.v"}
    assert acts["blocks.sub.v"].shape == (3, 2, 4)
    np.testing.assert_allclose(out, f(jnp.zeros(4)))


def test_errors():
    with pytest.raises(jp.TapError, match="matched no tap"):
        jp.run(model, P, X, capture="layers.typo")
    jitted = jax.jit(model)
    with pytest.raises(jp.TapError, match="transform"):  # first call: the taps run inside jit's own trace
        jp.run(jitted, P, X, capture="final")
    with pytest.raises(jp.TapError, match="already jitted"):  # cached: the taps don't run at all
        jitted(P, X)
        jp.run(jitted, P, X, capture="final")
    with pytest.raises(jp.TapError, match="transform"):
        jp.run(jax.vmap(lambda x: jp.tap("v", x)), X, capture="v")
    with pytest.raises(jp.TapError, match="shape"):
        jp.run(model, P, X, patch={"final": lambda x: x[:, :2]})
    with pytest.raises(jp.TapError, match="twice"):
        jp.run(lambda x: jp.tap("a", x) + jp.tap("a", x), X, capture="a")
    _, acts = jp.run(model, P, X, capture="nothing", strict=False)
    assert acts == {}


def test_pytree_taps_and_dtype():
    def f(x):
        y = jp.tap("pair", {"a": x, "b": x.astype(jnp.bfloat16)})
        return y["a"].sum() + y["b"].astype(jnp.float32).sum()

    _, acts = jp.run(f, X, capture="pair", patch={"pair": lambda t: {"a": t["a"] * 2, "b": t["b"] * 0.0}})
    assert acts["pair"]["b"].dtype == jnp.bfloat16
    np.testing.assert_allclose(acts["pair"]["a"], 2 * X)
