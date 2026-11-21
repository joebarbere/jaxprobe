"""jaxprobe.steer: patches, and the weight edit == runtime ablation identity on a tiny residual model."""
import jax
import jax.numpy as jnp
import numpy as np

import jaxprobe as jp
from jaxprobe import steer

L, D, H = 3, 8, 12
k = iter(jax.random.split(jax.random.key(0), 4))
P = {"embed": jax.random.normal(next(k), (20, D)), "w1": jax.random.normal(next(k), (L, D, H)) * 0.3,
     "w2": jax.random.normal(next(k), (L, H, D)) * 0.3}
TOK = jnp.arange(6) % 20


def model(p, tok):
    x = jp.tap("embed", p["embed"][tok])

    def body(x, w):
        return jp.tap("resid", x + jp.tap("mlp_out", jnp.tanh(x @ w["w1"]) @ w["w2"])), None

    return jp.scan(body, x, {"w1": p["w1"], "w2": p["w2"]})[0]


def test_mean_difference_and_actadd():
    pos, neg = jnp.ones((4, 3)) * 2, jnp.ones((5, 3))
    np.testing.assert_allclose(steer.mean_difference(pos, neg), jnp.ones(3))
    np.testing.assert_allclose(jnp.linalg.norm(steer.mean_difference(pos, neg, normalize=True)), 1.0, rtol=1e-6)
    np.testing.assert_allclose(steer.actadd(pos[0], neg[0]), jnp.ones(3))


def test_add_and_positions():
    x = jnp.zeros((2, 5, 3))
    np.testing.assert_allclose(steer.add(jnp.ones(3), 2.0)(x), 2.0)
    y = steer.add(jnp.ones(3), 1.0, positions=jnp.array([4]))(x)
    assert float(y[:, :4].sum()) == 0 and float(y[:, 4].sum()) == 6


def test_ablate_removes_component():
    u = jax.random.normal(jax.random.key(5), (D,))
    x = jax.random.normal(jax.random.key(6), (4, D))
    np.testing.assert_allclose(steer.ablate(u)(x) @ steer.unit(u), 0.0, atol=1e-5)
    np.testing.assert_allclose(steer.ablate(u, 2.0)(x) @ steer.unit(u), -(x @ steer.unit(u)), atol=1e-5)


def test_weight_edit_equals_runtime_ablation():
    """Project u out after every write to the stream == orthogonalize every matrix that writes to it."""
    u = jax.random.normal(jax.random.key(7), (D,))
    runtime, _ = jp.run(model, P, TOK, patch={"embed": steer.ablate(u), "layers.mlp_out": steer.ablate(u)})
    edited = dict(P, embed=steer.orthogonalize(P["embed"], u), w2=steer.orthogonalize(P["w2"], u))
    np.testing.assert_allclose(model(edited, TOK), runtime, atol=1e-5)
    np.testing.assert_allclose(runtime @ steer.unit(u), 0.0, atol=1e-5)
    assert not np.allclose(model(P, TOK), runtime, atol=1e-3)


def test_steer_at_one_layer_and_sweep():
    v = jnp.ones(D)
    _, base = jp.run(model, P, TOK, capture="layers.resid")
    _, st = jp.run(model, P, TOK, capture="layers.resid", patch={"layers.resid": jp.at(1, steer.add(v, 3.0))})
    np.testing.assert_allclose(st["layers.resid"][1], base["layers.resid"][1] + 3.0, atol=1e-5)
    rows = steer.sweep(lambda layer, c: {"shift": (jp.run(model, P, TOK, patch={
        "layers.resid": jp.at(layer, steer.add(v, c))})[0] - base["layers.resid"][-1]).mean()}, [0, 2], [0.0, 1.0])
    assert len(rows) == 4 and rows[0]["shift"] == 0.0 and rows[-1]["shift"] > 0
    assert float(steer.kl(st["layers.resid"], st["layers.resid"]).max()) < 1e-6
