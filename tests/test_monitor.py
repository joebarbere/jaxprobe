"""0.6.0: monitor — in-jit health reducers, running baselines, SAE drift, JSON lines."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprobe as jp
from jaxprobe import dash, monitor

L, B, T, D, H, NH, V = 3, 2, 6, 8, 12, 2, 20
k = iter(jax.random.split(jax.random.key(5), 12))
P = {"embed": jax.random.normal(next(k), (V, D)), "wq": jax.random.normal(next(k), (L, NH, D, D)) * 0.5,
     "w1": jax.random.normal(next(k), (L, D, H)) * 0.5, "w2": jax.random.normal(next(k), (L, H, D)) * 0.3,
     "W_U": jax.random.normal(next(k), (D, V))}
TOK = jax.random.randint(next(k), (B, T), 0, V)
MASK = jnp.array([[1] * T, [0, 0] + [1] * (T - 2)], bool)
TRACES = {"n": 0}


def model(p, tok):
    TRACES["n"] += 1
    x = jp.tap("embed", p["embed"][tok])
    causal = jnp.tril(jnp.ones((T, T), bool))

    def layer(x, w):
        s = jnp.einsum("btd,hde,bse->bhts", x, w["wq"], x) / math.sqrt(D)
        a = jp.tap("pattern", jax.nn.softmax(jnp.where(causal, s, -1e9), -1))         # (B, NH, T, T)
        x = x + jnp.einsum("bhts,bsd->btd", a, x) / NH
        act = jp.tap("mlp_act", jax.nn.relu(x @ w["w1"]))
        return jp.tap("resid", x + act @ w["w2"]), None

    x = jp.scan(layer, x, {"wq": p["wq"], "w1": p["w1"], "w2": p["w2"]})[0]
    return x @ p["W_U"]


def _caps():
    return jp.run(model, P, TOK, capture=["layers.resid", "layers.pattern", "layers.mlp_act"])


def test_health_matches_brute_force_per_layer():
    _, raw = _caps()
    _, m = jp.run(model, P, TOK, capture={"layers.resid": monitor.health(mask=MASK, outlier=2.0)})
    h = m["layers.resid"]
    assert h["rms"].shape == (L,)
    for i in range(L):
        x = np.asarray(raw["layers.resid"][i])[np.asarray(MASK)]                     # (n, D) real tokens
        rms = np.sqrt((x ** 2).mean(-1))
        np.testing.assert_allclose(h["rms"][i], rms.mean(), rtol=1e-5)
        np.testing.assert_allclose(h["rms_max"][i], rms.max(), rtol=1e-5)
        np.testing.assert_allclose(h["max_abs"][i], np.abs(x).max(), rtol=1e-5)
        per = np.abs(x).mean(0)
        assert h["outlier_dims"][i] == (per > 2.0 * np.median(per)).sum()
    assert (h["nonfinite"] == 0).all()
    x = jnp.ones((B, T, D)).at[0, 1, 3].set(jnp.nan).at[1, 2, 0].set(jnp.inf).at[1, 3, 5].set(40.0)
    g = monitor.health(outlier=6.0)(x)
    assert int(g["nonfinite"]) == 2 and float(g["max_abs"]) == 40.0
    assert int(g["outlier_dims"]) == 0                                               # one big entry, not a dim
    y = jnp.ones((B, T, D)).at[..., 2].set(20.0)
    assert int(monitor.health()(y)["outlier_dims"]) == 1


def test_attention_and_units():
    _, raw = _caps()
    _, m = jp.run(model, P, TOK, capture={"layers.pattern": monitor.attention(mask=MASK),
                                          "layers.mlp_act": monitor.units(mask=MASK)})
    a, u = m["layers.pattern"], m["layers.mlp_act"]
    assert a["entropy"].shape == (L, NH) and a["sink"].shape == (L,)
    msk = np.asarray(MASK)
    for i in range(L):
        pat = np.asarray(raw["layers.pattern"][i])                                    # (B, NH, T, S)
        ent = -(pat * np.log(np.where(pat > 0, pat, 1))).sum(-1)                     # (B, NH, T)
        w = np.broadcast_to(msk[:, None, :], ent.shape)
        np.testing.assert_allclose(a["entropy"][i], (ent * w).sum((0, 2)) / w.sum((0, 2)), rtol=1e-5)
        np.testing.assert_allclose(a["sink"][i], (pat[..., 0] * w).sum() / w.sum(), rtol=1e-5)
        np.testing.assert_allclose(a["max_prob"][i], (pat.max(-1) * w).sum() / w.sum(), rtol=1e-5)
        act = np.asarray(raw["layers.mlp_act"][i])[msk]                               # (n, H)
        np.testing.assert_allclose(u["dead"][i], ((act > 0).sum(0) == 0).mean(), rtol=1e-6)
        np.testing.assert_allclose(u["active"][i], (act > 0).mean(), rtol=1e-6)
    uni = jnp.where(jnp.tril(jnp.ones((T, T), bool)), 1.0, 0.0)
    uni = jnp.broadcast_to(uni / uni.sum(-1, keepdims=True), (B, NH, T, T))
    want = np.log(np.arange(1, T + 1)).mean()                                         # uniform causal attention
    np.testing.assert_allclose(monitor.attention()(uni)["entropy"], np.full(NH, want), rtol=1e-5)


def test_logit_stats_and_project():
    logits = jax.random.normal(next(k), (B, T, V)) * 3
    s = monitor.logit_stats(logits, mask=MASK)
    lg = np.asarray(logits)[np.asarray(MASK)]
    z = np.log(np.exp(lg).sum(-1))
    np.testing.assert_allclose(s["z"], z.mean(), rtol=1e-5)
    np.testing.assert_allclose(s["z2"], (z ** 2).mean(), rtol=1e-5)
    np.testing.assert_allclose(s["max_logit"], lg.max(), rtol=1e-6)
    lp = lg - z[:, None]
    np.testing.assert_allclose(s["entropy"], -(np.exp(lp) * lp).sum(-1).mean(), rtol=1e-5)
    u, v = jax.random.normal(next(k), (D,)), jax.random.normal(next(k), (D,))
    _, raw = _caps()
    _, m = jp.run(model, P, TOK, capture={"layers.resid": monitor.project({"u": u, "v": v}, bias=1.0)})
    np.testing.assert_allclose(m["layers.resid"]["u"], raw["layers.resid"][:, :, -1] @ u + 1.0, rtol=1e-5)
    _, m2 = jp.run(model, P, TOK, capture={"layers.resid": monitor.project(jnp.stack([u, v]))})
    assert m2["layers.resid"].shape == (L, B, 2)
    np.testing.assert_allclose(m2["layers.resid"][..., 1], m["layers.resid"]["v"] - 1.0, rtol=1e-5, atol=1e-5)


def test_inside_jit_and_grad_compiles_once():
    probes = {"layers.resid": monitor.health(mask=MASK), "layers.pattern": monitor.attention(mask=MASK)}

    @jax.jit
    def step(p, tok):
        def loss(p):
            logits, m = jp.run(model, p, tok, capture=probes)
            return jax.nn.logsumexp(logits, -1).mean(), (m, monitor.logit_stats(logits, MASK))
        (l, aux), g = jax.value_and_grad(loss, has_aux=True)(p)
        return jax.tree.map(lambda a, b: a - 0.01 * b, p, g), l, aux

    TRACES["n"] = 0
    p = P
    for _ in range(3):
        p, _, aux = step(p, TOK)
    assert TRACES["n"] == 1
    assert aux[0]["layers.resid"]["rms"].shape == (L,) and np.isfinite(float(aux[1]["z"]))


def test_running_baseline_and_flags():
    r = np.random.default_rng(0)
    xs = [{"a": r.normal(5, 2, size=3), "b": {"c": float(r.normal())}} for _ in range(50)]
    run = monitor.Running()
    assert run.flags(xs[0]) == {}
    for x in xs:
        run.update(x)
    a = np.stack([x["a"] for x in xs])
    np.testing.assert_allclose(run.mean["a"], a.mean(0), rtol=1e-10)
    np.testing.assert_allclose(run.var["a"], a.var(0), rtol=1e-10)
    spike = {"a": a.mean(0) + np.array([0.0, 30.0, 0.0]), "b": {"c": 0.0}}
    f = run.flags(spike, z=6.0)
    assert list(f) == ["a/1"] and f["a/1"] > 6
    ew = monitor.Running(decay=0.9, warmup=1)
    for x in [{"q": 1.0}] * 20 + [{"q": 2.0}] * 20:
        ew.update(x)
    assert abs(ew.mean["q"] - 2.0) < 0.2                                             # forgets the old level
    const = monitor.Running(warmup=1).update({"q": 3.0}).update({"q": 3.0})
    assert const.flags({"q": 3.0}) == {} and "q" in const.flags({"q": 3.5})


def _stats(rates, n_seq=200, T=40, seed=0, scale=1.0):
    r = np.random.default_rng(seed)
    z = (r.random((n_seq, T, len(rates))) < rates).astype(np.float32) * r.gamma(2.0, scale, (n_seq, T, len(rates)))
    col = dash.Collector(len(rates), top=2)
    col.update(z.astype(np.float32), np.zeros((n_seq, T), int))
    return col.result()


def test_drift():
    rates = np.array([0.05, 0.05, 0.02, 0.0, 0.01, 0.05])
    base = _stats(rates, seed=1)
    same = monitor.drift(base, _stats(rates, seed=2))
    assert same["more"] == same["less"] == same["gone"] == same["new"] == 0 and same["median_js"] < 0.05
    moved = rates.copy()
    moved[1], moved[2], moved[3], moved[4] = 0.15, 0.0, 0.03, 0.01
    d = monitor.drift(base, _stats(moved, seed=3))
    assert d["more"] == 1 and d["gone"] == 1 and d["new"] == 1
    assert d["top"][0][0] == 1 and {t[0] for t in d["top"][:3]} == {1, 2, 3}         # more, then new and gone
    assert d["rate_z"][1] > 5 and d["log2_ratio"][1] == pytest.approx(np.log2(3), abs=0.3)
    bigger = monitor.drift(base, _stats(rates, seed=4, scale=4.0))                     # same rates, larger sizes
    assert bigger["more"] == 0 and bigger["median_js"] > 0.3


def test_drift_counts_bursts_by_sequence():
    def bursty(seed):                                  # a feature is either on for a whole passage or off
        r = np.random.default_rng(seed)
        on = r.random((300, 1, 8)) < 0.1
        z = ((r.random((300, 50, 8)) < 0.5) & on).astype(np.float32)
        col = dash.Collector(8, top=1)
        col.update(z, np.zeros((300, 50), int))
        return col.result()
    zs, zb = [], []
    for seed in range(5):
        a, b = bursty(2 * seed), bursty(2 * seed + 1)
        zs.append(monitor.drift(a, b)["rate_z"])
        a2 = {k: v for k, v in a.items() if k not in ("fired_sq", "fired_m", "tok_sq")}
        b2 = {k: v for k, v in b.items() if k not in ("fired_sq", "fired_m", "tok_sq")}
        zb.append(monitor.drift(a2, b2)["rate_z"])                                   # token-level binomial
    zs, zb = np.concatenate(zs), np.concatenate(zb)
    assert 0.6 < zs.std() < 1.5                                                       # calibrated: z ~ N(0, 1)
    assert zb.std() > 3 * zs.std()                                                    # binomial: far too confident


def test_scalars_and_jsonl(tmp_path):
    m = {"layers.resid": {"rms": jnp.array([1.0, 2.0]), "n": jnp.int32(3)}, "z": jnp.float32(jnp.nan),
         "grid": np.arange(4.0).reshape(2, 2)}
    s = monitor.scalars(m)
    assert s["layers.resid/rms/1"] == 2.0 and s["layers.resid/n"] == 3.0 and s["grid/1/0"] == 2.0
    assert math.isnan(s["z"])
    with monitor.JsonlWriter(tmp_path / "log" / "m.jsonl") as w:
        w.write(0, m, loss=1.5)
        w.write(10, m, loss=jnp.float32(1.25))
    rows = monitor.read_jsonl(tmp_path / "log" / "m.jsonl")
    assert [r["step"] for r in rows] == [0, 10] and rows[1]["loss"] == 1.25
    assert rows[0]["metrics"]["layers.resid"]["rms"] == [1.0, 2.0] and rows[0]["metrics"]["z"] is None
    assert rows[0]["metrics"]["grid"] == [[0.0, 1.0], [2.0, 3.0]] and rows[0]["metrics"]["layers.resid"]["n"] == 3
