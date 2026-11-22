"""jaxprobe.trace on a tiny pre-norm transformer (attention + gated MLP, float32) with random transcoders."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprobe as jp
from jaxprobe import sae as S
from jaxprobe import trace

V, D, F, L, T, H = 23, 12, 20, 3, 6, 2
ks = iter(jax.random.split(jax.random.key(0), 16))
n = lambda *s: jax.random.normal(next(ks), s) * 0.4
P = {"embed": n(V, D) * 2, "wq": n(L, D, D), "wk": n(L, D, D), "wv": n(L, D, D), "wo": n(L, D, D),
     "w1": n(L, D, F), "w2": n(L, D, F), "w3": n(L, F, D), "g1": 1 + n(L, D), "g2": 1 + n(L, D), "gf": 1 + n(D)}


def norm(x, g, name):
    s = jp.tap(name, jax.lax.rsqrt((x * x).mean(-1, keepdims=True) + 1e-5))
    return x * s * g


def model(tokens):
    x = jp.tap("embed", P["embed"][tokens])
    Tn = tokens.shape[1]

    def layer(x, p):
        h = norm(x, p["g1"], "attn_scale")
        q, k, v = (h @ p["wq"]).reshape(1, Tn, H, -1), (h @ p["wk"]).reshape(1, Tn, H, -1), (h @ p["wv"]).reshape(
            1, Tn, H, -1)
        s = jnp.einsum("bthd,bshd->bhts", q, k) + jnp.where(jnp.tril(jnp.ones((Tn, Tn))) > 0, 0.0, -1e9)
        w = jp.tap("pattern", jax.nn.softmax(s, -1))
        x = x + jnp.einsum("bhts,bshd->bthd", w, v).reshape(1, Tn, D) @ p["wo"]
        m = jp.tap("mlp_in", norm(x, p["g2"], "mlp_scale"))
        x = x + jp.tap("mlp_out", (jax.nn.silu(m @ p["w1"]) * (m @ p["w2"])) @ p["w3"])
        return x, None

    lp = {k: P[k] for k in ("wq", "wk", "wv", "wo", "w1", "w2", "w3", "g1", "g2")}
    x, _ = jp.scan(layer, x, lp)
    return norm(x, P["gf"], "final_scale") @ P["embed"].T


TOK = jnp.array([[3, 7, 1, 9, 4, 11]])
TC = jax.vmap(lambda k: {kk: v for kk, v in S.init(k, D, 32).items() if kk != "log_threshold"})(
    jax.random.split(jax.random.key(5), L))


@pytest.mark.parametrize("skip", [(), (0,), (0, 2)])
def test_replacement_is_exact_and_edges_sum_to_values(skip):
    g = trace.attribute(model, TOK, TC, k=4, skip_positions=skip)
    assert g.scores["logit_check"] < 1e-4
    T_, N = g.T, g.N
    assert g.adj.shape[0] == T_ + N + L * T_ + len(g.logit_ids)
    # linear model: a target's value = Σ incoming edges + its constant part
    rows = np.concatenate([np.arange(T_, T_ + N), np.arange(T_ + N + L * T_, len(g.adj))])
    np.testing.assert_allclose(g.adj[rows].sum(1) + g.bias, g.values, rtol=1e-3, atol=1e-3)


def test_causality():
    g = trace.attribute(model, TOK, TC, k=4, skip_positions=())
    sl = g.slices()
    for t in range(sl["feature"].start, sl["feature"].stop):
        ti = g.node_info(t)
        for s in np.flatnonzero(g.adj[t]):
            si = g.node_info(int(s))
            assert si["pos"] <= ti["pos"]  # nothing flows backwards in time
            assert si["layer"] < ti["layer"]  # a feature reads only earlier layers' outputs (or the embedding)


def test_prune_and_scores():
    g = trace.prune(trace.attribute(model, TOK, TC, k=4, skip_positions=(0,)), 0.8, 0.98)
    s = g.scores
    assert 0 <= s["replacement"] <= 1 and 0 <= s["completeness"] <= 1
    assert abs(s["replacement"] + s["error_share"] - 1) < 1e-6
    assert 0 < s["n_features"] <= g.N
    assert g.top_features(3)[0]["influence"] >= g.top_features(3)[-1]["influence"]
    small = trace.prune(trace.attribute(model, TOK, TC, k=4, skip_positions=(0,)), 0.8, 0.98, max_features=3,
                        max_edges=10)
    assert small.scores["n_features"] <= 3 and small.scores["n_edges"] <= 10
    js = trace.to_json(g, lambda t: f"t{t}")
    ids = {nd["node_id"] for nd in js["nodes"]}
    assert all(lk["source"] in ids and lk["target"] in ids for lk in js["links"])
    logits = [nd for nd in js["nodes"] if nd["feature_type"] == "logit"]
    assert sum(nd["is_target_logit"] for nd in logits) == 1 and logits[0]["layer"] == str(L + 1)
    assert trace.cantor(3, 1234) == (1237 * 1238) // 2 + 1234 == 766937
    feats = [nd for nd in js["nodes"] if nd["feature_type"] == "cross layer transcoder"]
    assert all(0 < nd["influence"] <= 1 + 1e-6 for nd in feats)


def test_influence_is_neumann_series():
    A = np.triu(np.random.default_rng(0).random((6, 6)), 1)  # strictly upper triangular: a DAG
    An = trace.normalize(A)
    w = np.zeros(6)
    w[0] = 1.0
    closed = w @ An @ np.linalg.inv(np.eye(6) - An)
    np.testing.assert_allclose(trace.influence(An, w), closed, atol=1e-9)


def test_ablate_features_patch():
    g = trace.attribute(model, TOK, TC, k=4, skip_positions=())
    feats = [tuple(int(v) for v in g.feat[0])]
    _, a0 = jp.run(model, TOK, capture="layers.mlp_out")
    _, a1 = jp.run(model, TOK, capture="layers.mlp_out",
                   patch=trace.ablate_features(TC, jnp.ones(L), feats, [g.act[0]]))  # default: remove
    layer, pos, f = feats[0]
    np.testing.assert_allclose(a1["layers.mlp_out"][layer, 0, pos],
                               a0["layers.mlp_out"][layer, 0, pos] - g.act[0] * TC["W_dec"][layer, f], atol=1e-5)


def test_write_graph(tmp_path):
    import json
    g = trace.prune(trace.attribute(model, TOK, TC, k=4))
    for slug in ("a", "b", "a"):
        trace.write_graph(tmp_path, trace.to_json(g, str, slug=slug))
    assert [m["slug"] for m in json.loads((tmp_path / "graph-metadata.json").read_text())["graphs"]] == ["b", "a"]
    page = trace.feature_page(2, 5, [(["a", "b"], [0.0, 1.5])], top_logits=["x"])
    assert page["index"] == trace.cantor(2, 5) and page["act_max"] == 1.5
