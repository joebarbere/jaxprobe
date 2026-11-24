"""0.5.0: logit and tuned lens, direct logit attribution, PCA directions, norm-matched random controls."""
import json
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxprobe as jp
from jaxprobe import directions, lens, steer

L, D, H, V = 4, 8, 16, 30
k = iter(jax.random.split(jax.random.key(1), 12))
P = {"embed": jax.random.normal(next(k), (V, D)), "wa": jax.random.normal(next(k), (L, D, D)) * 0.3,
     "w1": jax.random.normal(next(k), (L, D, H)) * 0.3, "w2": jax.random.normal(next(k), (L, H, D)) * 0.3,
     "norm": 1 + 0.1 * jax.random.normal(next(k), (D,)), "W_U": jax.random.normal(next(k), (D, V))}
TOK = jax.random.randint(next(k), (3, 7), 0, V)


def model(p, tok):
    x = jp.tap("embed", p["embed"][tok])

    def layer(x, w):
        x = x + jp.tap("attn_out", jnp.cumsum(x, 1) / jnp.arange(1, x.shape[1] + 1)[:, None] @ w["wa"])
        return jp.tap("resid", x + jp.tap("mlp_out", jnp.tanh(x @ w["w1"]) @ w["w2"])), None

    x = jp.scan(layer, x, {"wa": p["wa"], "w1": p["w1"], "w2": p["w2"]})[0]
    return U(x), x


U = lens.rms_unembed(P["norm"], P["W_U"], eps=1e-6)


def _acts():
    (logits, h), a = jp.run(model, P, TOK, capture=["embed", "layers.resid", "layers.attn_out", "layers.mlp_out"])
    return logits, h, a, jnp.concatenate([a["embed"][None], a["layers.resid"]])


def test_unembed_kinds():
    h = jax.random.normal(next(k), (5, D)) * 3
    x = h / np.sqrt((np.asarray(h) ** 2).mean(-1, keepdims=True) + 1e-6) * P["norm"]
    np.testing.assert_allclose(U(h), x @ P["W_U"], rtol=1e-5, atol=1e-5)
    g = lens.rms_unembed(P["norm"] - 1, P["W_U"], eps=1e-6, offset=1.0)                # Gemma-style (1 + w)
    np.testing.assert_allclose(g(h), U(h), rtol=1e-5, atol=1e-5)
    w, b, hb = P["norm"], jnp.linspace(-1, 1, D), jnp.linspace(0, 2, V)
    ln = lens.layer_unembed(w, b, P["W_U"], eps=1e-6, head_b=hb)
    c = h - h.mean(-1, keepdims=True)
    want = (c / jnp.sqrt((c ** 2).mean(-1, keepdims=True) + 1e-6) * w + b) @ P["W_U"] + hb
    np.testing.assert_allclose(ln(h), want, rtol=1e-5, atol=1e-5)
    b16 = lens.rms_unembed(P["norm"], P["W_U"].astype(jnp.bfloat16), eps=1e-6)        # bf16 matmul, f32 out
    assert b16(h).dtype == jnp.float32
    np.testing.assert_allclose(b16(h), U(h), rtol=3e-2, atol=5e-2)


def test_stats_match_brute_force():
    logits, _, _, R = _acts()
    nxt = jnp.roll(TOK, -1, 1)
    yes, no = [1, 4, 9], [2, 3]
    st = jax.jit(lambda U, R: lens.stats(U, R, token=nxt, sets={"yes": yes, "no": no}, top=3))(U, R)
    full = lens.logits(U, R)                                                            # (L+1, B, T, V)
    np.testing.assert_allclose(full[-1], logits, rtol=1e-5, atol=1e-5)
    lp = jax.nn.log_softmax(full)
    lpf = lp[-1]
    np.testing.assert_allclose(st["kl"], (jnp.exp(lpf) * (lpf - lp)).sum(-1), rtol=1e-4, atol=1e-5)
    assert float(jnp.abs(st["kl"][-1]).max()) < 1e-5 and bool(st["agree"][-1].all())     # last layer is the model
    np.testing.assert_array_equal(st["agree"], lp.argmax(-1) == lpf.argmax(-1))
    np.testing.assert_allclose(st["entropy"], -(jnp.exp(lp) * lp).sum(-1), rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(st["top_ids"], jnp.argsort(-lp, -1)[..., :3])
    np.testing.assert_allclose(st["logprob"], jnp.take_along_axis(lp, nxt[None, ..., None], -1)[..., 0], rtol=1e-5)
    np.testing.assert_array_equal(st["rank"], (jnp.argsort(-full, -1) == nxt[None, ..., None]).argmax(-1))
    np.testing.assert_allclose(st["sets"]["yes"], jnp.log(jnp.exp(lp[..., yes]).sum(-1)), rtol=1e-4, atol=1e-5)
    assert st["sets"]["no"].shape == (L + 1, 3, 7)


def test_tuned_identity_loss_and_training():
    logits, _, _, R = _acts()
    t0 = lens.init_tuned(L, D)
    np.testing.assert_allclose(lens.translate(t0, 2, R[2]), R[2])
    a, b = lens.stats(U, R[:-1], logits), lens.stats(U, R[:-1], logits, tuned=t0)
    np.testing.assert_allclose(a["kl"], b["kl"], rtol=1e-6)
    np.testing.assert_allclose(lens.tuned_loss(t0, U, R[:-1], logits), a["kl"].mean((1, 2)).sum(), rtol=1e-5)
    with pytest.raises(ValueError):
        lens.stats(U, R, logits, tuned=t0)                                              # 4 translators, 5 layers

    def collect(tok):
        (lg, _), a = jp.run(model, P, tok, capture=["embed", "layers.resid"])
        R = jnp.concatenate([a["embed"][None], a["layers.resid"]])
        return R[:-1].reshape(L, -1, D), lg.reshape(-1, V)

    toks = (jax.random.randint(jax.random.key(i), (8, 7), 0, V) for i in range(10_000))
    tuned, hist = lens.fit_tuned(U, collect, toks, steps=300, lr=3e-2, warmup=10, final_lr=1e-3, log=None)
    assert len(hist) == 4 and hist[-1][1] < 0.5 * hist[0][1]
    Rt, ft = collect(jax.random.randint(jax.random.key(99999), (8, 7), 0, V))
    before = float(lens.tuned_loss(t0, U, Rt, ft))
    after = float(lens.tuned_loss(tuned, U, Rt, ft))
    assert after < 0.5 * before                                                         # and on held-out tokens


@pytest.mark.parametrize("kind", ["rms", "layer"])
def test_dla_parts_sum_to_the_logit(kind):
    _, h, a, _ = _acts()
    u = U if kind == "rms" else lens.layer_unembed(P["norm"], jnp.linspace(-1, 1, D), P["W_U"], 1e-6,
                                                   jnp.linspace(0, 1, V))
    parts = {"embed": a["embed"], "attn": a["layers.attn_out"], "mlp": a["layers.mlp_out"]}
    np.testing.assert_allclose(a["embed"] + a["layers.attn_out"].sum(0) + a["layers.mlp_out"].sum(0), h, atol=1e-4)
    tok, minus = jnp.roll(TOK, -1, 1), (TOK + 3) % V
    c = lens.dla(u, parts, ref=h, token=tok, minus=minus)
    assert c["attn"].shape == (L, 3, 7) and c["embed"].shape == (3, 7)
    total = c["embed"] + c["attn"].sum(0) + c["mlp"].sum(0) + c["const"]
    lg = u(h)
    want = jnp.take_along_axis(lg, tok[..., None], -1)[..., 0] - jnp.take_along_axis(lg, minus[..., None], -1)[..., 0]
    np.testing.assert_allclose(total, want, rtol=1e-4, atol=1e-4)
    one = lens.dla(u, a["layers.mlp_out"][2], ref=h, token=5)                           # a bare array, one token
    np.testing.assert_allclose(one, lens.dla(u, parts, ref=h, token=5)["mlp"][2], rtol=1e-6)


def test_html(tmp_path):
    _, _, _, R = _acts()
    st = lens.stats(U, R, token=jnp.roll(TOK, -1, 1), top=4)
    decode = lambda i: f"<t{i}>" if i != 3 else "</script>"
    p = lens.to_html(tmp_path / "l.html", st, TOK[1], decode, row=1, layer_names=["emb", "1", "2", "3", "4"])
    page = p.read_text()
    blob = re.search(r"const D=(.*?);\n", page, re.DOTALL).group(1)
    assert "</script>" not in blob
    d = json.loads(blob.replace("<\\/", "</"))
    assert d["names"][0] == "emb" and len(d["cells"]) == L + 1 and len(d["cells"][0]) == 7 and len(d["cells"][0][0]) == 4
    assert d["cols"] == [decode(int(t)) for t in TOK[1]]
    assert d["vocab"][d["cells"][2][5][0][0]] == decode(int(st["top_ids"][2, 1, 5, 0]))
    assert len(d["target"]) == L + 1 and np.asarray(d["kl"]).shape == (L + 1, 7)


def test_pca_recovers_a_planted_direction():
    r = np.random.default_rng(0)
    u = r.normal(size=(3, 16))                                                         # one per "layer"
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    base = r.normal(size=(200, 3, 16))
    amp = 2 + r.normal(size=(200, 3, 1))                                               # varies along u
    pos, neg = base + amp * u + 0.1 * r.normal(size=base.shape), base
    v, ex = directions.pca(pos, neg, return_explained=True)
    assert v.shape == (3, 16) and ex.shape == (3,)
    np.testing.assert_allclose((np.asarray(v) * u).sum(-1), 1.0, atol=2e-2)          # found, with the right sign
    assert (ex > 0.5).all()
    v2 = directions.pca(-np.asarray(pos), -base)                                       # sign follows pos − neg
    np.testing.assert_allclose((np.asarray(v2) * u).sum(-1), -1.0, atol=2e-2)
    V2 = np.asarray(directions.pca(pos, neg, k=2))
    assert V2.shape == (3, 2, 16)
    np.testing.assert_allclose(np.einsum("lkd,ljd->lkj", V2, V2), np.broadcast_to(np.eye(2), (3, 2, 2)), atol=1e-5)


def test_random_like():
    v = jax.random.normal(jax.random.key(2), (4, 32)) * jnp.arange(1, 5)[:, None]
    r = steer.random_like(v, jax.random.key(3))
    np.testing.assert_allclose(jnp.linalg.norm(r, axis=-1), jnp.linalg.norm(v, axis=-1), rtol=1e-5)
    assert float(jnp.abs((r * v).sum(-1) / (jnp.linalg.norm(r, axis=-1) * jnp.linalg.norm(v, axis=-1))).max()) < 0.8
    o = steer.random_like(v, jax.random.key(3), orthogonal=True)
    np.testing.assert_allclose((o * v).sum(-1), 0.0, atol=1e-3)
    np.testing.assert_allclose(jnp.linalg.norm(o, axis=-1), jnp.linalg.norm(v, axis=-1), rtol=1e-5)
    assert not np.allclose(r, steer.random_like(v, jax.random.key(4)))
