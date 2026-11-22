"""0.3.0: pytree patches + bind, reducers, batches, directions, probes, safetensors io, checkpoint edits, streaming."""
import json

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

import jaxprobe as jp
from jaxprobe import batch, directions, io, probe, reduce, steer

L, D, H, V = 3, 8, 12, 20
k = iter(jax.random.split(jax.random.key(0), 8))
P = {"embed": jax.random.normal(next(k), (V, D)), "w1": jax.random.normal(next(k), (L, D, H)) * 0.3,
     "w2": jax.random.normal(next(k), (L, H, D)) * 0.3, "unembed": jax.random.normal(next(k), (D, V))}
TOK = (jnp.arange(12).reshape(2, 6) * 7) % V
TRACES = {"n": 0}


def layer(w, x):
    return jp.tap("resid", x + jp.tap("mlp_out", jnp.tanh(x @ w["w1"]) @ w["w2"]))


def model(p, tok):
    TRACES["n"] += 1
    x = jp.tap("embed", p["embed"][tok])
    x = jp.scan(lambda x, w: (layer(w, x), None), x, {"w1": p["w1"], "w2": p["w2"]})[0]
    return x @ p["unembed"]


# ---------------------------------------------------------------- bind, pytree patches, reducers
def test_bind_compiles_once_for_many_coefficients():
    g = jax.jit(jp.bind(model))
    u = jnp.ones(D)
    TRACES["n"] = 0
    outs = [g({"layers.resid": jp.at(1, steer.add(u, c))}, P, TOK)[0] for c in (0.0, 1.0, 2.5)]
    assert TRACES["n"] == 1
    want, _ = jp.run(model, P, TOK, patch={"layers.resid": jp.at(1, steer.add(u, 2.5))})
    np.testing.assert_allclose(outs[2], want, rtol=1e-5, atol=1e-5)
    assert not np.allclose(outs[0], outs[2])


def test_vmap_over_coefficient_and_layer():
    g = jp.bind(model)
    u = jax.random.normal(jax.random.key(3), (D,))
    cs, ls = jnp.array([0.0, 1.0, 3.0]), jnp.array([0, 2, 1])
    out = jax.vmap(lambda c, l: g({"layers.resid": jp.at(l, steer.add(u, c))}, P, TOK)[0])(cs, ls)
    for i in range(3):
        want, _ = jp.run(model, P, TOK, patch={"layers.resid": jp.at(int(ls[i]), steer.add(u, float(cs[i])))})
        np.testing.assert_allclose(out[i], want, rtol=1e-5, atol=1e-5)


def test_ablate_patch_is_pytree_and_plain_at_still_works():
    leaves = jax.tree.leaves(jp.at(1, steer.ablate(jnp.ones(D), 2.0)))
    assert len(leaves) == 3  # layers, u, alpha
    _, base = jp.run(model, P, TOK, capture="layers.resid")
    _, caps = jp.run(model, P, TOK, capture="layers.resid", patch={"layers.resid": jp.at([2], lambda x: x * 0)})
    assert float(jnp.abs(caps["layers.resid"][2]).max()) == 0
    np.testing.assert_allclose(caps["layers.resid"][:2], base["layers.resid"][:2])


def test_capture_reducers_run_inside_scan():
    _, full = jp.run(model, P, TOK, capture="layers.resid")
    _, red = jp.run(model, P, TOK, capture={"layers.resid": reduce.last, "embed": None})
    assert red["layers.resid"].shape == (L, 2, D)
    np.testing.assert_allclose(red["layers.resid"], full["layers.resid"][..., -1, :])
    assert red["embed"].shape == (2, 6, D)
    w = jnp.array([1.0, 0.0])
    _, s = jp.run(model, P, TOK, capture={"layers.resid": reduce.compose(reduce.positions([-2, -1]),
                                                                          reduce.batch_sum(w))})
    np.testing.assert_allclose(s["layers.resid"], full["layers.resid"][:, 0, -2:], rtol=1e-6)
    m = jnp.array([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], bool)
    mm = reduce.masked_mean(m)(full["layers.resid"])
    np.testing.assert_allclose(mm[:, 0], full["layers.resid"][:, 0, 2:].mean(1), rtol=1e-5)


def test_batch_sum_ignores_nan_in_weight_zero_rows():
    x = jnp.array([[1.0, 2.0], [jnp.nan, jnp.nan]])
    np.testing.assert_allclose(reduce.batch_sum(jnp.array([1.0, 0.0]))(x), [1.0, 2.0])


# ---------------------------------------------------------------- batches
def test_padded_batches():
    seqs = [[1, 2, 3], [4], [5, 6, 7, 8, 9], [1, 1]]
    bs = list(batch.padded(seqs, batch_size=2, buckets=(4, 8), pad_id=0))
    assert [b["tokens"].shape for b in bs] == [(2, 4), (2, 4), (2, 8)]
    b0 = bs[0]
    assert list(b0["index"]) == [1, 3] and list(b0["tokens"][0]) == [0, 0, 0, 4]
    last = bs[-1]
    assert list(last["weights"]) == [1, 0] and last["index"][1] == -1
    assert (last["tokens"][1] == last["tokens"][0]).all()  # filler repeats row 0
    r = next(batch.padded([[1, 2]], 1, buckets=(4,), side="right"))
    assert list(r["tokens"][0]) == [1, 2, 0, 0] and list(r["mask"][0]) == [True, True, False, False]
    with pytest.raises(ValueError):
        list(batch.padded([[1] * 9], 1, buckets=(4, 8)))
    assert next(batch.padded([list(range(10))], 1, buckets=(4,), truncate=True))["tokens"].tolist() == [[6, 7, 8, 9]]


# ---------------------------------------------------------------- directions
def _batches(n=5, bs=2, T=6, seed=0):
    rng = np.random.default_rng(seed)
    seqs = [rng.integers(1, V, rng.integers(2, T + 1)).tolist() for _ in range(n)]
    return seqs, list(batch.padded(seqs, bs, buckets=(T,)))


def test_collect_matches_manual_mean_and_spills(tmp_path):
    seqs, bs = _batches()
    st = directions.collect(model, P, bs, at="layers.resid", positions=[-2, -1])
    rows = []
    for s in seqs:
        t = jnp.asarray([[0] * (6 - len(s)) + s])
        _, c = jp.run(model, P, t, capture="layers.resid")
        rows.append(c["layers.resid"][:, 0, -2:])
    np.testing.assert_allclose(st.mean, jnp.stack(rows).mean(0), rtol=1e-4, atol=1e-5)
    assert st.n == 5 and st.sum.shape == (L, 2, D)
    last = directions.collect(model, P, bs, at="layers.resid")
    assert last.sum.shape == (L, D)

    sp = io.Spill(tmp_path / "acts.npy", (5, L, 2, D), np.float32)
    st2 = directions.collect(model, P, bs, at="layers.resid", positions=[-2, -1], spill=sp, dtype=np.float32)
    np.testing.assert_allclose(st2.mean, st.mean, rtol=1e-5)
    assert sp.complete
    np.testing.assert_allclose(sp.array, np.stack(rows), rtol=1e-4, atol=1e-5)
    sp2 = io.Spill(tmp_path / "acts.npy", (5, L, 2, D), np.float32)  # resume: nothing left to do
    with pytest.raises(ValueError, match="already in the spill"):
        directions.collect(model, P, bs, at="layers.resid", positions=[-2, -1], spill=sp2, dtype=np.float32)
    with pytest.raises(jp.TapError):
        directions.collect(model, P, bs, at="layers.*")


def test_refusal_score():
    logits = jnp.log(jnp.array([[0.25, 0.25, 0.5]]))
    np.testing.assert_allclose(directions.refusal_score(logits, [2]), [0.0], atol=1e-5)
    np.testing.assert_allclose(directions.refusal_score(logits, [0]), [np.log(0.25 / 0.75)], rtol=1e-5)


def test_grid_and_select_pick_arditis_candidate():
    cands = jnp.arange(L * 2 * D, dtype=jnp.float32).reshape(L, 2, D)

    def score(layer, u):
        return {"bypass": -u[0], "induce": jnp.where(layer == 1, 1.0, 1.0), "kl": jnp.where(layer == 2, 1.0, 0.0)}
    s = directions.grid(score, cands, batch_size=2)
    assert s["bypass"].shape == (L, 2)
    best = directions.select(s, n_layers=L, max_layer_frac=1.0)
    assert (best["layer"], best["pos"]) == (1, 1)  # layer 2 fails the kl filter; lowest bypass = largest u[0]
    with pytest.raises(ValueError):
        directions.select({**s, "induce": -np.ones((L, 2))})


def test_arditi_score_runs_under_grid():
    g = jp.bind(model)
    last = lambda patch, t: g(patch, P, t)[0][:, -1]
    score = directions.arditi_score(last, TOK, TOK[::-1], [1, 2], ablate_at=["embed", "layers.mlp_out"],
                                    add_at="layers.resid")
    s = directions.grid(score, jax.random.normal(jax.random.key(1), (L, D)))
    assert set(s) == {"bypass", "induce", "kl"} and np.isfinite(s["kl"]).all() and (s["kl"] >= -1e-6).all()


def test_abliterate_equals_runtime_ablation():
    u = jax.random.normal(jax.random.key(7), (D,))
    edited = directions.abliterate(P, u, writers={"embed": -1, "w2": -1})
    assert directions.check_edit(model, P, edited, u, TOK, ablate_at=["embed", "layers.mlp_out"]) < 1e-4
    np.testing.assert_allclose(edited["unembed"], P["unembed"])
    with pytest.raises(KeyError):
        directions.abliterate(P, u, writers=["nope"])


# ---------------------------------------------------------------- probe
def test_probe_finds_the_direction():
    rng = np.random.default_rng(0)
    true = rng.normal(size=6)
    x = rng.normal(size=(400, 6)) * np.array([1, 5, 1, 1, 0.2, 1])
    y = (x @ true + 0.1 * rng.normal(size=400)) > 0
    pr = probe.fit_logistic(x, y, l2=1e-4)
    assert probe.auc(probe.score(pr, x), y) > 0.99
    assert float(probe.direction(pr) @ (true / np.linalg.norm(true))) > 0.95
    assert probe.auc([1, 2, 3, 3], [0, 0, 1, 0]) == pytest.approx(5 / 6)


# ---------------------------------------------------------------- io
def test_save_load_roundtrip_and_shards(tmp_path):
    tree = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": {"c": np.ones(4, ml_dtypes.bfloat16)},
            "layers": [np.zeros(3, np.int32), np.full(3, 7, np.int32)]}
    io.save(tmp_path / "t", tree)
    assert (tmp_path / "t.safetensors").exists() and not list(tmp_path.glob("*.tmp"))
    back = io.load(tmp_path / "t")
    np.testing.assert_array_equal(back["a"], tree["a"])
    assert back["b"]["c"].dtype == ml_dtypes.bfloat16 and isinstance(back["layers"], list)
    assert isinstance(back["a"], np.memmap) or isinstance(back["a"].base, np.memmap)
    fs = io.save(tmp_path / "t", tree, shard_size=30)
    assert len(fs) > 1 and (tmp_path / "t.safetensors.index.json").exists()
    assert not (tmp_path / "t.safetensors").exists()
    sel = io.load(tmp_path / "t", select="layers.*", flat=True)
    assert sorted(sel) == ["layers.0", "layers.1"]
    np.testing.assert_array_equal(io.load(tmp_path / "t")["layers"][1], tree["layers"][1])
    with pytest.raises(ValueError):
        io.save(tmp_path / "x", {"a.b": np.ones(1)})


def _hf_checkpoint(path, nshards=2):
    """A tiny HF-style checkpoint: PyTorch (out, in) layouts, sharded with an index."""
    rng = np.random.default_rng(0)
    t = {"model.embed_tokens.weight": rng.normal(size=(V, D)).astype(np.float32)}
    for i in range(L):
        t[f"model.layers.{i}.mlp.up_proj.weight"] = rng.normal(size=(H, D)).astype(ml_dtypes.bfloat16)
        t[f"model.layers.{i}.mlp.down_proj.weight"] = rng.normal(size=(D, H)).astype(np.float32)
        t[f"model.layers.{i}.self_attn.o_proj.weight"] = rng.normal(size=(D, D)).astype(ml_dtypes.bfloat16)
    t["lm_head.weight"] = rng.normal(size=(V, D)).astype(np.float32)
    t["model.layers.0.counts"] = np.arange(3, dtype=np.int64)
    path.mkdir()
    keys = list(t)
    wm = {}
    for s in range(nshards):
        name = f"model-{s + 1:05d}-of-{nshards:05d}.safetensors"
        part = {kk: t[kk] for kk in keys[s::nshards]}
        io.write_file(path / name, part)
        wm.update({kk: name for kk in part})
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    (path / "config.json").write_text(json.dumps({"tie_word_embeddings": False}))
    return t


def test_edit_checkpoint_abliterates_hf_layout(tmp_path):
    t = _hf_checkpoint(tmp_path / "src")
    u = np.random.default_rng(1).normal(size=D).astype(np.float32)
    edit, match = directions.hf_edit(u)
    rep = io.edit_checkpoint(tmp_path / "src", tmp_path / "dst", edit, match)
    assert len(rep["edited"]) == 1 + 2 * L and (tmp_path / "dst" / "config.json").exists()
    out = io.load(tmp_path / "dst", flat=True)
    uh = u / np.linalg.norm(u)
    for kk in rep["edited"]:
        axis = 0 if "proj" in kk else -1
        x = np.moveaxis(out[kk].astype(np.float32), axis, -1)
        assert np.abs(x @ uh).max() < 2e-2 if out[kk].dtype == ml_dtypes.bfloat16 else np.abs(x @ uh).max() < 1e-5
        assert out[kk].dtype == t[kk].dtype
    np.testing.assert_array_equal(out["lm_head.weight"], t["lm_head.weight"])
    np.testing.assert_array_equal(out["model.layers.0.mlp.up_proj.weight"], t["model.layers.0.mlp.up_proj.weight"])
    want = np.asarray(steer.orthogonalize(jnp.asarray(t["model.layers.1.mlp.down_proj.weight"]), u, axis=0))
    np.testing.assert_allclose(out["model.layers.1.mlp.down_proj.weight"], want, rtol=1e-5, atol=1e-5)
    with pytest.raises(TypeError):
        io.edit_checkpoint(tmp_path / "src", tmp_path / "d2", lambda n, a: a, "*counts")
    with pytest.raises(ValueError):
        io.edit_checkpoint(tmp_path / "src", tmp_path / "src", edit, match)


def test_stream_layers_matches_scan(tmp_path):
    # save each layer in HF-like naming, then stream it back one layer at a time
    io.save(tmp_path / "m", {"model": {"layers": [{"w1": np.asarray(P["w1"][i]), "w2": np.asarray(P["w2"][i])}
                                                   for i in range(L)]}})
    load_layer = io.layer_loader(tmp_path / "m", prefix="model.layers.{i}.")
    x0 = P["embed"][TOK]
    u = jnp.ones(D)
    patch = {"layers.resid": jp.at(1, steer.add(u, 2.0))}
    want, caps = jp.run(lambda p, t: jp.scan(lambda x, w: (layer(w, x), None), p["embed"][t],
                                             {"w1": p["w1"], "w2": p["w2"]})[0], P, TOK,
                        capture={"layers.resid": reduce.last}, patch=patch)
    got, scaps = io.stream_layers(layer, load_layer, L, x0, capture={"layers.resid": reduce.last}, patch=patch)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(scaps["layers.resid"], caps["layers.resid"], rtol=1e-5, atol=1e-5)
    seen = []
    io.stream_layers(layer, load_layer, L, x0, capture={"layers.resid": reduce.last}, prefetch=False,
                     on_layer=lambda i, c: seen.append((i, c["layers.resid"].shape)))
    assert seen == [(0, (2, D)), (1, (2, D)), (2, (2, D))]
