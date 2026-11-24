import json
import re

import jax
import numpy as np
import pytest

from jaxprobe import dash

RNG = np.random.default_rng(0)


def _codes(B, T, n, density=0.1, seed=0):
    r = np.random.default_rng(seed)
    z = r.gamma(2.0, 1.0, (B, T, n)).astype(np.float32)
    return np.where(r.random((B, T, n)) < density, z, 0.0).astype(np.float32)


def _collect(z, tokens, mask=None, step=None, **kw):
    col = dash.Collector(z.shape[-1], **kw)
    step = step or len(z)
    for i in range(0, len(z), step):
        col.update(z[i:i + step], tokens[i:i + step], None if mask is None else mask[i:i + step])
    return col.result()


def test_counts_hist_and_moments():
    z = _codes(6, 20, 7)
    tokens = RNG.integers(0, 50, (6, 20))
    mask = RNG.random((6, 20)) < 0.8
    s = _collect(z, tokens, mask)
    zm = np.where(mask[..., None], z, 0.0)
    fired = (zm > 0).sum((0, 1))
    assert s["n_tokens"] == mask.sum() and s["n_seqs"] == 6
    np.testing.assert_array_equal(s["fired"], fired)
    np.testing.assert_allclose(s["density"], fired / mask.sum())
    np.testing.assert_allclose(s["max"], zm.max((0, 1)), rtol=1e-6)
    np.testing.assert_allclose(s["mean"], zm.sum((0, 1)) / np.maximum(fired, 1), rtol=1e-5)
    np.testing.assert_array_equal(s["hist"].sum(1), fired)                       # every firing lands in one bin
    for f in range(7):                                                            # and in the right one
        v = np.log10(zm[..., f][zm[..., f] > 0])
        e = s["log10_edges"]
        want = np.histogram(np.clip(v, e[0], e[-1] - 1e-6), bins=e)[0]
        np.testing.assert_array_equal(s["hist"][f], want)


def test_top_contexts_match_brute_force_and_batching():
    B, T, n = 9, 15, 5
    z, tokens = _codes(B, T, n, 0.3, seed=1), RNG.integers(0, 99, (B, T))
    s = _collect(z, tokens, top=4, before=3, after=2)
    s2 = _collect(z, tokens, top=4, before=3, after=2, step=2)                 # streamed in pieces: same answer
    for k in ("top_value", "top_seq", "top_pos", "top_tokens", "top_acts", "hist", "fired"):
        np.testing.assert_array_equal(s[k], s2[k])
    for f in range(n):
        peaks = z[..., f].max(1)
        order = [b for b in np.argsort(-peaks, kind="stable") if peaks[b] > 0][:4]
        assert s["top_seq"][f][:len(order)].tolist() == order
        for k, b in enumerate(order):
            p = int(z[b, :, f].argmax())
            assert s["top_pos"][f, k] == p and s["top_value"][f, k] == pytest.approx(peaks[b])
            for w, t in enumerate(range(p - 3, p + 3)):
                if 0 <= t < T:
                    assert s["top_tokens"][f, k, w] == tokens[b, t]
                    assert s["top_acts"][f, k, w] == pytest.approx(z[b, t, f])
                else:
                    assert s["top_tokens"][f, k, w] == -1 and s["top_acts"][f, k, w] == 0


def test_dead_feature_and_padding():
    z = _codes(3, 8, 4, 0.5)
    z[..., 2] = 0.0
    s = _collect(z, np.zeros((3, 8), int), top=6)
    assert s["fired"][2] == 0 and (s["top_seq"][2] == -1).all()
    assert (s["top_seq"][0][:3] >= 0).all() and (s["top_seq"][0][3:] == -1).all()   # only 3 sequences exist
    c = dash.card(s, 0, str)
    assert len(c["contexts"]) == 3 and all(len(x["tokens"]) == len(x["acts"]) for x in c["contexts"])
    assert dash.card(s, 2, str)["contexts"] == []


def test_logit_effects_and_neighbours():
    W_dec, W_U = RNG.normal(size=(37, 8)), RNG.normal(size=(8, 50))
    e = dash.logit_effects(W_dec, W_U, top=5, chunk=10)
    full = W_dec @ W_U
    np.testing.assert_array_equal(e["pos_ids"], np.argsort(-full, 1)[:, :5])
    np.testing.assert_array_equal(e["neg_ids"], np.argsort(full, 1)[:, :5])
    np.testing.assert_allclose(e["pos_values"], np.sort(full, 1)[:, ::-1][:, :5], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(e["neg_values"], np.sort(full, 1)[:, :5], rtol=1e-4, atol=1e-5)
    W = W_dec.copy()
    W[5] = 3 * W[2]                                                              # a duplicate direction
    nb = dash.neighbours(W, top=3, chunk=8)
    U = W / np.linalg.norm(W, axis=1, keepdims=True)
    C = U @ U.T
    np.fill_diagonal(C, -np.inf)
    np.testing.assert_allclose(nb["values"], np.sort(C, 1)[:, ::-1][:, :3], atol=1e-5)    # ids can tie (2 and 5)
    np.testing.assert_allclose(np.take_along_axis(C, nb["ids"], 1), nb["values"], atol=1e-5)
    assert nb["ids"][2, 0] == 5 and nb["values"][2, 0] == pytest.approx(1.0, abs=1e-5)
    assert (nb["ids"] != np.arange(37)[:, None]).all()


def test_feature_map_pca():
    W = RNG.normal(size=(30, 6))
    W[7] = 2 * W[3]
    xy = dash.feature_map(W)
    assert xy.shape == (30, 2)
    np.testing.assert_allclose(xy[7], xy[3], atol=1e-5)                          # same direction, same place
    U = W / np.linalg.norm(W, axis=1, keepdims=True)
    var = np.linalg.svd(U - U.mean(0), compute_uv=False)[:2] ** 2
    np.testing.assert_allclose(np.sort((xy ** 2).sum(0))[::-1], var, rtol=1e-4)  # the top two components
    with pytest.raises(ValueError):
        dash.feature_map(W, method="tsne")


def test_html_and_neuronpedia(tmp_path):
    z, tokens = _codes(4, 10, 6, 0.4), RNG.integers(0, 20, (4, 10))
    z[..., 1] = 0.0
    s = _collect(z, tokens, top=3)
    W = RNG.normal(size=(6, 5))
    eff = dash.logit_effects(W, RNG.normal(size=(5, 20)), top=4)
    nb = dash.neighbours(W, top=2)
    decode = lambda i: f"<t{i}>"                                                 # needs escaping in HTML
    xy = dash.feature_map(W)
    xy[3] = np.nan                                                               # no place on the map: kept off it
    p = dash.to_html(tmp_path / "f.html", s, decode, effects=eff, neighbours=nb, coords=xy,
                     cofire=dash.top_pairs(RNG.random((6, 6)), top=2), labels={0: "zeroth </script>"})
    page = p.read_text()
    blob = re.search(r"const D=(.*?);const V=", page, re.DOTALL).group(1)
    data = json.loads(blob.replace("<\\/", "</"))
    assert [f["i"] for f in data["features"]] == [0, 2, 3, 4, 5]                 # the dead feature is left out
    f0 = data["features"][0]
    assert len(f0["c"]) == 3 and max(max(a) for _, a in f0["c"]) == 255
    assert f0["l"] == "zeroth </script>" and "</script>" not in blob
    assert "xy" not in data["features"][2] and "xy" in data["features"][1] and len(f0["cf"]) == 2
    assert data["vocab"][f0["p"][0][0]] == decode(eff["pos_ids"][0, 0])
    nbatch = dash.to_neuronpedia(s, decode, model_id="m", layer=3, sae_set="s", effects=eff)
    feat = nbatch["features"][0]
    assert nbatch["layer"] == 3 and feat["feature_index"] == 0 and len(feat["pos_str"]) == 4
    assert sum(feat["freq_hist_data_bar_heights"]) == s["fired"][0]
    a = feat["activations"][0]
    assert a["bin_min"] == -1 and a["values"][a["qualifying_token_index"]] == pytest.approx(s["top_value"][0, 0])
    json.dumps(nbatch)


def test_coactivation():
    z, tokens = _codes(5, 12, 6, 0.3, seed=4), RNG.integers(0, 9, (5, 12))
    z[..., 4] = 0.0                                                              # never fires
    z[..., 5] = 1.0                                                              # always fires
    mask = RNG.random((5, 12)) < 0.9
    s = _collect(z, tokens, mask, step=2, cooc=True)
    fired = ((z > 0) & mask[..., None]).reshape(-1, 6).astype(float)[mask.reshape(-1)]
    np.testing.assert_array_equal(s["cooc"], fired.T @ fired)
    phi = dash.coactivation(s)
    want = np.corrcoef(fired[:, :4].T)
    np.fill_diagonal(want, 0)
    np.testing.assert_allclose(phi[:4, :4], want, atol=1e-5)
    assert (phi[4] == 0).all() and (phi[:, 5] == 0).all() and phi.dtype == np.float32
    tp = dash.top_pairs(phi, top=2)
    for f in range(4):
        assert f not in tp["ids"][f]
        np.testing.assert_allclose(tp["values"][f], np.sort(np.delete(phi[f], f))[::-1][:2])
    with pytest.raises(ValueError):
        dash.coactivation(_collect(z, tokens))
    c = dash.card(s, 0, str, cofire=tp)
    assert all(v > 0 for _, v in c["fires_with"])


def test_update_compiles_once():
    col = dash.Collector(4, top=2)
    col.update(_codes(2, 6, 4), np.zeros((2, 6), int))
    n = dash._update._cache_size()
    for i in range(1, 3):
        col.update(_codes(2, 6, 4, seed=i), np.zeros((2, 6), int))
    assert dash._update._cache_size() == n
    assert jax.tree.map(lambda x: x.shape, col.state)["top_tokens"] == (4, 2, 17)
