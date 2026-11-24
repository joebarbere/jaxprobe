"""Feature dashboards for sparse autoencoders and transcoders: what makes each feature fire, and what it does.

Stream the codes of any feature dictionary over a dataset once, with `Collector`, and get per-feature firing
statistics, an activation histogram, the top-activating contexts (one per sequence, with the activations
around the peak) and, optionally, how often each pair of features fires together. Add what the weights say on
their own (`logit_effects`, `neighbours`), lay the features out on a map (`feature_map`), and write one
self-contained HTML page (`to_html`) or Neuronpedia's dashboard JSON (`to_neuronpedia`).

    col = jaxprobe.dash.Collector(n_features=8192, cooc=True)
    for tokens in batches:                                     # (B, T) int
        acts = capture(params, tokens)                          # (B, T, d), e.g. from jaxprobe.run
        col.update(jaxprobe.sae.encode(sae, acts * scale, "topk", 32), tokens)
    stats = col.result()
    eff = jaxprobe.dash.logit_effects(sae["W_dec"], W_U)        # W_U: (d, vocab), final norm's gain folded in
    nb = jaxprobe.dash.neighbours(sae["W_dec"])                 # similar decoder directions
    phi = jaxprobe.dash.coactivation(stats)                     # which features fire together
    xy = jaxprobe.dash.feature_map(phi.clip(0), method="umap")  # or feature_map(sae["W_dec"]): no cooc needed
    jaxprobe.dash.to_html("features.html", stats, decode, effects=eff, neighbours=nb,
                          cofire=jaxprobe.dash.top_pairs(phi), coords=xy)

Nothing here depends on the model: codes and tokens go in, numbers and pages come out.

Which map? On a 8,192-feature TopK SAE of SmolLM2-135M (learn-jax module 47), features specific to one of three
text domains had 10 map neighbours from the same domain 88% of the time on a UMAP of co-activation, 77% on a PCA
of co-activation, 75% on a UMAP of decoder directions, against 66% by chance; the decoder PCA was one round
blob. Decoder directions of a wide SAE are nearly orthogonal, so where features are used says more than where
they point.

References: Bricken et al. 2023, "Towards Monosemanticity" (feature dashboards: top contexts, logit effects,
activation density); SAEDashboard (Decode Research, MIT) for the Neuronpedia JSON layout.
"""
from __future__ import annotations

import functools
import html
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# log10 bin edges for the activation histogram: 1e-4 … 1e3 in steps of 0.125 decades (activations above or below
# land in the outermost bins)
LOG10_EDGES = np.arange(-4.0, 3.0 + 1e-9, 0.125, dtype=np.float32)


def _empty(n: int, top: int, width: int, nbins: int, cooc: bool):
    out = {"n_tokens": jnp.zeros((), jnp.float32), "n_seqs": jnp.zeros((), jnp.int32),
            "fired": jnp.zeros(n, jnp.float32), "sum": jnp.zeros(n, jnp.float32), "max": jnp.zeros(n, jnp.float32),
            "fired_sq": jnp.zeros(n, jnp.float32), "fired_m": jnp.zeros(n, jnp.float32),
            "tok_sq": jnp.zeros((), jnp.float32),
            "hist": jnp.zeros((n, nbins), jnp.float32),
            "top_value": jnp.zeros((n, top), jnp.float32), "top_seq": jnp.full((n, top), -1, jnp.int32),
            "top_pos": jnp.zeros((n, top), jnp.int32), "top_tokens": jnp.full((n, top, width), -1, jnp.int32),
            "top_acts": jnp.zeros((n, top, width), jnp.float32)}
    if cooc:
        out["cooc"] = jnp.zeros((n, n), jnp.float32)
    return out


@functools.partial(jax.jit, static_argnames=("before", "after"), donate_argnums=(0,))
def _update(s, z, tokens, mask, edges, *, before: int, after: int):
    B, T, n = z.shape
    top = s["top_value"].shape[1]
    z = jnp.where(mask[..., None], z.astype(jnp.float32), 0.0)
    pos = z > 0
    out = dict(s)
    out["n_tokens"] = s["n_tokens"] + mask.sum()
    out["n_seqs"] = s["n_seqs"] + B
    out["fired"] = s["fired"] + pos.sum((0, 1))
    c, m = pos.sum(1).astype(jnp.float32), mask.sum(1).astype(jnp.float32)          # per sequence: (B, n), (B,)
    out["fired_sq"] = s["fired_sq"] + (c * c).sum(0)                               # for burst-aware drift tests
    out["fired_m"] = s["fired_m"] + (c * m[:, None]).sum(0)
    out["tok_sq"] = s["tok_sq"] + (m * m).sum()
    out["sum"] = s["sum"] + z.sum((0, 1))
    out["max"] = jnp.maximum(s["max"], z.max((0, 1)))
    # histogram of log10(activation) over the tokens where the feature fired: count >= each edge, then difference
    lz = jnp.where(pos, jnp.log10(jnp.where(pos, z, 1.0)), -jnp.inf)
    ge = jax.lax.map(lambda e: (lz >= e).sum((0, 1)), edges[1:-1])                    # (nbins - 1, n)
    ge = jnp.concatenate([pos.sum((0, 1))[None], ge, jnp.zeros((1, n), ge.dtype)])   # edge 0 catches everything
    out["hist"] = s["hist"] + (ge[:-1] - ge[1:]).T
    if "cooc" in s:                                    # exact counts: 0/1 in bf16, summed in float32
        fb = pos.reshape(-1, n).astype(jnp.bfloat16)
        out["cooc"] = s["cooc"] + jnp.dot(fb.T, fb, preferred_element_type=jnp.float32)
    # each sequence's peak for each feature, and the window around it
    peak = z.max(1)                                                                   # (B, n)
    at = z.argmax(1)                                                                  # (B, n)
    offs = jnp.arange(-before, after + 1)
    idx = at[..., None] + offs                                                        # (B, n, W)
    ok = (idx >= 0) & (idx < T)
    idx_c = jnp.clip(idx, 0, T - 1)
    win_tok = jnp.where(ok, jnp.take_along_axis(tokens[:, None, :], idx_c, axis=2), -1)
    win_act = jnp.where(ok, jnp.take_along_axis(jnp.swapaxes(z, 1, 2), idx_c, axis=2), 0.0)
    seq = s["n_seqs"] + jnp.arange(B, dtype=jnp.int32)
    # merge with the running top: (n, top + B) candidates per feature
    cv = jnp.concatenate([s["top_value"], peak.T], 1)
    cs = jnp.concatenate([s["top_seq"], jnp.broadcast_to(jnp.where(peak > 0, seq[:, None], -1).T, (n, B))], 1)
    cp = jnp.concatenate([s["top_pos"], at.T], 1)
    ct = jnp.concatenate([s["top_tokens"], jnp.swapaxes(win_tok, 0, 1)], 1)
    ca = jnp.concatenate([s["top_acts"], jnp.swapaxes(win_act, 0, 1)], 1)
    _, o = jax.lax.top_k(jnp.where(cs >= 0, cv, -1.0), top)                           # ties: the earlier entry
    take = functools.partial(jnp.take_along_axis, indices=o, axis=1)
    out["top_value"], out["top_seq"], out["top_pos"] = take(cv), take(cs), take(cp)
    out["top_tokens"] = jnp.take_along_axis(ct, o[..., None], axis=1)
    out["top_acts"] = jnp.take_along_axis(ca, o[..., None], axis=1)
    return out


class Collector:
    """Accumulates dashboard statistics over batches of codes.

    `update(z, tokens, mask=None)` takes codes z (B, T, n_features), the tokens they were computed on (B, T) and
    an optional mask (B, T) of positions to count (padding, or outlier tokens you'd rather skip). For every
    feature it keeps: how many tokens it fired on, the sum and maximum of its activation, a histogram of
    log10(activation) where it fired, and its `top` highest-activating sequences (one context per sequence, the
    `before` tokens before the peak and `after` tokens after it, with the feature's activation at each).
    With `cooc=True` it also counts, for every pair of features, the tokens where both fired: an (n, n) float32
    matrix on the device (268 MB for 8,192 features; past ~20k features, collect it for a subset yourself).
    Everything stays on the device as fixed-shape arrays, so each update is one compiled call.
    """

    def __init__(self, n_features: int, top: int = 16, before: int = 12, after: int = 4,
                 log10_edges: Sequence[float] = LOG10_EDGES, cooc: bool = False):
        self.edges = jnp.asarray(np.asarray(log10_edges, np.float32))
        self.before, self.after = before, after
        self.state = _empty(n_features, top, before + after + 1, len(self.edges) - 1, cooc)

    def update(self, z, tokens, mask=None):
        tokens = jnp.asarray(tokens, jnp.int32)
        mask = jnp.ones(tokens.shape, bool) if mask is None else jnp.asarray(mask, bool)
        self.state = _update(self.state, jnp.asarray(z), tokens, mask, self.edges, before=self.before,
                             after=self.after)

    def result(self) -> dict:
        """NumPy arrays: n_tokens, n_seqs, fired, density, mean (over firing tokens), max, hist (n, bins),
        log10_edges, top_value / top_seq / top_pos (n, top), top_tokens / top_acts (n, top, window) with -1 / 0
        padding, `before` (the peak's index in each window, where the window isn't cut by a sequence edge), and
        `cooc` (n, n) if it was collected; fired_sq, fired_m and tok_sq are per-sequence sums (Σc², Σc·m, Σm²
        over sequences, c = a feature's firings and m = real tokens in the sequence) for `monitor.drift`."""
        s = jax.tree.map(np.asarray, self.state)
        n_tok = float(s["n_tokens"])
        return {"n_tokens": n_tok, "n_seqs": int(s["n_seqs"]), "fired": s["fired"],
                "density": s["fired"] / max(n_tok, 1.0), "mean": s["sum"] / np.maximum(s["fired"], 1), "max": s["max"],
                "hist": s["hist"], "log10_edges": np.asarray(self.edges), "top_value": s["top_value"],
                "top_seq": s["top_seq"], "top_pos": s["top_pos"], "top_tokens": s["top_tokens"],
                "top_acts": s["top_acts"], "before": self.before, "after": self.after,
                "fired_sq": s["fired_sq"], "fired_m": s["fired_m"], "tok_sq": float(s["tok_sq"])} | (
                    {"cooc": s["cooc"]} if "cooc" in s else {})


def coactivation(stats: dict) -> np.ndarray:
    """The phi coefficient between features' firing (the Pearson correlation of the two "fired" indicators over
    tokens), (n, n), from a Collector run with cooc=True. 0 on the diagonal, and for features that never fired
    or always fired. Positive = they fire on the same tokens more often than chance."""
    if "cooc" not in stats:
        raise ValueError("coactivation needs Collector(..., cooc=True)")
    N = max(stats["n_tokens"], 1.0)
    p = np.asarray(stats["fired"], np.float64) / N
    v = p * (1 - p)
    ok = v > 0
    phi = (stats["cooc"] / N - np.outer(p, p)) / np.sqrt(np.outer(np.where(ok, v, 1.0), np.where(ok, v, 1.0)))
    phi[~ok] = 0.0
    phi[:, ~ok] = 0.0
    np.fill_diagonal(phi, 0.0)
    return phi.astype(np.float32)


def top_pairs(M, top: int = 10) -> dict:
    """Each row's `top` largest entries of a square similarity matrix (e.g. `coactivation`), the diagonal
    excluded: ids and values, each (n, top). The same layout as `neighbours`."""
    M = np.array(M, np.float32)
    np.fill_diagonal(M, -np.inf)
    top = min(top, M.shape[1] - 1)
    ids = np.argpartition(-M, top - 1, axis=1)[:, :top]
    ids = np.take_along_axis(ids, np.argsort(-np.take_along_axis(M, ids, 1), axis=1, kind="stable"), 1)
    return {"ids": ids, "values": np.take_along_axis(M, ids, 1)}


def logit_effects(W_dec, W_U, top: int = 10, chunk: int = 1024) -> dict:
    """Each feature's direct effect on the logits: its decoder row through the unembedding, W_dec @ W_U.

    W_dec (n, d); W_U (d, vocab), with the final norm's gain already multiplied into its rows if the model has
    one (this ignores the norm's scaling by the token's own size, so it's the direction of the effect, not its
    size in any one context). Returns the `top` most promoted and most suppressed token ids and their values,
    each (n, top). Computed in chunks of features, so the (n, vocab) matrix never exists at once."""
    W_dec, W_U = jnp.asarray(W_dec, jnp.float32), jnp.asarray(W_U, jnp.float32)

    @jax.jit
    def one(w):
        e = w @ W_U
        pv, pi = jax.lax.top_k(e, top)
        nv, ni = jax.lax.top_k(-e, top)
        return pi, pv, ni, -nv

    parts = [one(W_dec[i:i + chunk]) for i in range(0, W_dec.shape[0], chunk)]
    pi, pv, ni, nv = (np.concatenate([np.asarray(p[j]) for p in parts]) for j in range(4))
    return {"pos_ids": pi, "pos_values": pv, "neg_ids": ni, "neg_values": nv}


def neighbours(W_dec, top: int = 10, chunk: int = 2048) -> dict:
    """Each feature's `top` nearest features by cosine similarity of decoder rows, itself excluded.
    Returns ids and cosines, each (n, top). Near-duplicates (cosine close to 1) are often split features."""
    W = jnp.asarray(W_dec, jnp.float32)
    W = W / jnp.maximum(jnp.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    n = W.shape[0]

    @jax.jit
    def one(w, start):
        c = w @ W.T
        c = jnp.where(jnp.arange(n)[None] == (start + jnp.arange(w.shape[0]))[:, None], -jnp.inf, c)
        v, i = jax.lax.top_k(c, top)
        return i, v

    parts = [one(W[i:i + chunk], i) for i in range(0, n, chunk)]
    return {"ids": np.concatenate([np.asarray(p[0]) for p in parts]),
            "values": np.concatenate([np.asarray(p[1]) for p in parts])}


def feature_map(rows, method: str = "pca", dims: int = 2, seed: int = 0, **umap_kwargs) -> np.ndarray:
    """Coordinates (n, dims) for a map of the features, one per row of `rows`, after normalizing each row:
    decoder directions (W_dec) or co-activation profiles (`coactivation(stats).clip(0)`, usually better).

    "pca": the top principal directions (JAX, exact, fast; shows the broad axes, crowds the rest).
    "umap": UMAP with cosine distance (needs `pip install umap-learn`; about a minute for 8,192 features; keeps
    local neighbourhoods, so related features form islands, but distances between islands mean little)."""
    W = np.asarray(rows, np.float32)
    W = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    if method == "pca":
        X = jnp.asarray(W - W.mean(0))
        _, _, vt = jnp.linalg.svd(X, full_matrices=False)
        return np.array(X @ vt[:dims].T)
    if method == "umap":
        try:
            import umap
        except ImportError as e:
            raise ImportError("feature_map(method='umap') needs umap-learn: pip install umap-learn") from e
        kw = {"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine", "random_state": seed} | umap_kwargs
        return np.asarray(umap.UMAP(n_components=dims, **kw).fit_transform(W), np.float32)
    raise ValueError(f"method must be 'pca' or 'umap', got {method!r}")


def _decoder(decode: Callable[[int], str]):
    cache: dict[int, str] = {}

    def piece(i: int) -> str:
        i = int(i)
        if i not in cache:
            cache[i] = decode(i)
        return cache[i]
    return piece


def card(stats: dict, f: int, decode: Callable[[int], str], *, effects: dict | None = None,
         neighbours: dict | None = None, cofire: dict | None = None, n_contexts: int | None = None) -> dict:
    """One feature's dashboard as plain Python data (for a notebook, a test, or your own page)."""
    piece = _decoder(decode)
    ctx = []
    for k in range(stats["top_value"].shape[1] if n_contexts is None else n_contexts):
        if stats["top_seq"][f, k] < 0:
            break
        keep = stats["top_tokens"][f, k] >= 0
        ctx.append({"seq": int(stats["top_seq"][f, k]), "pos": int(stats["top_pos"][f, k]),
                    "value": float(stats["top_value"][f, k]),
                    "tokens": [piece(t) for t in stats["top_tokens"][f, k][keep]],
                    "acts": [float(a) for a in stats["top_acts"][f, k][keep]],
                    "peak": int(np.argmax(stats["top_acts"][f, k][keep]))})
    out = {"feature": int(f), "density": float(stats["density"][f]), "mean": float(stats["mean"][f]),
           "max": float(stats["max"][f]), "hist": stats["hist"][f].tolist(), "contexts": ctx}
    if effects is not None:
        out["promotes"] = [(piece(i), float(v)) for i, v in zip(effects["pos_ids"][f], effects["pos_values"][f])]
        out["suppresses"] = [(piece(i), float(v)) for i, v in zip(effects["neg_ids"][f], effects["neg_values"][f])]
    if neighbours is not None:
        out["neighbours"] = [(int(i), float(v)) for i, v in zip(neighbours["ids"][f], neighbours["values"][f])]
    if cofire is not None:
        out["fires_with"] = [(int(i), float(v)) for i, v in zip(cofire["ids"][f], cofire["values"][f]) if v > 0]
    return out


def _payload(stats, decode, effects, neighbours, cofire, coords, features, labels, n_contexts):
    piece = _decoder(decode)
    vocab: dict[int, int] = {}
    strings: list[str] = []

    def vid(t):
        t = int(t)
        if t not in vocab:
            vocab[t] = len(strings)
            strings.append(piece(t))
        return vocab[t]

    feats = []
    for f in features:
        mx = float(stats["max"][f]) or 1.0
        ctx = []
        for k in range(min(n_contexts, stats["top_value"].shape[1])):
            if stats["top_seq"][f, k] < 0:
                break
            keep = stats["top_tokens"][f, k] >= 0
            ctx.append([[vid(t) for t in stats["top_tokens"][f, k][keep]],
                        np.rint(255 * np.clip(stats["top_acts"][f, k][keep], 0, None) / mx).astype(int).tolist()])
        rec = {"i": int(f), "d": float(stats["density"][f]), "m": round(mx, 4), "a": round(float(stats["mean"][f]), 4),
               "h": [int(h) for h in stats["hist"][f]], "c": ctx}
        if effects is not None:
            rec["p"] = [[vid(i), round(float(v), 3)] for i, v in zip(effects["pos_ids"][f], effects["pos_values"][f])]
            rec["n"] = [[vid(i), round(float(v), 3)] for i, v in zip(effects["neg_ids"][f], effects["neg_values"][f])]
        if neighbours is not None:
            rec["nb"] = [[int(i), round(float(v), 3)] for i, v in zip(neighbours["ids"][f], neighbours["values"][f])]
        if cofire is not None:
            rec["cf"] = [[int(i), round(float(v), 3)] for i, v in zip(cofire["ids"][f], cofire["values"][f]) if v > 0]
        if coords is not None and np.isfinite(coords[f][:2]).all():
            rec["xy"] = [round(float(c), 4) for c in coords[f][:2]]
        if labels is not None and f in labels:
            rec["l"] = str(labels[f])
        feats.append(rec)
    return {"vocab": strings, "edges": [round(float(e), 4) for e in stats["log10_edges"]],
            "n_tokens": stats["n_tokens"], "features": feats}


def to_html(path, stats: dict, decode: Callable[[int], str], *, effects: dict | None = None,
            neighbours: dict | None = None, cofire: dict | None = None, coords=None, features: Sequence[int] | None = None,
            labels: dict | None = None, n_contexts: int = 8, title: str = "Feature dashboard") -> Path:
    """Write one self-contained HTML page (no network access needed to open it): a map of the features (from
    `coords`, coloured by firing density; a list sorted by density without it), a search box, and a card per
    feature with its histogram, top contexts (tokens shaded by activation), logit effects, neighbours by decoder
    direction and the features it fires with (`cofire`, from `top_pairs(coactivation(stats))`).

    `features` limits the page to some features (default: every feature that fired). Activations are stored
    as 0–255 of each feature's max, so the page stays small: about 1 KB per feature with 8 contexts."""
    if features is None:
        features = np.flatnonzero(stats["fired"] > 0).tolist()
    data = _payload(stats, decode, effects, neighbours, cofire, None if coords is None else np.asarray(coords),
                    list(features), labels, n_contexts)
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    page = _PAGE.replace("__TITLE__", html.escape(title)).replace("__DATA__", blob)
    path = Path(path)
    path.write_text(page, encoding="utf-8")
    return path


def to_neuronpedia(stats: dict, decode: Callable[[int], str], *, model_id: str, layer: int, sae_set: str,
                   effects: dict | None = None, features: Sequence[int] | None = None, n_contexts: int = 16,
                   dataset: str = "") -> dict:
    """A dashboard batch in the JSON layout SAEDashboard writes for Neuronpedia uploads (model_id, layer, sae_set,
    features[...] with pos_str/pos_values, neg_str/neg_values, frac_nonzero, freq_hist_data_bar_*, activations).
    Fields jaxprobe doesn't compute (neuron alignment, correlations, logit histograms) are left empty. Check
    Neuronpedia's current upload instructions before relying on it; this follows SAEDashboard's export classes.
    """
    piece = _decoder(decode)
    if features is None:
        features = np.flatnonzero(stats["fired"] > 0).tolist()
    centres = 10 ** ((stats["log10_edges"][:-1] + stats["log10_edges"][1:]) / 2)
    out = []
    for f in features:
        acts = []
        for k in range(min(n_contexts, stats["top_value"].shape[1])):
            if stats["top_seq"][f, k] < 0:
                break
            keep = stats["top_tokens"][f, k] >= 0
            vals = [float(a) for a in stats["top_acts"][f, k][keep]]
            acts.append({"bin_min": -1, "bin_max": 99, "bin_contains": -1,
                         "tokens": [piece(t) for t in stats["top_tokens"][f, k][keep]], "values": vals,
                         "qualifying_token_index": int(np.argmax(vals))})
        h = stats["hist"][f]
        nz = np.flatnonzero(h)
        rec = {"feature_index": int(f), "neuron_alignment_indices": [], "neuron_alignment_values": [],
               "neuron_alignment_l1": [], "correlated_neurons_indices": [], "correlated_neurons_l1": [],
               "correlated_neurons_pearson": [], "correlated_features_indices": [], "correlated_features_l1": [],
               "correlated_features_pearson": [], "neg_str": [], "neg_values": [], "pos_str": [], "pos_values": [],
               "frac_nonzero": float(stats["density"][f]),
               "freq_hist_data_bar_values": [round(float(c), 4) for c in centres[nz]],
               "freq_hist_data_bar_heights": [int(x) for x in h[nz]],
               "logits_hist_data_bar_heights": [], "logits_hist_data_bar_values": [],
               "n_prompts_total": int(stats["n_seqs"]), "n_tokens_in_prompt": 0, "dataset": dataset,
               "decoder_weights_dist": [], "activations": acts, "vector": []}
        if effects is not None:
            rec["pos_str"] = [piece(i) for i in effects["pos_ids"][f]]
            rec["pos_values"] = [float(v) for v in effects["pos_values"][f]]
            rec["neg_str"] = [piece(i) for i in effects["neg_ids"][f]]
            rec["neg_values"] = [float(v) for v in effects["neg_values"][f]]
        out.append(rec)
    return {"model_id": model_id, "layer": int(layer), "sae_set": sae_set, "sae_id_suffix": None, "features": out}


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fbfbf9;--fg:#1d1d1b;--mut:#6b6b66;--line:#deded8;--card:#fff;--hi:31,111,235;--pos:#1a7f37;--neg:#b42318}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#e8e8e3;--mut:#9b9b94;--line:#33332f;--card:#1e1e1c;
--hi:88,166,255;--pos:#4ac26b;--neg:#f47067}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0}input{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;
background:var(--card);color:var(--fg);width:12em}.mut{color:var(--mut)}
main{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.1fr);gap:16px;padding:16px}
@media (max-width:860px){main{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;min-width:0}
canvas{width:100%;aspect-ratio:1/1;display:block;cursor:crosshair}
#list{max-height:70vh;overflow:auto}#list div{cursor:pointer;padding:2px 4px;border-radius:4px}
#list div:hover{background:rgba(var(--hi),.12)}
.ctx{font-family:ui-monospace,monospace;font-size:12.5px;white-space:pre-wrap;word-break:break-word;
padding:4px 0;border-bottom:1px solid var(--line)}.ctx span{border-radius:2px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:12px}.tok{font-family:ui-monospace,monospace}
.pos{color:var(--pos)}.neg{color:var(--neg)}a{color:rgb(var(--hi));cursor:pointer}
h2{font-size:15px;margin:0 0 6px}h3{font-size:13px;margin:12px 0 4px;color:var(--mut);font-weight:600}
svg{width:100%;height:70px;display:block}
</style></head><body>
<header><h1>__TITLE__</h1><input id="q" placeholder="feature id" inputmode="numeric">
<span class="mut" id="info"></span></header>
<main><div class="panel"><canvas id="map" width="800" height="800"></canvas><div id="list"></div>
<p class="mut" id="legend"></p></div><div class="panel" id="card"><p class="mut">Pick a feature.</p></div></main>
<script>
const D=__DATA__;const V=D.vocab,F=D.features,byId=new Map(F.map(f=>[f.i,f]));
const $=s=>document.querySelector(s);const esc=s=>s.replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const tok=i=>esc(V[i]).replace(/\n/g,"⏎");
$("#info").textContent=`${F.length} features · ${Math.round(D.n_tokens).toLocaleString()} tokens`;
const P=F.filter(f=>f.xy),hasXY=P.length>0;let cur=null;
function color(d){const t=Math.min(1,Math.max(0,(Math.log10(d)+5)/4));return `hsl(${220-200*t},70%,50%)`}
const cv=$("#map"),cx=cv.getContext("2d");let box;
function draw(){if(!hasXY){cv.style.display="none";return}
 const xs=P.map(f=>f.xy[0]),ys=P.map(f=>f.xy[1]);box=[Math.min(...xs),Math.max(...xs),Math.min(...ys),Math.max(...ys)];
 cx.clearRect(0,0,800,800);for(const f of P){const [x,y]=px(f);cx.fillStyle=color(f.d);cx.globalAlpha=.55;
 cx.fillRect(x-1.5,y-1.5,3,3)}cx.globalAlpha=1;if(cur&&cur.xy){const [x,y]=px(cur);cx.strokeStyle=getComputedStyle(document.body).color;
 cx.lineWidth=2;cx.beginPath();cx.arc(x,y,7,0,7);cx.stroke()}
 $("#legend").textContent="Each dot is a feature; nearby dots are related (by the map you chose). Colour = firing density (blue rare, red common). Click a dot."}
function px(f){return[20+760*(f.xy[0]-box[0])/((box[1]-box[0])||1),780-760*(f.xy[1]-box[2])/((box[3]-box[2])||1)]}
cv.addEventListener("click",e=>{const r=cv.getBoundingClientRect(),x=(e.clientX-r.left)*800/r.width,y=(e.clientY-r.top)*800/r.height;
 let best=null,bd=1e9;for(const f of P){const [a,b]=px(f),d=(a-x)**2+(b-y)**2;if(d<bd){bd=d;best=f}}if(best)show(best.i)});
if(!hasXY){$("#list").innerHTML=[...F].sort((a,b)=>b.d-a.d).slice(0,500).map(f=>`<div data-i="${f.i}">#${f.i}
 <span class="mut">${(100*f.d).toFixed(3)}%</span> ${f.l?esc(f.l):""}</div>`).join("");
 $("#list").addEventListener("click",e=>{const i=e.target.closest("div")?.dataset.i;if(i)show(+i)})}
function hist(h){const m=Math.max(...h,1);let lo=h.findIndex(x=>x>0),hi=h.length-1-[...h].reverse().findIndex(x=>x>0);
 if(lo<0)return"";const w=100/(hi-lo+1);let s=`<svg viewBox="0 0 100 40" preserveAspectRatio="none">`;
 for(let k=lo;k<=hi;k++)s+=`<rect x="${(k-lo)*w}" y="${40-38*h[k]/m}" width="${w*0.9}" height="${38*h[k]/m}" fill="rgb(var(--hi))"/>`;
 return s+`</svg><div class="mut">activation 10<sup>${D.edges[lo]}</sup> … 10<sup>${D.edges[hi+1]}</sup> (log scale)</div>`}
function show(i){const f=byId.get(i);if(!f){$("#card").innerHTML=`<p class="mut">Feature ${i} isn't on this page.</p>`;return}
 cur=f;draw();location.hash=i;let s=`<h2>Feature ${f.i}${f.l?" · "+esc(f.l):""}</h2><div class="mut">fires on
 ${(100*f.d).toFixed(4)}% of tokens · mean ${f.a} · max ${f.m}</div><h3>Activation histogram</h3>${hist(f.h)}<h3>Top contexts</h3>`;
 for(const [t,a] of f.c)s+=`<div class="ctx">`+t.map((v,k)=>`<span style="background:rgba(var(--hi),${(a[k]/255*.85).toFixed(2)})"
 title="${(a[k]/255*f.m).toFixed(2)}">${tok(v)}</span>`).join("")+`</div>`;
 if(f.p)s+=`<div class="cols"><div><h3>Promotes</h3>${f.p.map(([v,x])=>`<div><span class="tok">${tok(v)}</span>
 <span class="pos">+${x}</span></div>`).join("")}</div><div><h3>Suppresses</h3>${f.n.map(([v,x])=>`<div><span class="tok">${tok(v)}</span>
 <span class="neg">${x}</span></div>`).join("")}</div></div>`;
 if(f.cf&&f.cf.length)s+=`<h3>Fires with (φ correlation of firing)</h3>`+f.cf.map(([j,c])=>`<a data-j="${j}">#${j}</a> <span class="mut">${c}</span>`).join(" · ");
 if(f.nb)s+=`<h3>Nearest features (decoder cosine)</h3>`+f.nb.map(([j,c])=>`<a data-j="${j}">#${j}</a> <span class="mut">${c}</span>`).join(" · ");
 $("#card").innerHTML=s;$("#card").querySelectorAll("a[data-j]").forEach(a=>a.onclick=()=>show(+a.dataset.j))}
$("#q").addEventListener("change",e=>show(+e.target.value));draw();
const h0=+location.hash.slice(1);if(byId.has(h0))show(h0);else if(F.length)show([...F].sort((a,b)=>b.d-a.d)[0].i);
</script></body></html>
"""
