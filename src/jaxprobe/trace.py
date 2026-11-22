"""Attribution graphs: which features (and which errors) a model's output depends on, on one prompt.

The method is Ameisen et al. 2025, "Circuit Tracing: Revealing Computational Graphs in Language Models"
(transformer-circuits.pub/2025/attribution-graphs), with per-layer transcoders (Dunefsky et al. 2024, arXiv
2406.11944) rather than cross-layer ones:

1. Replace every MLP by a transcoder: a sparse dictionary that reads the MLP's input and writes its output. What
   the transcoder gets wrong at each (layer, position) is kept as an *error* term, so nothing changes yet.
2. Freeze the attention patterns and the normalisation factors at their values on this prompt. Everything left
   between the token embeddings, the feature activations, the errors and the logits is then *linear*.
3. In a linear model the effect of a source node on a target node is exact and additive: the gradient of the
   target's pre-activation with respect to the source, times the source's value. Those are the graph's edges.
4. Prune: keep the nodes and edges that carry most of the (indirect) influence on the output.

The model is any function with jaxprobe taps for: its embeddings, each MLP's input and output, and every
nonlinearity outside the MLPs that must be frozen (attention patterns, norm scales). `learnjax.llama` has them.

    g = jaxprobe.trace.attribute(lambda t: llama.forward(params, t, cfg)[0], tokens, transcoders)
    g = jaxprobe.trace.prune(g)
    g.error_share, g.top_features(), jaxprobe.trace.to_json(g, tok)
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxprobe import core
from jaxprobe import sae as S

FREEZE = ("layers.pattern", "layers.attn_scale", "layers.mlp_scale", "final_scale")


@dataclasses.dataclass
class Graph:
    """A dense attribution graph. Node order: embeddings (T), features (N), errors (L·T), logits (K).

    adj[t, s] is the direct effect of source s on target t: for a feature target, on its pre-activation; for a
    logit target, on that logit minus the mean logit."""

    tokens: np.ndarray  # (T,)
    adj: np.ndarray  # (n, n) float32
    feat: np.ndarray  # (N, 3) int: layer, position, feature index
    act: np.ndarray  # (N,) activation of each feature
    logit_ids: np.ndarray  # (K,) token ids of the logit nodes
    logit_probs: np.ndarray  # (K,) their probabilities
    n_layers: int
    values: np.ndarray | None = None  # (N + K,) target values: feature pre-activations, then logits − mean
    bias: np.ndarray | None = None  # (N + K,) their constant parts (biases, frozen skip positions)
    keep: np.ndarray | None = None  # bool (n,) after prune
    influence: np.ndarray | None = None  # (n,) total influence on the logits, after prune
    scores: dict = dataclasses.field(default_factory=dict)

    @property
    def T(self):
        return len(self.tokens)

    @property
    def N(self):
        return len(self.feat)

    def kind(self, i: int) -> str:
        T, N, LT = self.T, self.N, self.n_layers * self.T
        return "embed" if i < T else "feature" if i < T + N else "error" if i < T + N + LT else "logit"

    def slices(self):
        T, N, LT, K = self.T, self.N, self.n_layers * self.T, len(self.logit_ids)
        return {"embed": slice(0, T), "feature": slice(T, T + N), "error": slice(T + N, T + N + LT),
                "logit": slice(T + N + LT, T + N + LT + K)}

    def node_info(self, i: int) -> dict:
        """layer (-1 = embedding, n_layers = logit), position, feature index (or token id for logits)."""
        k, s = self.kind(i), self.slices()
        if k == "embed":
            return {"kind": k, "layer": -1, "pos": i, "index": int(self.tokens[i])}
        if k == "feature":
            layer, pos, f = self.feat[i - s["feature"].start]
            return {"kind": k, "layer": int(layer), "pos": int(pos), "index": int(f),
                    "act": float(self.act[i - s["feature"].start])}
        if k == "error":
            j = i - s["error"].start
            return {"kind": k, "layer": j // self.T, "pos": j % self.T, "index": -1}
        j = i - s["logit"].start
        return {"kind": k, "layer": self.n_layers, "pos": self.T - 1, "index": int(self.logit_ids[j]),
                "prob": float(self.logit_probs[j])}

    def top_features(self, n: int = 20):
        """Feature nodes ranked by influence on the logits (call `prune` first)."""
        s = self.slices()["feature"]
        inf = self.influence[s]
        order = np.argsort(-inf)[:n]
        return [dict(self.node_info(s.start + int(j)), influence=float(inf[j])) for j in order]


def _pick(tree, i):
    return jax.tree.map(lambda a: a[i], tree)


def codes(transcoders, x_in, scale_in, kind: str = "topk", k: int = 32):
    """Feature activations (L, T, n) of stacked transcoders on stacked MLP inputs (L, T, d)."""
    return jax.vmap(lambda tc, x, s: S.encode(tc, x * s, kind, k))(transcoders, x_in.astype(jnp.float32), scale_in)


def reconstruct(transcoders, z, scale_out):
    """What the transcoders write, (L, T, d), in the model's units."""
    return jax.vmap(lambda tc, a, s: S.decode(tc, a) / s)(transcoders, z, scale_out)


def attribute(fn: Callable, tokens, transcoders, *, scale_in=None, scale_out=None, kind: str = "topk", k: int = 32,
              embed: str = "embed", mlp_in: str = "layers.mlp_in", mlp_out: str = "layers.mlp_out",
              freeze: Sequence[str] = FREEZE, skip_positions: Sequence[int] = (0,), n_logits: int = 10,
              logit_prob: float = 0.95, batch: int = 256) -> Graph:
    """Build the attribution graph of fn(tokens) -> logits (1, T, V) on its last position.

    transcoders: stacked over layers, {W_enc (L, d, n), b_enc, W_dec (L, n, d), b_dec}; they map MLP input ×
    scale_in to MLP output × scale_out (per-layer scalars, default 1). Features at skip_positions are left out of
    the graph and their MLP outputs held fixed (position 0 is usually an attention sink with huge activations).
    Logit nodes: the most likely next tokens until their probabilities sum to logit_prob (at most n_logits)."""
    tokens = jnp.asarray(tokens)
    assert tokens.ndim == 2 and tokens.shape[0] == 1, "one prompt at a time: tokens (1, T)"
    L = transcoders["W_enc"].shape[0]
    T = tokens.shape[1]
    scale_in = jnp.ones(L) if scale_in is None else jnp.asarray(scale_in, jnp.float32)
    scale_out = jnp.ones(L) if scale_out is None else jnp.asarray(scale_out, jnp.float32)

    logits, acts = jax.jit(lambda t: core.run(fn, t, capture=[embed, mlp_in, mlp_out, *freeze]))(tokens)
    x_in, y = acts[mlp_in][:, 0].astype(jnp.float32), acts[mlp_out][:, 0].astype(jnp.float32)
    z = codes(transcoders, x_in, scale_in, kind, k)
    skip = jnp.zeros(T, bool).at[jnp.asarray(skip_positions, int)].set(True) if len(skip_positions) else \
        jnp.zeros(T, bool)
    z = jnp.where(skip[None, :, None], 0.0, z)
    err = y - reconstruct(transcoders, z, scale_out)  # at skip positions this is the whole MLP output
    e0 = acts[embed][0].astype(jnp.float32)
    frozen = {name: acts[name] for name in freeze}

    feat = np.argwhere(np.asarray(z) > 0)  # (N, 3): layer, position, feature
    a0 = jnp.asarray(np.asarray(z)[tuple(feat.T)])
    fl, fp, fi = (jnp.asarray(c) for c in feat.T)

    probs = jax.nn.softmax(logits[0, -1].astype(jnp.float32))
    order = np.argsort(-np.asarray(probs))
    K = int(min(n_logits, np.searchsorted(np.cumsum(np.asarray(probs)[order]), logit_prob) + 1))
    logit_ids = order[:K]

    dtype = acts[embed].dtype

    def linear(e, a, er):
        """The frozen replacement model: (embeddings, active feature values, errors) -> (feature pre-activations,
        target logits minus mean logit). Affine in its inputs."""
        Z = jnp.zeros(z.shape, jnp.float32).at[fl, fp, fi].set(a)

        def write(x):
            i = core.index()
            out = S.decode(_pick(transcoders, i), Z[i]) / scale_out[i] + er[i]
            return out[None].astype(x.dtype)

        patch = {embed: e[None].astype(dtype), mlp_out: write, **{n: jax.lax.stop_gradient(v) for n, v in frozen.items()}}
        lg, cap = core.run(fn, tokens, capture=[mlp_in], patch=patch)
        xin = cap[mlp_in][:, 0].astype(jnp.float32)  # (L, T, d)
        pre = jax.vmap(lambda tc, x, s: S.pre(tc, x * s))(transcoders, xin, scale_in)  # (L, T, n)
        last = lg[0, -1].astype(jnp.float32)
        return pre[fl, fp, fi], last[logit_ids] - last.mean()

    (pre0, lt0), vjp = jax.vjp(linear, e0, a0, err)
    check = float(jnp.abs(lt0 - (logits[0, -1, logit_ids] - logits[0, -1].mean())).max())
    # what no source node explains: biases, and the MLP outputs at skipped positions (constants, not nodes)
    bias = jax.jit(linear)(jnp.zeros_like(e0), jnp.zeros_like(a0), jnp.where(skip[None, :, None], err, 0.0))

    N = len(feat)
    n = T + N + L * T + K
    adj = np.zeros((n, n), np.float32)

    @jax.jit
    def rows(ct_pre, ct_logit):
        ge, ga, gr = jax.vmap(vjp)((ct_pre, ct_logit))
        return jnp.concatenate([(ge * e0).sum(-1), ga * a0, (gr * err).sum(-1).reshape(ge.shape[0], -1)], 1)

    targets = list(range(N + K))
    for b0 in range(0, len(targets), batch):
        idx = np.array(targets[b0:b0 + batch])
        m = len(idx)
        ct_pre = np.zeros((batch, N), np.float32)
        ct_logit = np.zeros((batch, K), np.float32)
        for r, t in enumerate(idx):
            if t < N:
                ct_pre[r, t] = 1.0
            else:
                ct_logit[r, t - N] = 1.0
        R = np.asarray(rows(jnp.asarray(ct_pre), jnp.asarray(ct_logit)))[:m]  # (m, T + N + L·T)
        tgt = np.where(idx < N, T + idx, T + N + L * T + (idx - N))
        adj[tgt, :T + N + L * T] = R
    # errors at skipped positions are constants, not nodes
    for p in np.asarray(skip_positions, int):
        adj[:, T + N + np.arange(L) * T + p] = 0.0
    g = Graph(tokens=np.asarray(tokens[0]), adj=adj, feat=feat, act=np.asarray(a0), logit_ids=np.asarray(logit_ids),
              logit_probs=np.asarray(probs)[logit_ids], n_layers=L,
              values=np.concatenate([np.asarray(pre0), np.asarray(lt0)]),
              bias=np.concatenate([np.asarray(bias[0]), np.asarray(bias[1])]))
    g.scores["logit_check"] = check  # |replacement − model| on the target logits: ~0 up to float error
    return g


# ---------------------------------------------------------------- influence and pruning
def normalize(adj):
    """|A| with each target's incoming weights summing to 1 (rows with no inputs stay 0)."""
    a = np.abs(adj)
    s = a.sum(1, keepdims=True)
    return np.divide(a, s, out=np.zeros_like(a), where=s > 0)


def influence(adj_norm, w, max_iter: int = 1000, tol: float = 1e-9):
    """Total (direct + indirect) influence of every node on the weighted targets w: x = w Â + w Â² + … =
    w Â (I − Â)⁻¹. Converges in at most depth-of-graph steps, since a causal graph's Â is nilpotent."""
    x = np.zeros_like(w)
    v = w @ adj_norm
    for _ in range(max_iter):
        x = x + v
        if np.abs(v).sum() < tol:
            break
        v = v @ adj_norm
    return x


def _cumulative_keep(scores, frac):
    order = np.argsort(-scores)
    c = np.cumsum(scores[order])
    total = c[-1] if len(c) else 0.0
    n_keep = int(np.searchsorted(c, frac * total) + 1) if total > 0 else 0
    keep = np.zeros(len(scores), bool)
    keep[order[:n_keep]] = True
    return keep


def prune(g: Graph, node_threshold: float = 0.8, edge_threshold: float = 0.98, max_features: int | None = None,
          max_edges: int | None = None) -> Graph:
    """Keep the features that together carry node_threshold of the influence on the logits (embeddings, errors
    and logits always stay), then the edges that carry edge_threshold of the influence among what's left.
    Also computes two scores (Ameisen et al. 2025):
      replacement:  the share of the logits' influence that starts at embeddings rather than error nodes
      completeness: the share of each node's (influence-weighted) inputs that come from non-error nodes.
    max_features / max_edges cap the result (most influential first) when the thresholds keep too much to view."""
    sl = g.slices()
    A = normalize(g.adj)
    w = np.zeros(len(A), np.float32)
    w[sl["logit"]] = g.logit_probs / g.logit_probs.sum()
    inf = influence(A, w)
    keep = np.ones(len(A), bool)
    keep[sl["feature"]] = _cumulative_keep(inf[sl["feature"]], node_threshold)
    n_threshold = int(keep[sl["feature"]].sum())
    if max_features is not None and n_threshold > max_features:
        fk = np.zeros(g.N, bool)
        fk[np.argsort(-inf[sl["feature"]])[:max_features]] = True
        keep[sl["feature"]] = fk
    adj = np.where(keep[:, None] & keep[None, :], g.adj, 0.0)
    # edge pruning on the pruned graph: score = influence of the target × normalized weight
    A2 = normalize(adj)
    inf2 = influence(A2, w) + w
    escore = inf2[:, None] * A2
    flat_keep = _cumulative_keep(escore.ravel(), edge_threshold).reshape(escore.shape)
    if max_edges is not None and flat_keep.sum() > max_edges:
        cut = np.partition(np.where(flat_keep, escore, -1).ravel(), -max_edges)[-max_edges]
        flat_keep &= escore >= cut
    adj = np.where(flat_keep, adj, 0.0).astype(np.float32)
    err, emb = inf[sl["error"]].sum(), inf[sl["embed"]].sum()
    node_w = inf + w
    nonerr = A.copy()
    nonerr[:, sl["error"]] = 0.0
    completeness = float((node_w * nonerr.sum(1)).sum() / max((node_w * A.sum(1)).sum(), 1e-12))
    out = dataclasses.replace(g, adj=adj, keep=keep, influence=inf, scores=dict(g.scores))
    out.scores.update(replacement=float(emb / max(emb + err, 1e-12)), completeness=completeness,
                      error_share=float(err / max(emb + err, 1e-12)), n_features=int(keep[sl["feature"]].sum()),
                      n_features_threshold=n_threshold,
                      n_edges=int((adj != 0).sum()))
    return out


# ---------------------------------------------------------------- interventions on the real model
def ablate_features(transcoders, scale_out, features, values, mlp_out: str = "layers.mlp_out", scale: float = 0.0):
    """A patch for `jaxprobe.run` (batch size 1) that changes the given features' contributions to the real
    model's MLP outputs: each (layer, position, index) in `features` with activation `values` (from the graph)
    is multiplied by `scale` (0 = ablate, 2 = double, −1 = flip). Everything downstream is the real model and
    free to respond, so this tests a graph's claims rather than restating them."""
    by_layer: dict = {}
    for (layer, pos, f), a in zip(features, values, strict=True):
        by_layer.setdefault(int(layer), []).append((int(pos), int(f), float(a)))

    def patch(x):
        i = core.index()
        out = x
        for layer, items in by_layer.items():
            d = jnp.zeros(x.shape[1:], jnp.float32)
            for pos, f, a in items:
                d = d.at[pos].add((scale - 1.0) * a * transcoders["W_dec"][layer, f] / scale_out[layer])
            out = jnp.where(i == layer, out + d[None].astype(x.dtype), out)
        return out

    return {mlp_out: patch}


# ---------------------------------------------------------------- export
def cantor(x: int, y: int) -> int:
    """Cantor pairing: how circuit-tracer packs (layer, feature) into one integer id."""
    return (x + y) * (x + y + 1) // 2 + y


def _cumulative_influence(g: Graph):
    """circuit-tracer's node `influence`: the cumulative share of total feature influence up to and including this
    node, in decreasing order of influence (so lower = more important; the viewer keeps nodes ≤ its threshold)."""
    out = np.full(len(g.adj), np.nan)
    if g.influence is None:
        return out
    sl = g.slices()["feature"]
    inf = g.influence[sl]
    order = np.argsort(-inf)
    c = np.cumsum(inf[order]) / max(inf.sum(), 1e-12)
    out[sl.start + order] = c
    return out


def to_json(g: Graph, decode: Callable[[int], str], *, slug: str = "graph", scan: str = "jaxprobe",
            labels: dict | None = None, node_threshold: float | None = None) -> dict:
    """The graph in circuit-tracer's JSON format (circuit_tracer/frontend/graph_models.py, schema_version 1), which
    its local viewer and Neuronpedia's graph viewer load. Pruned nodes and edges only; nodes left with no edges
    are dropped except embeddings and logits, as circuit-tracer does. decode(token_id) -> str.
    labels: optional {(layer, feature): "short description"} shown as each feature's "clerp".
    scan: any string for local viewing; "./feats/<name>" makes the viewer fetch feature pages from
    <frontend>/feats/<name>/<cantor(layer, feature)>.json (see `feature_page`)."""
    labels = labels or {}
    keep = g.keep.copy() if g.keep is not None else np.ones(len(g.adj), bool)
    linked = (g.adj != 0).any(0) | (g.adj != 0).any(1)
    sl = g.slices()
    keep[sl["feature"]] &= linked[sl["feature"]]
    keep[sl["error"]] &= linked[sl["error"]]
    cum = _cumulative_influence(g)
    L = g.n_layers

    def ids(i):
        info = g.node_info(i)
        k, pos = info["kind"], info["pos"]
        if k == "embed":
            return f"E_{info['index']}_{pos}", f"E_{info['index']}-{pos}"
        if k == "feature":
            return f"{info['layer']}_{info['index']}_{pos}", f"{info['layer']}_{info['index']}-0"
        if k == "error":
            return f"{info['layer']}_-1_{pos}", f"{info['layer']}_-1-{pos}"
        return f"{L + 1}_{info['index']}_{pos}", f"L_{info['index']}-{pos}"

    nodes = []
    first_logit = sl["logit"].start
    for i in np.flatnonzero(keep):
        i = int(i)
        info = g.node_info(i)
        k = info["kind"]
        nid, js = ids(i)
        node = {"node_id": nid, "layer": "E" if k == "embed" else str(L + 1) if k == "logit" else str(info["layer"]),
                "ctx_idx": info["pos"], "token_prob": 0.0, "is_target_logit": False, "run_idx": 0,
                "reverse_ctx_idx": 0, "jsNodeId": js, "clerp": "", "influence": None, "activation": None}
        if k == "embed":
            node.update(feature=info["pos"], feature_type="embedding")
        elif k == "feature":
            node.update(feature=cantor(info["layer"], info["index"]), feature_type="cross layer transcoder",
                        activation=info["act"], influence=None if np.isnan(cum[i]) else float(cum[i]),
                        clerp=labels.get((info["layer"], info["index"]), ""))
        elif k == "error":
            node.update(feature=-1, feature_type="mlp reconstruction error")
        else:
            tok = decode(info["index"])
            node.update(feature=info["index"], feature_type="logit", token_prob=info["prob"],
                        is_target_logit=i == first_logit, clerp=f'Output "{tok}" (p={info["prob"]:.3f})')
        nodes.append(node)
    links = [{"source": ids(int(s))[0], "target": ids(int(t))[0], "weight": float(g.adj[t, s])}
             for t, s in zip(*np.nonzero(g.adj), strict=True) if keep[t] and keep[s]]
    meta = {"slug": slug, "scan": scan, "transcoder_list": [], "prompt_tokens": [decode(int(t)) for t in g.tokens],
            "prompt": "".join(decode(int(t)) for t in g.tokens), "node_threshold": node_threshold,
            "schema_version": 1}
    return {"metadata": meta, "qParams": {"pinnedIds": [], "supernodes": [], "linkType": "both", "clickedId": "",
                                          "sg_pos": ""}, "nodes": nodes, "links": links}


def write_graph(data_dir, graph_json: dict):
    """Write <data_dir>/<slug>.json and add its metadata to <data_dir>/graph-metadata.json (the layout circuit-tracer's
    local server reads: `circuit_tracer.frontend.local_server.serve(data_dir)`)."""
    import json
    from pathlib import Path

    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    meta = graph_json["metadata"]
    (d / f"{meta['slug']}.json").write_text(json.dumps(graph_json))
    index = d / "graph-metadata.json"
    graphs = json.loads(index.read_text())["graphs"] if index.exists() else []
    graphs = [m for m in graphs if m["slug"] != meta["slug"]] + [meta]
    index.write_text(json.dumps({"graphs": graphs}, indent=1))


def feature_page(layer: int, index: int, contexts, *, top_logits=(), bottom_logits=(), transcoder_id: str = "") -> dict:
    """One feature's page in circuit-tracer's feature JSON format (frontend/feature_models.py).
    contexts: list of (tokens: list[str], acts: list[float]) — e.g. the feature's top-activating text windows."""
    acts = [a for _, aa in contexts for a in aa]
    return {"transcoder_id": transcoder_id, "index": cantor(layer, index), "top_logits": list(top_logits),
            "bottom_logits": list(bottom_logits), "act_min": float(min(acts, default=0.0)),
            "act_max": float(max(acts, default=1.0)),
            "examples_quantiles": [{"quantile_name": "Top activations", "examples": [
                {"tokens": list(t), "tokens_acts_list": [float(a) for a in aa], "train_token_ind": 0,
                 "is_repeated_datapoint": False} for t, aa in contexts]}]}
