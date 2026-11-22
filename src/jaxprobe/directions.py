"""Find a behaviour direction from contrasting prompts and remove it from the weights: the difference-in-means
recipe of Arditi et al. 2024, "Refusal in language models is mediated by a single direction" (arXiv 2406.11717).

    pos = directions.collect(fwd, params, harmful_batches, at="layers.resid_pre", positions=[-3, -2, -1])
    neg = directions.collect(fwd, params, harmless_batches, at="layers.resid_pre", positions=[-3, -2, -1])
    cands = directions.candidates(pos, neg)                       # (layers, positions, d)
    score = directions.arditi_score(last_logits, harmful_toks, harmless_toks, refusal_ids,
                                    ablate_at=["embed", "layers.attn_out", "layers.mlp_out"], add_at="layers.resid_pre")
    s = directions.grid(score, cands)                            # bypass / induce / kl for every candidate
    best = directions.select(s, n_layers=L)                      # Arditi's rule
    u = cands[best["layer"], best["pos"]]
    edited = directions.abliterate(params, u, writers={"embed": -1, "layers.wo": -1, "layers.w_down": -1})
    directions.check_edit(fwd, params, edited, u, tokens, ablate_at=[...])      # weight edit == runtime ablation

For checkpoints too big to load, `hf_edit` gives the same edit for `jaxprobe.io.edit_checkpoint`, one tensor at
a time, on Hugging Face (PyTorch-layout) names.
"""
from __future__ import annotations

import dataclasses
import fnmatch
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import core, reduce, steer


@dataclasses.dataclass
class Stats:
    """Running sum and count of activations; mean = sum / n."""
    sum: jax.Array
    n: float

    @property
    def mean(self):
        return self.sum / max(self.n, 1e-12)

    def __add__(self, other: Stats) -> Stats:
        return Stats(self.sum + other.sum, self.n + other.n)


def collect(fn: Callable, params, batches: Iterable[Mapping[str, Any]], *, at: str, positions="last",
            args: Callable[[Mapping], tuple] = lambda b: (b["tokens"],), spill=None, dtype=np.float16):
    """Mean activation of tap `at` over many prompts, without keeping them: fn(params, *args(batch)) is run under
    jaxprobe with a reducer at the tap, inside the scan body, so each step returns only a (layers, P, d) sum.

    batches: dicts as from `jaxprobe.batch.padded` (tokens, mask, weights, index); rows with weight 0 are ignored.
    positions: "last" (-1), an int or list of ints (negative = from the end; with left padding these are aligned
               across rows), or "mean" (mean over the batch's mask).
    spill: a `jaxprobe.io.Spill` (N, layers, [P,] d) to also keep every row's reduced activations on disk, in
           input order. Batches already written are skipped, so a rerun resumes; the returned Stats then cover only
           the rows computed in this call (take the mean of spill.array for all of them).
    Returns Stats (sum has shape (layers, P, d) for positions lists, (layers, d) for "last"/"mean")."""
    if positions == "last":
        sel, tail = reduce.last, 1
    elif positions == "mean":
        sel, tail = None, 1
    else:
        sel, tail = reduce.positions(positions), 2

    def per_row(mask):
        return reduce.masked_mean(mask) if sel is None else sel

    @jax.jit
    def summed(params, weights, mask, *a):
        red = reduce.compose(per_row(mask), lambda x: reduce.batch_sum(weights, axis=x.ndim - tail - 1)(x))
        _, caps = core.run(fn, params, *a, capture={at: red})
        return _one(caps, at)

    @jax.jit
    def rows(params, weights, mask, *a):
        _, caps = core.run(fn, params, *a, capture={at: per_row(mask)})
        x = _one(caps, at)
        bax = x.ndim - tail - 1
        s = reduce.batch_sum(weights, axis=bax)(x)
        return s, jnp.moveaxis(x, bax, 0)

    total, n = None, 0.0
    for b in batches:
        w = jnp.asarray(b.get("weights", np.ones(len(b["tokens"]), np.float32)))
        m = jnp.asarray(b.get("mask", np.ones(np.shape(b["tokens"]), bool)))
        if spill is not None:
            if spill.done(b["index"]):
                continue
            s, r = rows(params, w, m, *args(b))
            spill.write(b["index"], np.asarray(r).astype(dtype))
        else:
            s = summed(params, w, m, *args(b))
        total = s if total is None else total + s
        n += float(np.asarray(b.get("weights", np.ones(len(b["tokens"])))).sum())
    if spill is not None:
        spill.flush()
    if total is None:
        raise ValueError("no batches (or every batch was already in the spill: read spill.array instead)")
    return Stats(total, n)


def _one(caps: Mapping, pat: str):
    if len(caps) != 1:
        raise core.TapError(f"pattern {pat!r} matched {sorted(caps)}; give the name of exactly one tap")
    return next(iter(caps.values()))


def candidates(pos: Stats, neg: Stats):
    """Difference in means, one candidate direction per layer (and position): mean(pos) − mean(neg)."""
    return pos.mean - neg.mean


def refusal_score(logits, token_ids, eps: float = 1e-8):
    """Arditi et al.'s refusal metric from next-token logits (..., V): log-odds that the next token is one of
    token_ids (the first tokens of typical refusals, e.g. "I", "As", "Sorry"). > 0 means a refusal is more likely
    than not. One forward pass, no generation."""
    p = jax.nn.softmax(logits.astype(jnp.float32), -1)
    pr = p[..., jnp.asarray(token_ids)].sum(-1)
    return jnp.log(pr + eps) - jnp.log1p(-pr + eps)


def arditi_score(last_logits: Callable, harmful, harmless, refusal_ids, *, ablate_at: Sequence[str],
                 add_at: str) -> Callable:
    """score(layer, u) -> {"bypass", "induce", "kl"} for `grid`, from a jittable last_logits(patch, tokens) ->
    (B, V) (build it with `jaxprobe.bind`):
      bypass: mean refusal score on harmful prompts with u ablated at every ablate_at tap (lower = refusal gone)
      induce: mean refusal score on harmless prompts with u added at layer `layer` of add_at (higher = induces it)
      kl:     KL(base ‖ ablated) of the next-token distribution on harmless prompts (damage)."""
    base = last_logits({}, harmless)

    def score(layer, u):
        abl = {p: steer.ablate(u) for p in ablate_at}
        bypass = refusal_score(last_logits(abl, harmful), refusal_ids).mean()
        induce = refusal_score(last_logits({add_at: core.at(layer, steer.add(u))}, harmless), refusal_ids).mean()
        kl = steer.kl(base, last_logits(abl, harmless)).mean()
        return {"bypass": bypass, "induce": induce, "kl": kl}
    return score


def grid(score: Callable, cands, batch_size: int = 1) -> dict[str, np.ndarray]:
    """Evaluate score(layer, u) for every candidate in cands (layers, P, d) or (layers, d), compiled once
    (lax.map, batch_size candidates at a time through vmap). Returns {key: (layers, P)} numpy arrays."""
    c = cands if cands.ndim == 3 else cands[:, None]
    L, P, d = c.shape
    layer = jnp.repeat(jnp.arange(L, dtype=jnp.int32), P)
    out = jax.jit(lambda lay, u: jax.lax.map(lambda a: score(*a), (lay, u), batch_size=batch_size))(
        layer, c.reshape(L * P, d))
    return {k: np.asarray(v).reshape(L, P) for k, v in out.items()}


def select(scores: Mapping[str, np.ndarray], *, n_layers: int | None = None, max_kl: float = 0.1,
           max_layer_frac: float = 0.8, min_induce: float = 0.0) -> dict:
    """Arditi et al.'s choice: among candidates that induce refusal (induce > min_induce), barely change harmless
    outputs (kl < max_kl) and sit in the first max_layer_frac of layers, the one with the lowest bypass score."""
    b, ind, kl = (np.asarray(scores[k]) for k in ("bypass", "induce", "kl"))
    L, P = b.shape
    n_layers = n_layers or L
    ok = (ind > min_induce) & (kl < max_kl) & (np.arange(L)[:, None] < max_layer_frac * n_layers)
    if not ok.any():
        raise ValueError("no candidate passes the filters; loosen max_kl / min_induce or check the prompts")
    flat = np.where(ok, b, np.inf).argmin()
    layer, pos = divmod(int(flat), P)
    return {"layer": layer, "pos": pos, "bypass": float(b[layer, pos]), "induce": float(ind[layer, pos]),
            "kl": float(kl[layer, pos]), "n_valid": int(ok.sum())}


def _path(kp) -> str:
    parts = []
    for k in kp:
        parts.append(str(getattr(k, "key", getattr(k, "idx", getattr(k, "name", k)))))
    return ".".join(parts)


def abliterate(params, u, writers: Mapping[str, int] | Sequence[str], alpha: float = 1.0):
    """Project u out of every parameter that writes to the residual stream: W ← W − α (W û) ûᵀ along the axis
    that lands in the stream. writers: {glob on dotted param paths: that axis} (or a list of globs, axis -1).
    Afterwards the model behaves as if u were ablated after every write — check with `check_edit`.

    Tied embeddings: if the unembedding reuses the embedding matrix, editing it also edits the read-out, and the
    final norm's gain can bring u back in front of it. Untie first (keep the original as the unembedding)."""
    w = dict.fromkeys(writers, -1) if not isinstance(writers, Mapping) else dict(writers)
    hit = set()

    def f(kp, x):
        name = _path(kp)
        for pat, axis in w.items():
            if fnmatch.fnmatchcase(name, pat):
                hit.add(pat)
                return steer.orthogonalize(x, u, alpha, axis=axis)
        return x
    out = jax.tree_util.tree_map_with_path(f, params)
    missing = set(w) - hit
    if missing:
        raise KeyError(f"writer pattern(s) {sorted(missing)} matched no parameter")
    return out


def check_edit(fn: Callable, params, edited, u, *args, ablate_at: Sequence[str], **kwargs) -> float:
    """max |fn(edited) − fn(params) with u ablated at every ablate_at tap|. ~0 (to dtype precision) means the
    weight edit is exactly the runtime ablation."""
    runtime, _ = core.run(fn, params, *args, patch={p: steer.ablate(u) for p in ablate_at}, **kwargs)
    return float(jnp.abs(fn(edited, *args, **kwargs).astype(jnp.float32) - runtime.astype(jnp.float32)).max())


# Hugging Face names of the tensors that write to the residual stream, and the axis that lands there
# (PyTorch Linear weights are (out, in): axis 0). The globs also cover MoE experts and shared experts.
HF_WRITERS = {"*embed_tokens.weight": -1, "*o_proj.weight": 0, "*down_proj.weight": 0, "*o_proj.bias": 0,
              "*down_proj.bias": 0}


def hf_edit(u, alpha: float = 1.0, writers: Mapping[str, int] = HF_WRITERS, per_layer=None):
    """(edit, match) for `jaxprobe.io.edit_checkpoint`: abliterate a Hugging Face checkpoint tensor by tensor on the
    CPU in float32 (one tensor in memory at a time). per_layer: optional (L, d) directions, picked by the
    "layers.<i>." in each name (the embedding uses u). Check config.json for tie_word_embeddings first: a tied
    lm_head would be edited too (see `abliterate`)."""
    u0 = np.asarray(u, np.float32)
    u0 = u0 / np.linalg.norm(u0)
    per = None if per_layer is None else np.asarray(per_layer, np.float32)

    def edit(name, a):
        axis = next(ax for pat, ax in writers.items() if fnmatch.fnmatchcase(name, pat))
        v = u0
        m = re.search(r"layers\.(\d+)\.", name)
        if per is not None and m:
            v = per[int(m[1])] / np.linalg.norm(per[int(m[1])])
        x = np.moveaxis(np.asarray(a, np.float32), axis, -1)
        x = x - alpha * (x @ v)[..., None] * v
        return np.moveaxis(x, -1, axis).astype(a.dtype)
    return edit, list(writers)
