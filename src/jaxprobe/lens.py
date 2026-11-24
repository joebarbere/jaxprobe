"""Decode intermediate states as predictions: the logit lens, the tuned lens and direct logit attribution.

The **logit lens** (nostalgebraist 2020) applies the model's own final norm and unembedding to the residual stream
after an earlier layer and reads the result as "what the model would predict if it stopped here". The **tuned
lens** (Belrose et al. 2023, arXiv 2303.08112) first applies a learned affine map per layer, h ↦ h + hA + b,
trained so each layer's prediction matches the final one; it is far more faithful in early layers.
**Direct logit attribution** splits a final logit into what each component wrote to the residual stream, with the
final norm's scale frozen at its real value, so the parts add up exactly.

    U = lens.rms_unembed(params["final_norm"], params["embed"].T, eps=1e-5)       # or any callable h -> logits
    _, acts = jaxprobe.run(model, params, tokens, capture=["embed", "layers.resid"])
    R = jnp.concatenate([acts["embed"][None], acts["layers.resid"]])              # (L+1, B, T, d)
    st = lens.stats(U, R, token=next_tokens, sets={"yes": yes_ids, "no": no_ids})
    st["kl"]          # (L+1, B, T) KL(final ‖ lens), nats
    st["sets"]["yes"] - st["sets"]["no"]                                          # log-odds of yes vs no, per layer
    lens.to_html("lens.html", st, tokens[0], decode)                              # layer × position grid

    tuned, hist = lens.fit_tuned(U, collect, batches, steps=1000)                 # collect(batch) -> (R, final)
    st = lens.stats(U, R[:-1], final, tuned=tuned)

    parts = {"embed": acts["embed"], "attn": acts["layers.attn_out"], "mlp": acts["layers.mlp_out"]}
    c = lens.dla(U, parts, ref=h_final, token=tok, minus=other)                   # each (…, B, T); sum = logit diff

`R` is any stack of residual-stream states with the layer axis first. The lens works on any tapped site whose
value lives in the residual stream (`layers.mid`, `layers.resid`); for what a sublayer *adds* (`attn_out`,
`mlp_out`), use `dla`, which is the linear part of the same readout.
"""
from __future__ import annotations

import dataclasses
import html
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["Unembed", "dla", "fit_tuned", "init_tuned", "layer_unembed", "logits", "rms_unembed", "stats",
           "to_html", "translate", "tuned_loss"]


# ---------------------------------------------------------------- the readout
@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Unembed:
    """Final norm + unembedding: logits = norm(h) @ W_U (+ head_b). kind is "rms" (x/rms·w), "layer"
    ((x−mean)/std·w + b) or "none". A pytree, so it can be passed into jitted functions."""
    W_U: jax.Array                                     # (d, V)
    norm_w: jax.Array | None = None                    # (d,)
    norm_b: jax.Array | None = None                    # (d,), layer norm only
    head_b: jax.Array | None = None                    # (V,)
    eps: float = dataclasses.field(default=1e-5, metadata={"static": True})
    kind: str = dataclasses.field(default="rms", metadata={"static": True})

    def inv_scale(self, h):
        """1/rms(h) or 1/std(h), float32, shape (…, 1): the factor `dla` freezes."""
        h = jnp.asarray(h, jnp.float32)
        if self.kind == "none":
            return jnp.ones(h.shape[:-1] + (1,), jnp.float32)
        if self.kind == "layer":
            h = h - h.mean(-1, keepdims=True)
        return jax.lax.rsqrt(jnp.mean(h * h, -1, keepdims=True) + self.eps)

    def _center(self, h):
        return h - h.mean(-1, keepdims=True) if self.kind == "layer" else h

    def normed(self, h):
        x = self._center(jnp.asarray(h, jnp.float32)) * self.inv_scale(h)
        if self.norm_w is not None:
            x = x * jnp.asarray(self.norm_w, jnp.float32)
        if self.norm_b is not None:
            x = x + jnp.asarray(self.norm_b, jnp.float32)
        return x

    def __call__(self, h):
        """float32 logits. The matmul runs in W_U's dtype, like the model's own head, and is upcast after: with
        bf16 weights this gives the model's own logits, and its backward pass (training a tuned lens) stays a bf16
        matmul too, 1.75× faster on an RX 7600 XT than asking for float32 output."""
        W = jnp.asarray(self.W_U)
        out = (self.normed(h).astype(W.dtype) @ W).astype(jnp.float32)
        return out if self.head_b is None else out + jnp.asarray(self.head_b, jnp.float32)

    def readout(self, token, minus=None):
        """The vector u with logit[token] (− logit[minus]) = (centered h · u) / scale + const, and that const.
        token/minus: ints or int arrays (…); returns u (…, d) and const (…)."""
        def col(t):
            u = jnp.asarray(self.W_U, jnp.float32)[:, jnp.asarray(t)]
            u = jnp.moveaxis(u, 0, -1)                                         # (…, d)
            c = jnp.zeros(u.shape[:-1], jnp.float32)
            if self.norm_b is not None:
                c = c + u @ jnp.asarray(self.norm_b, jnp.float32)
            if self.head_b is not None:
                c = c + jnp.asarray(self.head_b, jnp.float32)[jnp.asarray(t)]
            if self.norm_w is not None:
                u = u * jnp.asarray(self.norm_w, jnp.float32)
            return u, c

        u, c = col(token)
        if minus is not None:
            u2, c2 = col(minus)
            u, c = u - u2, c - c2
        return u, c


def rms_unembed(norm_w, W_U, eps: float = 1e-5, head_b=None, offset: float = 0.0) -> Unembed:
    """RMSNorm models (Llama, Qwen, SmolLM, Mistral). W_U is (d, V): for tied embeddings pass embed.T.
    Gemma's norm multiplies by (1 + w): pass offset=1."""
    w = None if norm_w is None else jnp.asarray(norm_w, jnp.float32) + offset
    return Unembed(jnp.asarray(W_U), w, None, None if head_b is None else jnp.asarray(head_b), eps, "rms")


def layer_unembed(norm_w, norm_b, W_U, eps: float = 1e-5, head_b=None) -> Unembed:
    """LayerNorm models (GPT-2, Pythia)."""
    return Unembed(jnp.asarray(W_U), jnp.asarray(norm_w), jnp.asarray(norm_b),
                   None if head_b is None else jnp.asarray(head_b), eps, "layer")


# ---------------------------------------------------------------- tuned lens
def init_tuned(n: int, d: int) -> dict:
    """n affine translators h ↦ h + hA + b, starting at the identity (A = b = 0), float32."""
    return {"A": jnp.zeros((n, d, d), jnp.float32), "b": jnp.zeros((n, d), jnp.float32)}


def translate(tuned, i, h):
    """Apply translator i to h (…, d); the result keeps h's dtype."""
    hf = jnp.asarray(h, jnp.float32)
    return (hf + hf @ tuned["A"][i] + tuned["b"][i]).astype(h.dtype)


def _rows(R, tuned):
    """Per-layer inputs for lax.map: the states, and the translator for each if there is one."""
    if tuned is None:
        return (R,)
    if tuned["A"].shape[0] != R.shape[0]:
        raise ValueError(f"tuned lens has {tuned['A'].shape[0]} translators for {R.shape[0]} layers")
    return R, tuned["A"], tuned["b"]


def _lens(unembed, row):
    if len(row) == 1:
        return unembed(row[0])
    h, A, b = row
    hf = h.astype(jnp.float32)
    return unembed((hf + hf @ A + b).astype(h.dtype))


def logits(unembed: Callable, R, tuned=None):
    """Lens logits for every layer: (n, …, V) float32. Materializes all of them: for long inputs use `stats`."""
    return jax.lax.map(lambda row: _lens(unembed, row), _rows(R, tuned))


def _kl(lp, lq):
    return (jnp.exp(lp) * (lp - lq)).sum(-1)


def tuned_loss(tuned, unembed: Callable, R, final):
    """Σ over layers of mean KL(final ‖ tuned lens): the training loss of Belrose et al. One layer's logits are
    alive at a time (each is rematerialized in the backward pass)."""
    lp = jax.nn.log_softmax(jnp.asarray(final, jnp.float32))

    @jax.checkpoint
    def one(row):
        return _kl(lp, jax.nn.log_softmax(_lens(unembed, row))).mean()

    return jax.lax.map(one, _rows(R, tuned)).sum()


def _schedule(i, steps, lr, warmup, final_lr):
    """Linear warmup from 0, then cosine decay to final_lr (optax.warmup_cosine_decay_schedule)."""
    i = jnp.asarray(i, jnp.float32)
    warm = lr * i / max(warmup, 1)
    t = jnp.clip((i - warmup) / max(steps - warmup, 1), 0.0, 1.0)
    cos = final_lr + (lr - final_lr) * 0.5 * (1 + jnp.cos(jnp.pi * t))
    return jnp.where(i < warmup, warm, cos)


def fit_tuned(unembed: Callable, collect: Callable, batches: Iterable, *, steps: int, tuned=None, lr: float = 1e-3,
              warmup: int = 50, final_lr: float = 1e-4, log_every: int = 100, log: Callable | None = print):
    """Train a tuned lens with Adam. collect(batch) -> (R (n, …, d), final logits (…, V)) runs inside the jitted
    step, so the forward pass and the lens update compile together; batches yields at least `steps` batches.
    Returns (tuned, history [[step, mean KL over layers], …])."""
    batches = iter(batches)
    first = next(batches)
    if tuned is None:
        Rs, _ = jax.eval_shape(collect, first)
        tuned = init_tuned(Rs.shape[0], Rs.shape[-1])
    n = tuned["A"].shape[0]
    m = jax.tree.map(jnp.zeros_like, tuned)
    v = jax.tree.map(jnp.zeros_like, tuned)

    @jax.jit
    def step(tuned, m, v, i, batch):
        R, final = collect(batch)
        loss, g = jax.value_and_grad(tuned_loss)(tuned, unembed, jax.lax.stop_gradient(R),
                                                 jax.lax.stop_gradient(final))
        m = jax.tree.map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree.map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
        t = i + 1.0
        a = _schedule(i, steps, lr, warmup, final_lr)
        tuned = jax.tree.map(lambda p, mm, vv: p - a * (mm / (1 - 0.9 ** t)) / (jnp.sqrt(vv / (1 - 0.999 ** t)) + 1e-8),
                             tuned, m, v)
        return tuned, m, v, loss

    hist, batch = [], first
    for i in range(steps):
        if i:
            batch = next(batches)
        tuned, m, v, loss = step(tuned, m, v, jnp.float32(i), batch)
        if i % log_every == 0 or i == steps - 1:
            hist.append([i, float(loss) / n])
            if log is not None:
                log(f"tuned lens step {i}: mean KL {hist[-1][1]:.3f}")
    return tuned, hist


# ---------------------------------------------------------------- per-layer statistics
def stats(unembed: Callable, R, final=None, *, tuned=None, token=None, sets: Mapping[str, Sequence[int]] | None = None,
          top: int = 5) -> dict:
    """Per-layer readouts of R (n, …, d), computed one layer at a time (no (n, …, V) array). Returns, each (n, …):

    kl        KL(final ‖ lens) in nats (final defaults to the lens of R[-1])
    agree     lens top-1 == final top-1
    entropy   of the lens distribution, nats
    top_ids, top_logprobs   (n, …, top)
    logprob, rank           of `token` (…) ints, e.g. the next token; rank 0 = the lens's top choice
    sets      {name: log P(any token in the set)}: e.g. {"yes": [" yes", " Yes"], "no": [...]} as ids
    """
    lp_final = jax.nn.log_softmax(jnp.asarray(unembed(R[-1]) if final is None else final, jnp.float32))
    top_final = lp_final.argmax(-1)
    set_ids = {k: jnp.asarray(list(v), jnp.int32) for k, v in (sets or {}).items()}
    tok = None if token is None else jnp.asarray(token, jnp.int32)

    def one(row):
        lg = _lens(unembed, row)
        lp = jax.nn.log_softmax(lg)
        tv, ti = jax.lax.top_k(lp, top)
        out = {"kl": _kl(lp_final, lp), "agree": ti[..., 0] == top_final, "entropy": -(jnp.exp(lp) * lp).sum(-1),
               "top_ids": ti, "top_logprobs": tv}
        if tok is not None:
            t_lg = jnp.take_along_axis(lg, tok[..., None], -1)
            out["logprob"] = jnp.take_along_axis(lp, tok[..., None], -1)[..., 0]
            out["rank"] = (lg > t_lg).sum(-1)
        if set_ids:
            out["sets"] = {k: jax.nn.logsumexp(lp[..., ids], -1) for k, ids in set_ids.items()}
        return out

    return jax.lax.map(one, _rows(R, tuned))


# ---------------------------------------------------------------- direct logit attribution
def dla(unembed: Unembed, parts, ref, token, minus=None):
    """Each part's direct contribution to logit[token] (− logit[minus]), with the final norm's scale frozen at
    ref, the residual state the final norm actually saw (… , d). parts: an array (…, d) or a pytree of them
    (e.g. {"embed": (B, T, d), "attn": (L, B, T, d), "mlp": (L, B, T, d)}), broadcasting against ref.

    If the parts sum to ref, their contributions plus `dla(...)["const"]`-style constants sum to the logit
    (difference) exactly; the constant (norm and head biases) is returned under "const" when parts is a dict."""
    u, c = unembed.readout(token, minus)                                       # (…, d), (…)
    inv = unembed.inv_scale(ref)[..., 0]                                       # (…)

    def contrib(p):
        p = unembed._center(jnp.asarray(p, jnp.float32))
        return (p * u).sum(-1) * inv

    out = jax.tree.map(contrib, parts)
    if isinstance(out, dict):
        out = dict(out, const=jnp.broadcast_to(c, inv.shape))
    return out


# ---------------------------------------------------------------- the page
def to_html(path, st: Mapping[str, Any], tokens: Sequence[int], decode: Callable[[int], str], *, row=(),
            layer_names: Sequence[str] | None = None, title: str = "Logit lens", value: str = "prob") -> Path:
    """A layer × position grid: each cell shows the lens's top token there, shaded by its probability (value="prob")
    or by KL to the final layer (value="kl"); hover lists the top tokens. Spaces show as ␣ and tokens that decode
    to nothing (special tokens, with most decoders) as ∅. st comes from `stats`; row indexes the
    batch axes between the layer and position axes (e.g. row=0 for a (n, B, T) stats dict); tokens are that
    sequence's input ids, used as column headings."""
    pick = lambda a: np.asarray(a[(slice(None), *np.atleast_1d(row))] if row != () else a)
    ids, lps, kl = pick(st["top_ids"]), pick(st["top_logprobs"]), pick(st["kl"])
    n, T = ids.shape[:2]
    names = list(layer_names) if layer_names is not None else [str(i) for i in range(n)]
    vocab: dict[int, int] = {}
    def vid(t):
        return vocab.setdefault(int(t), len(vocab))
    cells = [[[[vid(t), round(math.exp(float(lp)), 4)] for t, lp in zip(ids[i, j], lps[i, j], strict=True)]
              for j in range(T)] for i in range(n)]
    data = {"names": names, "cols": [decode(int(t)) for t in tokens], "cells": cells, "kl": np.round(kl.astype(np.float64), 3).tolist(),
            "vocab": [decode(t) for t in vocab], "value": value}
    if "logprob" in st:
        data["target"] = np.round(np.exp(pick(st["logprob"]).astype(np.float64)), 4).tolist()
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = _PAGE.replace("__TITLE__", html.escape(title)).replace("__DATA__", blob)
    path = Path(path)
    path.write_text(page, encoding="utf-8")
    return path


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{--bg:#fbfbf9;--fg:#1d1d1b;--mute:#6b6b66;--line:#e2e1dc;--hi:37,99,235}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe6;--mute:#9b9a94;--line:#2c2c2a;--hi:96,145,255}}
body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.4 ui-sans-serif,system-ui,sans-serif}
header{padding:16px}h1{font-size:17px;margin:0 0 4px}p{margin:0;color:var(--mute)}
.wrap{overflow:auto;padding:0 16px 16px}table{border-collapse:collapse;font:12px ui-monospace,monospace}
td,th{border:1px solid var(--line);padding:3px 5px;white-space:pre;max-width:9em;overflow:hidden;text-overflow:ellipsis}
th{position:sticky;top:0;background:var(--bg);font-weight:600}th.l{left:0;position:sticky;background:var(--bg);color:var(--mute)}
td.t{outline:2px solid rgb(var(--hi))}#tip{position:fixed;pointer-events:none;background:var(--bg);border:1px solid var(--line);
padding:6px 8px;display:none;font:12px ui-monospace,monospace;white-space:pre}
</style></head><body><header><h1>__TITLE__</h1><p id="sub"></p></header><div class="wrap"><table id="g"></table></div>
<div id="tip"></div><script>
const D=__DATA__;
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const show=s=>{const t=JSON.stringify(s).slice(1,-1);return t===''?'\u2205':t.trim()===''?t.replace(/ /g,'\u2423'):t};
const g=document.getElementById('g'),tip=document.getElementById('tip');
document.getElementById('sub').textContent=(D.value==='kl'?'Shade: KL to the final layer (darker = closer). ':'Shade: top-1 probability. ')
 +'Rows: layers, last at the top. Hover a cell for its top tokens.'+(D.target?' Outlined: the lens\\'s top token is the target.':'');
const maxkl=Math.max(...D.kl.flat(),1e-6);
let h='<tr><th class="l">layer</th>'+D.cols.map(c=>'<th>'+esc(show(c))+'</th>').join('')+'</tr>';
for(let i=D.names.length-1;i>=0;i--){h+='<tr><th class="l">'+esc(D.names[i])+'</th>';
 D.cells[i].forEach((c,j)=>{const a=D.value==='kl'?1-D.kl[i][j]/maxkl:c[0][1];
  const t=D.target&&D.target[i][j]>=c[0][1]-1e-6;
  h+='<td data-i="'+i+'" data-j="'+j+'" class="'+(t?'t':'')+'" style="background:rgba(var(--hi),'+(0.08+0.6*a).toFixed(3)+')">'
   +esc(show(D.vocab[c[0][0]]))+'</td>'});h+='</tr>'}
g.innerHTML=h;
g.addEventListener('mousemove',e=>{const td=e.target.closest('td');if(!td){tip.style.display='none';return}
 const i=+td.dataset.i,j=+td.dataset.j;let s='layer '+D.names[i]+' · '+show(D.cols[j])+' · KL '+D.kl[i][j]+'\\n';
 D.cells[i][j].forEach(([v,p])=>s+=(p*100).toFixed(1).padStart(5)+'%  '+show(D.vocab[v])+'\\n');
 if(D.target)s+='target '+(D.target[i][j]*100).toFixed(1)+'%';
 tip.textContent=s;tip.style.display='block';
 tip.style.left=Math.min(e.clientX+12,innerWidth-tip.offsetWidth-8)+'px';tip.style.top=(e.clientY+12)+'px'});
g.addEventListener('mouseleave',()=>tip.style.display='none');
</script></body></html>"""
