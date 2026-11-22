# jaxprobe

Interpretability tools for JAX models: capture and patch activations, train sparse autoencoders, steer,
browse features, and trace circuits. PyTorch has TransformerLens, nnsight, SAELens and circuit-tracer;
JAX has had no maintained equivalent since Penzai was archived. jaxprobe aims to fill that gap.

**Status: pre-alpha.** `core`, `sae`, `steer` and `trace` work and are tested; `dash` is planned. The API
may still change.

## Quick start

Mark points in your model with `tap`, and use `jaxprobe.scan` where you scan over layers:

```python
import jax, jax.numpy as jnp
import jaxprobe as jp
from jaxprobe import steer

def model(params, tokens):
    x = jp.tap("embed", params["embed"][tokens])
    def block(x, lp):
        x = x + jp.tap("attn_out", attention(x, lp))
        x = x + jp.tap("mlp_out", mlp(x, lp))
        return jp.tap("resid", x), None
    x, _ = jp.scan(block, x, params["layers"])      # was: jax.lax.scan
    return unembed(params, x)

# Capture: activations from scanned layers come out stacked, (n_layers, batch, seq, d).
logits, acts = jax.jit(lambda p, t: jp.run(model, p, t, capture="layers.resid"))(params, tokens)

# Patch: add a steering vector at layer 12 only, or zero-ablate every MLP.
u = steer.mean_difference(pos_acts, neg_acts)
logits, _ = jp.run(model, params, tokens, patch={"layers.resid": jp.at(12, steer.add(u, 4.0))})
logits, _ = jp.run(model, params, tokens, patch={"layers.mlp_out": jnp.zeros_like})

# Activation patching: run B with A's activations (stacked arrays are indexed by layer).
_, a = jp.run(model, params, tokens_a, capture="layers.mlp_out")
logits, _ = jp.run(model, params, tokens_b, patch={"layers.mlp_out": a["layers.mlp_out"]})
```

Outside `run`, `tap` returns its input and the model is unchanged. Names take glob patterns (`"layers.*"`) and
`capture=True` records every tap. `run` works inside `jit`, `grad` and `vmap`, and `jaxprobe.checkpoint` replaces
`jax.checkpoint`. Two rules follow from how it works (taps are trace-time side effects):

- Put `jit` around `run`, not around the model you pass to it. A jitted model's cached trace won't run its taps
  again.
- Scan and remat bodies must use `jaxprobe.scan` / `jaxprobe.checkpoint`, which carry captures out of the
  inner trace.

Both mistakes raise `TapError` rather than returning wrong or leaked values.

## Modules

- **core**: `tap`, `run`, `scan`, `checkpoint`, `scope`, `index`, `at`. It uses only the public JAX API, apart
  from one private call that detects a tap running in the wrong trace.
- **sae**: ReLU + L1, TopK with AuxK, BatchTopK and JumpReLU (straight-through threshold) sparse autoencoders.
  It provides losses, decoder renormalisation and a `splice` patch for loss-recovered evaluation. Metrics: FVU,
  loss recovered, density.
- **steer**: mean-difference (CAA) and ActAdd vectors; `add` and `ablate` patches; `orthogonalize`, which bakes
  an ablation into the weights; `sweep` over layer × coefficient; KL for damage.
- **dash** (planned): feature dashboards with top contexts, logit effects and density. Auto-interpretation
  through any OpenAI-compatible local server, with static HTML and Neuronpedia-compatible JSON.
- **trace**: attribution graphs (Ameisen et al. 2025) with per-layer transcoders. `attribute` replaces each MLP
  by its transcoder plus an error term, freezes the tapped nonlinearities (attention patterns, norm scales), and
  gets every edge of the resulting linear model from one batched `jax.vjp`. `prune` keeps the nodes and edges
  that carry most of the influence and reports replacement and completeness scores. `ablate_features` tests
  a graph on the real model. `to_json` and `write_graph` export circuit-tracer's graph format, which its viewer
  and Neuronpedia's read.
- **planned for sae**: streaming activation buffers, and export to the SAELens / Gemma Scope layout.

## Checked on a real model

On SmolLM2-135M (float32, AMD RX 7600 XT, JAX 0.12.1 on ROCm), with taps added to the
[learn-jax](https://github.com/joebarbere/learn-jax) Llama code:

- The residual stream and MLP outputs captured by `run` are bit-identical to an explicit loop, and the logits
  agree to 1.4e-5.
- An identity patch on every tap is a bit-exact no-op.
- Removing a direction with `orthogonalize` on the embedding, `wo` and `w_down` matches removing it at runtime with
  `ablate` to within 4.3e-4 in the logits, while the edit itself moves logits by up to 20.5.
- Patching one prompt's activations into a run on another reproduces the first prompt's logits (6e-5).
- `run` adds no measurable time over the hand-written forward pass (104 ms vs 105 ms for 8 × 128 tokens).
- `trace`, with 30 per-layer transcoders (8,192 features each) trained in
  [learn-jax module 48](https://github.com/joebarbere/learn-jax/tree/main/modules/48-tracing-circuits):
  - The frozen replacement model reproduces the real logits to about 1e-5.
  - Every node's incoming edges plus its constant part add up to its value, to 1e-4.
  - A graph of 9,000 nodes takes about 20 s.
  - Removing a graph's 30 most influential features drops P(" Texas") after "Dallas is a city in the state of"
    from 0.94 to 0.23. Removing 30 random active features leaves it at 0.94.
  - The exported JSON opens in circuit-tracer's viewer.

It grew out of the interpretability modules of [learn-jax](https://github.com/joebarbere/learn-jax).

## Development

```bash
uv sync
uv run pytest
```

MIT licensed.
