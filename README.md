# jaxprobe

Interpretability tools for JAX models: capture and patch activations, train sparse autoencoders, steer,
browse features, and trace circuits. PyTorch has TransformerLens, nnsight, SAELens and circuit-tracer;
JAX has had no maintained equivalent since Penzai was archived. jaxprobe aims to fill that gap.

**Status: pre-alpha.** `core`, `sae`, `steer`, `dash`, `lens`, `monitor`, `trace`, `directions`, `io`, `probe`,
`reduce` and `batch` work and are tested. The API may still change.

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

# Sweeps without recompiling: bind takes the patch as an argument, and jaxprobe's patches are pytrees.
g = jax.jit(jp.bind(model, capture={"layers.resid": jp.reduce.last}))   # keep only the last position
for c in (1.0, 2.0, 4.0):
    logits, acts = g({"layers.resid": jp.at(12, steer.add(u, c))}, params, tokens)
sweep = jax.vmap(lambda c: g({"layers.resid": jp.at(12, steer.add(u, c))}, params, tokens)[0])(coeffs)
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

- **core**: `tap`, `run`, `scan`, `checkpoint`, `scope`, `index`, `at`, `bind`, `run_layer`. It uses only the
  public JAX API, apart from one private call that detects a tap running in the wrong trace.
- **sae**: ReLU + L1, TopK with AuxK, BatchTopK and JumpReLU (straight-through threshold) sparse autoencoders.
  It provides losses, decoder renormalisation and a `splice` patch for loss-recovered evaluation. Metrics: FVU,
  loss recovered, density.
- **steer**: mean-difference (CAA) and ActAdd vectors; `add` and `ablate` patches; `orthogonalize`, which bakes
  an ablation into the weights; `sweep` over layer × coefficient; KL for damage; `random_like`, a random vector
  with the same norm, as the control for "would any push this size do it?".
- **lens**: the logit lens and the tuned lens (Belrose et al. 2023) over any stack of residual-stream states.
  `stats` reads every layer one at a time (no layers × tokens × vocab array): KL to the final prediction,
  agreement, entropy, top tokens, the rank and log-probability of a target token, and the log-probability of
  token sets such as {" yes", " Yes", …}. `fit_tuned` trains the per-layer translators with the forward pass in
  the same compiled step. `dla` (direct logit attribution) splits a logit, or a logit difference, into what each
  component wrote to the residual stream, with the final norm's scale frozen, so the parts add up to it exactly.
  `rms_unembed` / `layer_unembed` build the readout for RMSNorm and LayerNorm models; any `h -> logits` callable
  works for the lens. `to_html` writes a layer × position grid of top tokens.
- **dash**: feature dashboards for SAEs and transcoders. `Collector` streams codes over a dataset in one
  compiled call per batch: firing density, a log-activation histogram, top contexts (one per sequence, with
  the activations around the peak) and, optionally, how often every pair of features fires together.
  `logit_effects`, `neighbours` (decoder cosine), `coactivation` (φ of firing) and `feature_map` (PCA, or UMAP
  with the `umap` extra). `to_html` writes one self-contained page: a clickable map of the features and a card
  per feature. `to_neuronpedia` writes SAEDashboard's Neuronpedia JSON layout. Auto-interpretation is planned.
- **monitor**: watch a model's internals while it trains or serves. `health` (per-token RMS and its maximum,
  largest |x|, outlier dimensions, NaN/Inf), `attention` (entropy per head, peak weight, attention-sink share),
  `units` (dead and active fractions) and `project` (scores along a direction or probe) are capture reducers, so
  they run inside the compiled step and return a few numbers per layer; `logit_stats` covers the output (max
  logit, z and z², entropy). Nothing calls back to the host: fetch every N steps. `Running` keeps a running mean
  and variance of any metrics pytree and flags values far outside their own history. `drift` compares two
  `dash.Collector` results feature by feature, with a variance that counts passages rather than tokens (SAE
  features fire in bursts). `JsonlWriter`, `read_jsonl`, and `scalars` to flatten metrics for TensorBoard or W&B.
- **trace**: attribution graphs (Ameisen et al. 2025) with per-layer transcoders. `attribute` replaces each MLP
  by its transcoder plus an error term, freezes the tapped nonlinearities (attention patterns, norm scales), and
  gets every edge of the resulting linear model from one batched `jax.vjp`. `prune` keeps the nodes and edges
  that carry most of the influence and reports replacement and completeness scores. `ablate_features` tests
  a graph on the real model. `to_json` and `write_graph` export circuit-tracer's graph format, which its viewer
  and Neuronpedia's read.
- **reduce**: reducers for `capture={pattern: reducer}` (last position, chosen positions, masked mean, weighted
  batch sum). They run inside the scan body, so only the reduced value is stacked and returned.
- **batch**: length-bucketed, fixed-shape, left-padded batches with masks and weights, for few compiles and
  little padding.
- **directions**: the refusal-direction recipe of Arditi et al. 2024. `collect` streams difference-in-means
  statistics over many prompts (optionally spilling every row to disk, resumably); `candidates`;
  `refusal_score` (first-token log-odds, no generation); `arditi_score`, `grid` and `select` for choosing a layer
  and position; `abliterate` (globs over parameter paths) and `check_edit`; `hf_edit` for Hugging Face names.
  `pca`: the principal axis of paired differences, the representation-engineering recipe (Zou et al. 2023; the
  repeng library), per layer and position.
- **probe**: logistic-regression probes on activations (`fit_logistic`, `score`, `direction`, `auc`).
- **io**: safetensors without extra dependencies: `save`/`load` pytrees (sharded, memory-mapped, partial loads,
  atomic writes); `edit_checkpoint`, which edits a Hugging Face checkpoint one tensor at a time; `stream_layers`,
  which runs a model one layer at a time from disk with prefetch, with the same tap names as `scan`; `Spill`,
  a resumable on-disk activation array.
- **planned for sae**: export to the SAELens / Gemma Scope layout.

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

0.6.0, `monitor` on SmolLM2-135M (bf16) and module 47's TopK SAE:

- `health` and `attention` match a brute-force computation from the captured activations (1e-7 and 2e-5).
  They show the model's known attention sink: by block 16 the largest token RMS is about 858 against a
  typical 13–28, and later layers put 0.48–0.79 of their attention on the first token.
- Monitoring every layer's residual stream and attention adds 16% to a forward pass (25.2 vs 21.8 ms for
  8 × 128 tokens): 485 numbers per batch, one 4.9 kB JSON line.
- `Running` over 40 clean batches, then 40 more: 1 value beyond z = 8 out of 19,400 (15 beyond z = 6). A random
  push at block 12 of half the typical token norm raised 76 flags, all in blocks 12 and above; random tokens
  raised 253. Batches from another domain raised 25–30: health statistics are for numerical trouble, not content.
- `drift` between random halves of the detective windows: no feature flagged (largest |z| 3.7 of 8,192). The
  first test, a token-level binomial, flagged about 2,300 there. Between the two halves of the text in order:
  549 more and 728 less active (top z 23), since they are different parts of the stories. Against Shakespeare:
  567 more, 1,437 less, 36 gone (top z 63), and twice the median histogram divergence (0.095 vs 0.042 bits).

0.5.0, `lens` on SmolLM2-135M, against [learn-jax module 47](https://github.com/joebarbere/learn-jax/tree/main/modules/47-reading-activations)'s
own lens code (bf16):

- Logit-lens and tuned-lens KL and top-1 agreement for all 31 layers on module 47's validation windows match
  its saved results to within 5% (the largest gap is on a top-1 share near 0.1; KL agrees to the second decimal).
- `fit_tuned` with module 47's recipe (Adam, warmup 50, cosine to 1e-4, 1,000 steps of 16 × 128 tokens)
  retraces its loss curve (mean KL 17.23 → 2.81 → 2.05 → 1.63 at steps 0/100/200/300, 1.52 at the end) at the
  same speed, 0.99 s per step.
- "The capital of France is the city of": the logit lens first ranks " Paris" top at layer 27 of 30, the tuned
  lens at layer 24. `dla` puts the largest direct push on the attention of block 24 (+4.4) and the MLPs of
  blocks 26–29 (+1.4 to +2.0). In bf16 the parts sum to 9.68 against the model's 9.63, because the model's own
  residual additions round; in the float32 tests they agree to 1e-4.

`examples/introspection.py`: a small replication of "Latent Introspection" (Pearson-Vogel et al. 2026, arXiv
2602.20031) on Qwen2.5-1.5B-Instruct (the table is without the explanatory paragraph), using `directions.pca`, `steer.add` with a position mask, `random_like`
and `lens.stats` with token sets. Concept vectors are added at the middle third of the layers on the first turn
only; the model is then asked whether it detects an injected thought, and P(yes) is read from the logits.

| strength (× residual norm) | P(yes), concept | P(yes), random vector | P(yes), "Is the Earth flat?" & co. |
|---|---|---|---|
| 0 | 0.15 | 0.15 | 0.16 |
| 0.25 | 0.26 | 0.18 | 0.14 |
| 0.5 | 0.34 | 0.36 | 0.20 |
| 1 | 0.66 | 0.59 | 0.50 |
| 2 | 0.85 | 0.69 | 0.84 |

The vectors steer strongly at every strength (log-probability of the concept's words +5 nats after a neutral
request), but at 1.5B most of the "yes" is generic: a random vector of the same size and questions whose answer
is always "no" move nearly as much. Only at the weakest strength does the concept vector beat both controls
(0.26 against 0.18 and 0.14), on 9 concepts and one prompt; with the paper's kind of explanatory paragraph added
to the first turn that margin shrinks to 0.03–0.04. The paper's models were 32–72B; this is a demonstration of
the tools and controls, not a test of their claim.

0.4.0, `dash` on learn-jax module 47's TopK SAE (8,192 features, residual stream after block 15 of 30) over
its 430,475 validation tokens:

- Collecting density, histograms and the top 16 contexts takes 20 s (38 s with co-activation counts); the
  firing counts equal an independent count on the same codes, and the logit effects match module 47's own
  dashboard exactly.
- Features specific to one of the three text domains (detective fiction, astronomy abstracts, Shakespeare)
  had their 10 nearest map neighbours in the same domain 88% of the time on a UMAP of co-activation, 77% on a
  PCA of co-activation and 75% on a UMAP of decoder directions, against 66% by chance; a PCA of decoder
  directions is one round blob (their median best cosine is only 0.26). UMAP takes about a minute.
- The page for all 8,010 live features is 13 MB and opens without a network connection.

0.3.0 on Qwen2.5-1.5B-Instruct (bf16 unless stated, same GPU, 12 harmful and 12 harmless prompts):

- `directions.collect` over both sets takes 1.5 s; `grid` scores 140 candidates (28 layers × 5 positions, four
  forward passes each) in 66 s; `select` picks layer 11 (bypass −1.83, induce +3.47, KL 0.095).
- `abliterate` with that direction: greedy replies that open with "I'm sorry" drop from 12 of 12 to 3 of 12, the
  mean refusal score from +3.59 to −1.85, and harmless answers are unchanged (KL 0.092).
- Weight edit vs runtime ablation, float32: 1.5e-4 in the logits. (Storing the edited weights in bf16 leaves
  0.3 of difference: rounding puts some of the direction back.)
- `edit_checkpoint` with `hf_edit` rewrites the 3.1 GB checkpoint in 6.6 s, editing 57 tensors; the result
  matches the in-memory edit to bf16 rounding.
- `stream_layers` from the checkpoint matches the scanned model to 6e-5 (float32, SmolLM2-360M); in bf16 the
  same top token, 4.6 s for 28 layers.
- A 16-point steering sweep: 6.4 s with a new closure per point (16 compiles), 2.8 s with `bind` (one compile).
- `capture={"layers.resid": reduce.last}` returns 1.0 MB instead of 66 MB.

It grew out of the interpretability modules of [learn-jax](https://github.com/joebarbere/learn-jax).

## Development

```bash
uv sync
uv run pytest
```

MIT licensed.
