# jaxprobe

Interpretability tools for JAX models: capture and patch activations, train sparse autoencoders, steer,
browse features, and trace circuits. PyTorch has TransformerLens, nnsight, SAELens and circuit-tracer;
JAX has had no maintained equivalent since Penzai was archived. jaxprobe aims to fill that gap.

**Status: pre-alpha.** The API below is the plan and nothing is implemented yet.

## Design

- **core**: `tap(name, x)` marks a point in your model code; `run(fn, *args, capture=..., patch=...)` returns
  the outputs plus the captured activations, with patches applied at the taps. It works under `jit`, `grad` and
  `vmap`, and stacks captures across `lax.scan`-ned layers. It uses only the public JAX API, with no jaxpr
  rewriting.
- **sae**: ReLU + L1, TopK (with AuxK), BatchTopK and JumpReLU sparse autoencoders, streamed from `core`
  captures. Metrics: L0, FVU, loss recovered, dead and dense latents. Export to the SAELens / Gemma Scope
  layout.
- **steer**: mean-difference (CAA) and ActAdd vectors, directional ablation and weight orthogonalization,
  and SAE-decoder steering. Sweeps are scored on behaviour *and* damage (KL, perplexity).
- **dash**: feature dashboards with top contexts, logit effects and density. Auto-interpretation through any
  OpenAI-compatible local server, with static HTML and Neuronpedia-compatible JSON.
- **trace**: transcoders, a replacement model with error nodes, and attribution graphs exported in the
  circuit-tracer / Neuronpedia graph format.

Build order: core → sae → (steer, dash) → trace. Reference models are Llama-architecture checkpoints
(SmolLM2) and Gemma 3, each checked against Hugging Face `transformers`.

It grew out of the interpretability modules of [learn-jax](https://github.com/joebarbere/learn-jax).

## Development

```bash
uv sync
uv run pytest
```

MIT licensed.
