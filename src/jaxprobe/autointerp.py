"""Auto-interpretation: an LLM explains each feature from its top contexts, then the explanation is scored.

Reading thousands of dashboards by hand doesn't scale, so hand them to a chat model. `explain` shows it a
feature's top-activating contexts (from `dash.Collector`) with the firing tokens marked and asks what the feature
looks for. Then the explanation is tested on contexts the model hasn't seen (Paulo et al. 2024):

* **detection**: held-out top contexts of the feature, mixed with random windows where it does *not* fire; for
  each, does the feature fire anywhere in it? Tests whether the explanation says *which texts*.
* **fuzzing**: held-out top contexts with the firing tokens marked, or with as many other tokens marked; are the
  marked tokens the ones it fires on? Tests whether the explanation says *where*.

Both are balanced yes/no tasks, so 50% is chance. Always score a control alongside: `shuffled` gives every feature
another feature's explanation, which should score near 50%; if it doesn't, the test is too easy.

    llm = jaxprobe.autointerp.Client("http://localhost:1234/v1", "qwen/qwen3.5-9b",
                                     extra={"reasoning_effort": "none"}, cache="autointerp.jsonl")
    neg = jaxprobe.autointerp.random_windows(z, tokens, width=17)        # per batch while collecting; concatenate
    res = jaxprobe.autointerp.run(llm, stats, decode, features, negatives=neg)
    res["summary"]          # mean detection / fuzzing accuracy, for the real and the shuffled explanations

Any OpenAI-compatible server works (LM Studio, llama.cpp's server, vLLM, Ollama's /v1); so does any function
that maps a list of prompts to a list of replies, for tests or another API. Nothing here needs the model or a GPU.

Caveat: the held-out positives are the feature's *top* contexts, the easiest ones. Paulo et al. sample positives
across the activation range, which scores lower; treat these numbers as an upper bound on how well an
explanation covers the feature.

References: Bills et al. 2023, "Language models can explain neurons in language models"; Paulo et al. 2024,
"Automatically Interpreting Millions of Features in Large Language Models" (detection and fuzzing scoring;
EleutherAI's delphi).
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import re
import threading
import time
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from jaxprobe import dash

LEFT, RIGHT = "<<", ">>"


class Client:
    """A chat model behind an OpenAI-compatible endpoint: `client(prompts) -> replies`, `workers` requests at a time.

    `extra` is merged into every request body, e.g. {"reasoning_effort": "none"} to stop a thinking model (Qwen3.5
    on LM Studio) from spending its whole budget thinking. Replies are stripped of <think>…</think>. With `cache`
    (a .jsonl path) every reply is stored under a hash of the request, so re-running a script costs nothing."""

    def __init__(self, url: str, model: str, *, max_tokens: int = 256, temperature: float = 0.0,
                 extra: dict | None = None, workers: int = 4, timeout: float = 300.0, retries: int = 3,
                 cache: str | Path | None = None):
        self.url = url.rstrip("/") + "/chat/completions"
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.extra, self.workers, self.timeout, self.retries = dict(extra or {}), workers, timeout, retries
        self.cache_path = None if cache is None else Path(cache)
        self.cache: dict[str, str] = {}
        self.calls = 0
        self._lock = threading.Lock()
        if self.cache_path is not None and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.cache[r["key"]] = r["reply"]

    def _body(self, prompt: str) -> dict:
        return {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": self.max_tokens,
                "temperature": self.temperature, **self.extra}

    def _one(self, prompt: str) -> str:
        body = self._body(prompt)
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    msg = json.loads(r.read())["choices"][0]["message"]
                break
            except OSError:
                if attempt == self.retries:
                    raise
                time.sleep(2 ** attempt)
        reply = re.sub(r"<think>.*?</think>", "", msg.get("content") or "", flags=re.DOTALL).strip()
        with self._lock:
            self.calls += 1
            self.cache[key] = reply
            if self.cache_path is not None:
                with self.cache_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "reply": reply}, ensure_ascii=False) + "\n")
        return reply

    def __call__(self, prompts: Sequence[str]) -> list[str]:
        with cf.ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self._one, prompts))


# ---------------------------------------------------------------------------------------------------- formatting
def show(tokens: Sequence[str], marked: Sequence[bool] | None = None) -> str:
    """One context on one line: newlines as ⏎, runs of marked tokens wrapped in << >>."""
    out, inside = [], False
    for i, t in enumerate(tokens):
        m = bool(marked[i]) if marked is not None else False
        if m and not inside:
            out.append(LEFT)
        if not m and inside:
            out.append(RIGHT)
        inside = m
        out.append(t.replace("\n", "⏎"))
    if inside:
        out.append(RIGHT)
    return "".join(out)


def contexts(stats: dict, f: int, decode: Callable[[int], str]) -> list[dict]:
    """Feature f's top contexts from `dash.Collector().result()`: tokens (strings) and activations, best first."""
    return dash.card(stats, f, decode)["contexts"]


def _marks(acts, frac: float):
    a = np.asarray(acts, np.float32)
    return a >= frac * a.max() if a.max() > 0 else np.zeros(len(a), bool)


def random_windows(z, tokens, mask=None, *, width: int = 17, per_seq: int = 1, seed: int = 0):
    """Random windows to use as negatives: (windows (B·per_seq, width) int, fires (B·per_seq, n_features) bool),
    where fires says which features fire anywhere in each window. Call it on each batch while collecting (z: codes
    (B, T, n), tokens: (B, T)) and concatenate; `seed` should differ per batch."""
    z, tokens = np.asarray(z), np.asarray(tokens)
    B, T, _ = z.shape
    if mask is not None:
        z = np.where(np.asarray(mask, bool)[..., None], z, 0)
    rng = np.random.default_rng(seed)
    start = rng.integers(0, max(T - width, 0) + 1, (B, per_seq))
    idx = start[..., None] + np.arange(min(width, T))                                        # (B, per_seq, W)
    b = np.arange(B)[:, None, None]
    win = tokens[b, idx].reshape(-1, idx.shape[-1])
    fires = (z[b, idx] > 0).any(2).reshape(-1, z.shape[-1])
    return win.astype(np.int32), fires


# ---------------------------------------------------------------------------------------------------- explain
EXPLAIN = """We're studying features in a language model. Each feature looks for one particular thing in text.
Below are short excerpts where one feature fires strongly; the tokens where it fires are marked {left}like this{right}.
{effects}
{examples}

In one sentence of at most 15 words, what does this feature look for? Be specific; don't just list the marked words.
Answer with the description only."""


def explain(llm: Callable[[list[str]], list[str]], stats: dict, decode: Callable[[int], str], features: Sequence[int],
            *, n_examples: int = 8, mark: float = 0.3, effects: dict | None = None,
            template: str = EXPLAIN) -> dict[int, str]:
    """{feature: explanation}. Shown contexts are the even-ranked top contexts (0, 2, 4, …), so `score` can test
    on the odd ones. Tokens at or above `mark` × the context's peak activation are marked. With `effects` (from
    `dash.logit_effects`) the prompt also lists the tokens the feature promotes."""
    prompts, keep = [], []
    for f in features:
        ctx = contexts(stats, int(f), decode)[0::2][:n_examples]
        if not ctx:
            continue
        ex = "\n".join(f"{i + 1}. {show(c['tokens'], _marks(c['acts'], mark))}" for i, c in enumerate(ctx))
        eff = ""
        if effects is not None:
            eff = "Tokens this feature makes more likely next: " + ", ".join(
                repr(decode(int(i))) for i in effects["pos_ids"][f][:8]) + "\n"
        prompts.append(template.format(left=LEFT, right=RIGHT, effects=eff, examples=ex))
        keep.append(int(f))
    replies = llm(prompts)
    return {f: r.strip().splitlines()[0].strip() if r.strip() else "" for f, r in zip(keep, replies)}


# ---------------------------------------------------------------------------------------------------- score
DETECTION = """A feature of a language model has been described as: {explanation}

For each excerpt below, say whether the feature fires anywhere in it (1) or not (0).
{examples}

Answer with a list of {k} numbers such as [1, 0, 1] and nothing else."""

FUZZING = """A feature of a language model has been described as: {explanation}

In each excerpt below some tokens are marked {left}like this{right}. For each excerpt, say whether the marked tokens are
where this feature fires (1) or not (0).
{examples}

Answer with a list of {k} numbers such as [1, 0, 1] and nothing else."""


def _parse(reply: str, k: int):
    m = re.search(r"\[([^\]]*)\]", reply)
    vals = re.findall(r"[01]", m.group(1) if m else reply)
    return [int(v) for v in vals] if len(vals) == k else None


def _items(stats, f, decode, method, n_test, mark, negatives, rng, pool):
    test = contexts(stats, f, decode)[1::2][:n_test]
    items = []
    if method == "detection":                          # balanced: as many negatives as positives, or fewer of both
        if negatives is not None:
            win, fires = negatives
            cand = np.flatnonzero(~np.asarray(fires[:, f]))
            neg = [show([decode(int(t)) for t in win[i] if t >= 0]) for i in rng.permutation(cand)[:len(test)]]
        else:                                          # other features' contexts, not checked for this feature
            own = set(stats["top_seq"][f].tolist())
            cand = [c for c in pool if c[0] not in own]
            neg = [show(cand[i][1]) for i in rng.permutation(len(cand))[:len(test)]]
        items += [(show(c["tokens"]), 1) for c in test[:len(neg)]] + [(t, 0) for t in neg]
    elif method == "fuzzing":
        for j, c in enumerate(test):
            m = _marks(c["acts"], mark)
            if j % 2 == 0:
                items.append((show(c["tokens"], m), 1))
                continue
            off = np.flatnonzero(np.asarray(c["acts"]) <= 0)              # as many marks, where it doesn't fire
            if len(off) >= m.sum():
                wrong = np.zeros(len(m), bool)
                wrong[rng.choice(off, int(m.sum()), replace=False)] = True
                items.append((show(c["tokens"], wrong), 0))
        # balanced too: a feature that fires on every token has no tokens to mark wrongly, so no fuzzing test
        k = min(sum(y for _, y in items), sum(1 - y for _, y in items))
        seen = [0, 0]
        kept = []
        for t, y in items:
            if seen[y] < k:
                kept.append((t, y))
                seen[y] += 1
        items = kept
    else:
        raise ValueError(f"method must be 'detection' or 'fuzzing', got {method!r}")
    return [items[i] for i in rng.permutation(len(items))]


def score(llm: Callable[[list[str]], list[str]], stats: dict, decode: Callable[[int], str],
          explanations: dict[int, str], *, method: str = "detection", n_test: int = 8, batch: int = 5,
          mark: float = 0.3, negatives: tuple | None = None, seed: int = 0) -> dict[int, dict]:
    """{feature: {accuracy, n, tp, fp, tn, fn, failed}}. Positives are the odd-ranked top contexts (unseen by
    `explain`); detection negatives come from `negatives` = (windows, fires) from `random_windows`, keeping only
    windows where the feature doesn't fire. Without them, other features' top contexts are used unchecked, so a
    few "negatives" may really fire. Both tests are kept balanced: a feature that fires in (nearly) every window
    has no detection negatives and no tokens to mark wrongly, so it gets n = 0 and accuracy nan, and `summary`
    leaves it out. Excerpts go `batch` to a prompt;
    a reply that doesn't parse to the right number of 0/1s counts as `failed` and is left out of the accuracy."""
    pool = []
    if method == "detection" and negatives is None:
        for g in np.flatnonzero(stats["fired"] > 0)[:: max(1, int((stats["fired"] > 0).sum()) // 500)]:
            pool += [(c["seq"], c["tokens"]) for c in contexts(stats, int(g), decode)[:2]]
    tmpl = DETECTION if method == "detection" else FUZZING
    prompts, owners = [], []
    for f, expl in explanations.items():
        rng = np.random.default_rng([seed, int(f)])
        items = _items(stats, int(f), decode, method, n_test, mark, negatives, rng, pool)
        for s in range(0, len(items), batch):
            chunk = items[s:s + batch]
            ex = "\n".join(f"{i + 1}. {t}" for i, (t, _) in enumerate(chunk))
            prompts.append(tmpl.format(explanation=expl, examples=ex, k=len(chunk), left=LEFT, right=RIGHT))
            owners.append((int(f), [y for _, y in chunk]))
    replies = llm(prompts)
    out = {int(f): {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "failed": 0} for f in explanations}
    for (f, truth), reply in zip(owners, replies):
        got = _parse(reply, len(truth))
        r = out[f]
        if got is None:
            r["failed"] += len(truth)
            continue
        for y, p in zip(truth, got):
            r[("t" if y == p else "f") + ("p" if p else "n")] += 1
    for r in out.values():
        r["n"] = r["tp"] + r["fp"] + r["tn"] + r["fn"]
        r["accuracy"] = (r["tp"] + r["tn"]) / r["n"] if r["n"] else float("nan")
    return out


def shuffled(explanations: dict[int, str], seed: int = 0) -> dict[int, str]:
    """The control: every feature gets another feature's explanation (a random derangement)."""
    keys = list(explanations)
    if len(keys) < 2:
        raise ValueError("need at least two explanations to shuffle")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(keys))
    while (perm == np.arange(len(keys))).any():
        perm = rng.permutation(len(keys))
    return {k: explanations[keys[j]] for k, j in zip(keys, perm)}


def summary(scores: dict[int, dict]) -> dict:
    """Mean accuracy over features (with its standard error), and how many answers failed to parse."""
    acc = np.array([r["accuracy"] for r in scores.values() if r["n"]], np.float64)
    return {"accuracy": float(acc.mean()) if len(acc) else float("nan"),
            "stderr": float(acc.std(ddof=1) / np.sqrt(len(acc))) if len(acc) > 1 else float("nan"),
            "features": len(acc), "failed": int(sum(r["failed"] for r in scores.values()))}


def run(llm: Callable[[list[str]], list[str]], stats: dict, decode: Callable[[int], str], features: Sequence[int],
        *, negatives: tuple | None = None, effects: dict | None = None, n_examples: int = 8, n_test: int = 8,
        batch: int = 5, mark: float = 0.3, seed: int = 0) -> dict:
    """explain → detection and fuzzing scores, for the real explanations and for the shuffled control.
    Returns {"explanations", "detection", "fuzzing", "control": {"detection", "fuzzing"}, "summary"}."""
    expl = explain(llm, stats, decode, features, n_examples=n_examples, mark=mark, effects=effects)
    expl = {f: e for f, e in expl.items() if e}
    ctrl = shuffled(expl, seed)
    kw = {"n_test": n_test, "batch": batch, "mark": mark, "negatives": negatives, "seed": seed}
    out = {"explanations": expl, "control": {}}
    for method in ("detection", "fuzzing"):
        out[method] = score(llm, stats, decode, expl, method=method, **kw)
        out["control"][method] = score(llm, stats, decode, ctrl, method=method, **kw)
    out["summary"] = {m: summary(out[m]) for m in ("detection", "fuzzing")} | {
        f"control_{m}": summary(out["control"][m]) for m in ("detection", "fuzzing")}
    return out


def labels(result: dict, method: str = "detection") -> dict[int, str]:
    """Explanations with their score, for `dash.to_html(..., labels=...)` (no score where there was no test)."""
    out = {}
    for f, e in result["explanations"].items():
        acc = result.get(method, {}).get(f, {}).get("accuracy", float("nan"))
        out[f] = e if np.isnan(acc) else f"{e} ({method} {acc:.0%})"
    return out
