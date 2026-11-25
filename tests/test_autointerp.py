"""0.7.0: auto-interpretation (explain, detection and fuzzing scoring, shuffled control, OpenAI-compatible client)."""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from jaxprobe import autointerp as ai
from jaxprobe import dash

N, VOCAB, B, T = 6, 40, 64, 32


def decode(i):
    return f" t{int(i)}"


def _data(seed=0):
    """Feature f fires on token 10 + f (strength 1–3); tokens are random, so each fires on ~1/40 of tokens."""
    rng = np.random.default_rng(seed)
    tok = rng.integers(0, VOCAB, (B, T))
    z = np.zeros((B, T, N), np.float32)
    for f in range(N):
        z[..., f] = (tok == 10 + f) * rng.uniform(1, 3, (B, T))
    return tok, z


def _stats_and_negatives():
    col = dash.Collector(N, top=16, before=6, after=2)
    negs = []
    for s in range(3):
        tok, z = _data(s)
        col.update(z, tok)
        negs.append(ai.random_windows(z, tok, width=9, per_seq=2, seed=s))
    return col.result(), (np.concatenate([w for w, _ in negs]), np.concatenate([f for _, f in negs]))


def oracle(prompts):
    """A fake LLM that knows the rule: it explains with the marked token, and answers by looking for it."""
    out = []
    for p in prompts:
        if p.startswith("We're studying"):
            out.append("the token " + re.search(r"<<\s*(t\d+)\s*>>", p).group(1))
            continue
        want = re.search(r"described as: the token (t\d+)", p).group(1)
        lines = re.findall(r"^\d+\. (.*)$", p, re.MULTILINE)
        if "marked" in p.split("\n")[2]:                                  # fuzzing: is it inside the marks?
            ans = [int(any(re.search(rf"\b{want}\b", m) for m in re.findall(r"<<(.*?)>>", ln))) for ln in lines]
        else:
            ans = [int(bool(re.search(rf"\b{want}\b", ln))) for ln in lines]
        out.append(json.dumps(ans))
    return out


def test_random_windows():
    tok, z = _data(0)
    win, fires = ai.random_windows(z, tok, width=9, per_seq=3, seed=1)
    assert win.shape == (B * 3, 9) and fires.shape == (B * 3, N)
    for f in range(N):
        np.testing.assert_array_equal(fires[:, f], (win == 10 + f).any(1))


def test_show_and_parse():
    assert ai.show([" a", " b", " c", "\n", " d"], [0, 1, 1, 0, 1]) == " a<< b c>>⏎<< d>>"
    assert ai._parse("Sure: [1, 0, 1]", 3) == [1, 0, 1]
    assert ai._parse("[1, 0]", 3) is None


def test_oracle_scores_perfectly_and_the_control_does_not():
    stats, neg = _stats_and_negatives()
    res = ai.run(oracle, stats, decode, range(N), negatives=neg, batch=4)
    assert res["explanations"] == {f: f"the token t{10 + f}" for f in range(N)}
    for m in ("detection", "fuzzing"):
        assert res["summary"][m]["accuracy"] == 1.0 and res["summary"][m]["failed"] == 0
        assert res["summary"][f"control_{m}"]["accuracy"] < 0.75
        assert all(r["n"] == 16 if m == "detection" else r["n"] == 8 for r in res[m].values())
    assert all(v != res["explanations"][f] for f, v in ai.shuffled(res["explanations"]).items())
    lab = ai.labels(res)
    assert lab[0] == "the token t10 (detection 100%)"


def test_unchecked_negatives_and_failures():
    stats, _ = _stats_and_negatives()
    expl = {f: f"the token t{10 + f}" for f in range(N)}
    r = ai.score(oracle, stats, decode, expl, method="detection")              # negatives: other features' contexts
    assert ai.summary(r)["accuracy"] > 0.9
    bad = ai.score(lambda ps: ["no idea"] * len(ps), stats, decode, expl, method="fuzzing")
    s = ai.summary(bad)
    assert s["failed"] == 8 * N and s["features"] == 0
    tok, z = _data(0)
    win, fires = ai.random_windows(z, tok, width=9, per_seq=2)
    fires[:, 2] = True                                                          # feature 2 fires everywhere
    r = ai.score(oracle, stats, decode, expl, method="detection", negatives=(win, fires))
    assert r[2]["n"] == 0 and np.isnan(r[2]["accuracy"]) and r[3]["n"] == 16
    assert ai.summary(r)["features"] == N - 1
    dense = dict(stats, top_acts=np.where(stats["top_tokens"] >= 0, 1.0, 0.0))      # fires on every token
    r = ai.score(oracle, dense, decode, expl, method="fuzzing")
    assert all(v["n"] == 0 for v in r.values())
    assert ai.labels({"explanations": expl, "fuzzing": r}, "fuzzing")[0] == "the token t10"
    with pytest.raises(ValueError):
        ai.score(oracle, stats, decode, expl, method="simulation")


def test_client_http_cache_and_think():
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            reply = {"choices": [{"message": {"content": "<think>hmm</think>\n" + body["messages"][0]["content"][::-1]}}]}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            url = f"http://127.0.0.1:{srv.server_port}/v1/"
            c = ai.Client(url, "m", extra={"reasoning_effort": "none"}, cache=f"{d}/c.jsonl", workers=2)
            assert c(["abc", "xyz", "abc"]) in (["cba", "zyx", "cba"],)
            assert seen[0]["reasoning_effort"] == "none" and seen[0]["model"] == "m" and seen[0]["temperature"] == 0
            n = len(seen)
            c2 = ai.Client(url, "m", extra={"reasoning_effort": "none"}, cache=f"{d}/c.jsonl")
            assert c2(["abc", "xyz"]) == ["cba", "zyx"] and len(seen) == n and c2.calls == 0   # all from the cache
    finally:
        srv.shutdown()
