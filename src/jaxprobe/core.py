"""Capture and patch intermediate activations of a JAX function, using only the public JAX API.

Mark points in model code with `tap`:

    def block(x, p):
        h = jaxprobe.tap("attn_out", attention(x, p))
        return x + h

Outside `run`, `tap` returns its input unchanged and costs nothing. Inside `run`, it records the value and/or
replaces it:

    out, acts = jaxprobe.run(model, params, tokens, capture=["layers.attn_out"])
    out, _ = jaxprobe.run(model, params, tokens, patch={"layers.attn_out": jaxprobe.at(3, jnp.zeros_like)})

How it works. `tap` is a Python side effect at trace time: `run` sets a recording context, calls the function,
and returns what was recorded as ordinary outputs of the same trace. So `run` composes with `jit`, `grad`,
`vmap` and friends when it is called *inside* them (`jax.jit(lambda p, t: jaxprobe.run(model, p, t, ...))`).

Two rules follow from that:

1. Don't call an already-jitted function inside `run`: a cached trace doesn't run its taps again. `run` raises
   if a capture pattern matched no tap, which is what this mistake looks like.
2. A transform that traces its body separately (`lax.scan`, `jax.checkpoint`) must hand the captures out as
   outputs. Use `jaxprobe.scan` and `jaxprobe.checkpoint`, drop-in replacements for the JAX ones. Captures from
   inside a scan come out stacked along the scanned axis; names get the scan's name as a prefix.
"""
from __future__ import annotations

import contextlib
import dataclasses
import fnmatch
import functools
import threading
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import jax
import jax.numpy as jnp

__all__ = ["TapError", "at", "checkpoint", "index", "run", "scan", "scope", "tap"]

try:  # private, but the only way to tell which trace a tap runs in; without it the check is skipped
    from jax._src.core import get_opaque_trace_state as _trace_state
except ImportError:  # pragma: no cover
    def _trace_state():
        return None


class TapError(ValueError):
    """A capture or patch name matched no tap, or a patch returned the wrong shape."""


@dataclasses.dataclass
class _Frame:
    """One recording context: the outermost one comes from `run`, inner ones from `scan` / `checkpoint`."""

    prefix: str  # full name prefix for taps in this frame, e.g. "layers." inside scan(name="layers")
    capture: Callable[[str], bool]
    patches: list[tuple[str, Any]]  # (pattern, patch) on full names
    idx: tuple  # enclosing scan indices, outermost first (traced int32s)
    trace: Any = None  # the JAX trace this frame was opened in; taps must run in the same one
    captures: dict = dataclasses.field(default_factory=dict)
    seen: set = dataclasses.field(default_factory=set)  # full names of every tap reached


_state = threading.local()


def _stack() -> list[_Frame]:
    if not hasattr(_state, "frames"):
        _state.frames = []
    return _state.frames


def _top() -> _Frame | None:
    s = _stack()
    return s[-1] if s else None


@contextlib.contextmanager
def _push(frame: _Frame):
    _stack().append(frame)
    try:
        yield frame
    finally:
        _stack().pop()


def _matcher(capture) -> Callable[[str], bool]:
    if capture is True:
        return lambda name: True
    if not capture:
        return lambda name: False
    pats = [capture] if isinstance(capture, str) else list(capture)
    return lambda name: any(fnmatch.fnmatchcase(name, p) for p in pats)


def _patterns(capture) -> list[str]:
    if capture is True or not capture:
        return []
    return [capture] if isinstance(capture, str) else list(capture)


# ---------------------------------------------------------------- marking and reading taps
def tap(name: str, x):
    """Mark `x` as the activation `name`. Returns x, or its patched value inside `run`.

    x may be an array or a pytree of arrays. Captures record the value *after* any patch, i.e. what the rest of
    the model sees."""
    f = _top()
    if f is None:
        return x
    full = f.prefix + name
    if f.trace is not None and _trace_state() != f.trace:
        raise TapError(f"tap {full!r} ran inside a JAX transform that jaxprobe doesn't route (jax.jit, jax.vmap, "
                       "lax.scan, jax.checkpoint, ... called inside run). Call run inside the transform instead, "
                       "or use jaxprobe.scan / jaxprobe.checkpoint, which carry captures out")
    f.seen.add(full)
    for pat, p in f.patches:
        if fnmatch.fnmatchcase(full, pat):
            x = _apply(p, x, full, f.idx)
    if f.capture(full):
        if full in f.captures:
            raise TapError(f"tap {full!r} was reached twice in one trace; give each call site its own name "
                           "(or wrap repeated layers in jaxprobe.scan / jaxprobe.scope)")
        f.captures[full] = x
    return x


def index():
    """The current layer index (a traced int32) inside `jaxprobe.scan`, or None outside any scan.
    Inside nested scans it is the innermost one's index."""
    f = _top()
    return f.idx[-1] if f is not None and f.idx else None


def at(layer: int | Iterable[int], fn: Callable):
    """A patch that applies fn only at the given scan index (or indices) and leaves the other layers alone."""
    layers = (layer,) if isinstance(layer, int) else tuple(layer)

    def patch(x):
        i = index()
        if i is None:
            raise TapError("jaxprobe.at(...) patches only work on taps inside jaxprobe.scan")
        y = fn(x)
        hit = jnp.isin(i, jnp.asarray(layers))
        return jax.tree.map(lambda a, b: jnp.where(hit, b.astype(a.dtype), a), x, y)

    return patch


def _apply(p, x, name, idx):
    """A patch is a callable fn(x) -> x, or a value that replaces x. Inside a scan, a value with one more
    leading axis than x is indexed by the layer (so captures from one run can be patched into another)."""
    if callable(p):
        y = p(x)
    else:
        def pick(v, a):
            v = jnp.asarray(v)
            if idx and v.ndim == jnp.ndim(a) + 1:
                v = v[idx[-1]]
            return v
        y = jax.tree.map(pick, p, x) if jax.tree.structure(p) == jax.tree.structure(x) else pick(p, x)
    sx, sy = jax.tree.map(jnp.shape, x), jax.tree.map(jnp.shape, y)
    if sx != sy:
        raise TapError(f"patch for {name!r} changed the shape: {sx} -> {sy}")
    return jax.tree.map(lambda a, b: jnp.asarray(b).astype(jnp.result_type(a)), x, y)


# ---------------------------------------------------------------- running
def run(fn: Callable, *args, capture=(), patch: Mapping[str, Any] | None = None, strict: bool = True, **kwargs):
    """Call fn(*args, **kwargs) with taps recording and/or patched. Returns (fn's output, {name: activation}).

    capture: tap names to record. Glob patterns work ("layers.*", "*.mlp_out"); True records every tap.
    patch:   {name or pattern: fn(x) -> x, or a replacement value}. Patches see a tap's value before it's recorded.
    strict:  raise TapError if a capture or patch pattern matches no tap (usually a typo, or fn was jitted)."""
    patches = list((patch or {}).items())
    frame = _Frame(prefix="", capture=_matcher(capture), patches=patches, idx=(), trace=_trace_state())
    with _push(frame):
        out = fn(*args, **kwargs)
    if strict:
        _check(frame.seen, _patterns(capture), [p for p, _ in patches])
    return out, frame.captures


def _check(seen, capture_pats, patch_pats):
    for kind, pats in (("capture", capture_pats), ("patch", patch_pats)):
        for p in pats:
            if not any(fnmatch.fnmatchcase(n, p) for n in seen):
                hint = (" No taps ran at all: is fn already jitted (a cached trace skips its taps)?" if not seen
                        else f" Taps reached: {sorted(seen)}")
                raise TapError(f"{kind} pattern {p!r} matched no tap.{hint}")


def _child(parent: _Frame, name: str, idx: tuple) -> _Frame:
    """A frame for a nested trace (scan body, remat body) or a scope; created where its taps will run."""
    prefix = parent.prefix + (name + "." if name else "")
    return _Frame(prefix=prefix, capture=parent.capture, patches=parent.patches, idx=idx, trace=_trace_state())


def _merge(parent: _Frame, child: _Frame, captures: dict):
    parent.seen |= child.seen
    for k, v in captures.items():
        if k in parent.captures:
            raise TapError(f"tap {k!r} was captured twice; give the scans/scopes different names")
        parent.captures[k] = v


@contextlib.contextmanager
def scope(name: str):
    """Prefix the names of taps inside the block with `name.` (for repeated sub-modules called in a Python loop):

        for i, lp in enumerate(layers):
            with jaxprobe.scope(f"layer{i}"):
                x = block(x, lp)
    """
    f = _top()
    if f is None:
        yield
        return
    c = _child(f, name, f.idx)
    with _push(c):
        yield
    _merge(f, c, c.captures)


def scan(f: Callable, init, xs=None, length: int | None = None, *, name: str = "layers", **scan_kwargs):
    """`jax.lax.scan` that carries taps out of the body. Captures come out stacked along the scanned axis, named
    "<name>.<tap>"; inside the body `jaxprobe.index()` is the step number. Outside `run` it is plain lax.scan."""
    parent = _top()
    if parent is None:
        return jax.lax.scan(f, init, xs, length=length, **scan_kwargs)
    box = {}

    def body(carry, x):
        c, i = carry
        child = _child(parent, name, parent.idx + (i,))
        with _push(child):
            c, y = f(c, x)
        box["child"] = child
        return (c, i + 1), (y, child.captures)

    (c, _), (ys, caps) = jax.lax.scan(body, (init, jnp.int32(0)), xs, length=length, **scan_kwargs)
    _merge(parent, box["child"], caps)
    return c, ys


def checkpoint(f: Callable, **ckpt_kwargs):
    """`jax.checkpoint` (remat) that carries taps out of the rematerialised function."""
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        parent = _top()
        if parent is None:
            return jax.checkpoint(f, **ckpt_kwargs)(*args, **kwargs)
        box = {}

        def inner(*a, **k):
            child = _child(parent, "", parent.idx)
            with _push(child):
                out = f(*a, **k)
            box["child"] = child
            return out, child.captures

        out, caps = jax.checkpoint(inner, **ckpt_kwargs)(*args, **kwargs)
        _merge(parent, box["child"], caps)
        return out

    return wrapped
